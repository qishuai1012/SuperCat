"""
智能路由决策系统
负责分析查询特征并选择最优的处理策略和参数
核心功能：
1. 分析用户问题类型、复杂度
2. 调用大模型进行智能路由决策
3. 提供规则路由作为兜底保证稳定性
4. 动态调整检索参数
5. 全局单例 + 线程安全
"""

# ==================== 导入依赖库 ====================
from typing import Literal, TypedDict, List, Optional, Dict, Any
import os
import json
import logging
import re
import threading

# Langchain 模型初始化工具
from langchain.chat_models import init_chat_model
# 加载环境变量（.env 文件）
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 日志记录器
logger = logging.getLogger(__name__)

# 从环境变量读取大模型配置
API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# 导入之前定义的枚举与数据类（策略、复杂度、路由决策）
from query_understanding.types import QueryComplexity, RouteDecision, RouteStrategy


# ==================== 智能路由器核心类 ====================
class IntelligentRouter:
    """
    智能路由器 - 基于查询特征和历史表现选择最优策略
    工作模式：
    1. 优先使用大模型做智能决策
    2. 模型不可用时自动切换规则路由
    3. 输出标准 RouteDecision 数据结构
    """

    def __init__(self):
        # 路由决策大模型
        self.router_model = None
        # 线程锁：保证模型初始化安全
        self._lock = threading.Lock()
        # 初始化路由模型
        self._init_router_model()

        # 路由决策提示词：告诉大模型如何分析问题并返回策略
        self.routing_prompt = """
        你是一个智能路由决策器，负责分析用户查询并选择最优的处理策略。

        请分析以下查询，并选择最适合的策略：

        查询: {query}
        上下文: {context}
        历史对话长度: {history_length}

        策略选项:
        1. direct - 直接检索（适用于简单、明确的问题）
        2. step_back - 退步查询（适用于包含具体细节的问题）
        3. hyde - 假设文档生成（适用于概念性或定义性问题）
        4. complex - 复杂策略（适用于多步骤或综合性问题）
        5. parallel - 并行处理（适用于可以分解的复杂问题）

        复杂度评估:
        1. simple - 简单查询，单步即可解决
        2. moderate - 中等复杂度，需要策略优化
        3. complex - 复杂查询，需要多步处理

        请返回JSON格式决策：
        {{
            "strategy": "选择的策略",
            "complexity": "复杂度级别",
            "reasoning": "选择理由",
            "top_k": 检索数量,
            "merge_threshold": 合并阈值,
            "needs_decomposition": true/false,
            "parallel_paths": 并行路径数
        }}
        """

    def _init_router_model(self):
        """初始化路由决策大模型
        只有配置了 API_KEY、MODEL、BASE_URL 才会初始化
        """
        if API_KEY and MODEL and BASE_URL:
            try:
                # 初始化兼容 OpenAI 接口的大模型
                self.router_model = init_chat_model(
                    model=MODEL,
                    model_provider="openai",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    temperature=0.1,        # 低温度：输出更稳定、确定
                    stream_usage=True,      # 开启 token 使用统计
                )
                logger.info("路由模型初始化成功")
            except Exception as e:
                logger.warning(f"路由模型初始化失败: {e}")
                self.router_model = None

    def _analyze_query_features(self, query: str) -> Dict[str, Any]:
        """分析查询特征：提取问题的各类属性，用于规则判断
        返回：长度、是否概念问题、是否流程问题、是否复合问题等
        """
        features = {
            'length': len(query),  # 查询长度
            'has_specific_terms': any(term in query.lower() for term in ['什么', '如何', '为什么', '请', '告诉']),
            'has_concrete_entities': self._detect_concrete_entities(query),  # 是否有时间、地点、人名等实体
            'is_conceptual': self._is_conceptual_query(query),  # 是否是概念/定义类问题
            'is_procedural': self._is_procedural_query(query),  # 是否是步骤/方法类问题
            'has_multiple_questions': query.count('？') + query.count('?') > 1,  # 是否包含多个问题
            'is_compound': self._is_compound_query(query)  # 是否是复合问题（并且、同时、以及）
        }
        return features

    def _detect_concrete_entities(self, query: str) -> bool:
        """检测查询中是否包含具体实体：时间、地点、人名、公司、产品等"""
        concrete_indicators = ['年', '月', '日', '时间', '地点', '人名', '公司', '产品']
        return any(indicator in query for indicator in concrete_indicators)

    def _is_conceptual_query(self, query: str) -> bool:
        """判断是否为概念性查询：定义、是什么、什么意思、解释"""
        conceptual_indicators = ['定义', '概念', '含义', '是什么', '什么意思', '解释']
        return any(indicator in query for indicator in conceptual_indicators)

    def _is_procedural_query(self, query: str) -> bool:
        """判断是否为过程/方法类查询：步骤、怎么、如何、流程"""
        procedural_indicators = ['步骤', '流程', '方法', '怎么', '如何', '怎样', '过程']
        return any(indicator in query for indicator in procedural_indicators)

    def _is_compound_query(self, query: str) -> bool:
        """判断是否为复合查询：包含并且、还有、同时、以及等连接词"""
        compound_indicators = ['并且', '还有', '另外', '同时', '以及', '还有']
        return any(indicator in query for indicator in compound_indicators)

    def _rule_based_routing(self, query: str, context: Dict[str, Any]) -> RouteDecision:
        """基于规则的 fallback 路由策略
        当模型不可用/报错时，使用规则硬编码保证系统不崩溃
        根据问题特征自动匹配策略
        """
        features = self._analyze_query_features(query)

        # 复合问题 / 多个问号 → 并行策略 + 复杂级别
        if features['is_compound'] or features['has_multiple_questions']:
            return RouteDecision(
                strategy=RouteStrategy.PARALLEL,
                query_complexity=QueryComplexity.COMPLEX,
                needs_decomposition=True,
                parallel_paths=2,
                top_k=7
            )
        # 概念问题 → HYDE 策略
        elif features['is_conceptual']:
            return RouteDecision(
                strategy=RouteStrategy.HYDE,
                query_complexity=QueryComplexity.MEDIUM,
                top_k=5
            )
        # 包含具体实体 → STEP_BACK 退步抽象策略
        elif features['has_concrete_entities']:
            return RouteDecision(
                strategy=RouteStrategy.STEP_BACK,
                query_complexity=QueryComplexity.MEDIUM,
                top_k=6
            )
        # 过程/方法问题 → COMPLEX 复杂策略
        elif features['is_procedural']:
            return RouteDecision(
                strategy=RouteStrategy.COMPLEX,
                query_complexity=QueryComplexity.MEDIUM,
                top_k=5
            )
        # 其他 → 直接检索策略
        else:
            return RouteDecision(
                strategy=RouteStrategy.DIRECT,
                query_complexity=QueryComplexity.SIMPLE,
                top_k=4
            )

    def route_query(self, query: str, context: Dict[str, Any] = None) -> RouteDecision:
        """
        智能路由决策主函数（外部调用入口）
        :param query: 用户问题
        :param context: 上下文（对话历史等）
        :return: RouteDecision 路由决策对象
        """
        # 上下文安全初始化
        if context is None:
            context = {}

        if not isinstance(context, dict):
            context = {}

        # 无模型 → 直接使用规则路由
        if not self.router_model:
            return self._rule_based_routing(query, context)

        try:
            # 获取历史对话长度
            history = context.get('history', [])
            history_length = len(history) if isinstance(history, list) else 0

            # 格式化提示词
            routing_input = self.routing_prompt.format(
                query=query,
                context=json.dumps(context, ensure_ascii=False),
                history_length=history_length
            )

            # 调用大模型获取路由决策
            response = self.router_model.invoke(routing_input)
            # 解析模型返回的 JSON
            decision_data = self._parse_routing_response(response.content)
            # 构建标准 RouteDecision 对象
            return self._create_route_decision(decision_data)

        # 异常捕获：模型调用失败 → 自动降级为规则路由
        except Exception as e:
            logger.warning(f"智能路由决策失败，使用规则路由: {e}")
            return self._rule_based_routing(query, context)

    def _parse_routing_response(self, response_content: str) -> Dict[str, Any]:
        """解析路由模型响应：从返回文本中提取 JSON 结构
        支持大模型返回内容带多余文字
        """
        try:
            # 正则提取 {...} 格式 JSON
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"解析路由响应失败: {e}")

        # 解析失败 → 返回默认决策
        return {
            "strategy": "direct",
            "complexity": "simple",
            "reasoning": "fallback决策",
            "top_k": 5,
            "merge_threshold": 2,
            "needs_decomposition": False,
            "parallel_paths": 1
        }

    def _create_route_decision(self, decision_data: Dict[str, Any]) -> RouteDecision:
        """将模型返回的字典 → 转为标准 RouteDecision 类型对象
        做字符串 → 枚举的映射
        """
        try:
            strategy_str = decision_data.get("strategy", "direct")
            complexity_str = decision_data.get("complexity", "simple")

            # 字符串 → 枚举映射表
            strategy_map = {
                "direct": RouteStrategy.DIRECT,
                "step_back": RouteStrategy.STEP_BACK,
                "hyde": RouteStrategy.HYDE,
                "complex": RouteStrategy.COMPLEX,
                "parallel": RouteStrategy.PARALLEL
            }

            complexity_map = {
                "simple": QueryComplexity.SIMPLE,
                "moderate": QueryComplexity.MEDIUM,
                "medium": QueryComplexity.MEDIUM,
                "complex": QueryComplexity.COMPLEX,
                "complex_light": QueryComplexity.COMPLEX_LIGHT,
                "complex_heavy": QueryComplexity.COMPLEX_HEAVY
            }

            # 获取映射后的枚举
            strategy = strategy_map.get(strategy_str, RouteStrategy.DIRECT)
            complexity = complexity_map.get(complexity_str, QueryComplexity.SIMPLE)

            # 构建并返回路由决策对象
            return RouteDecision(
                strategy=strategy,
                query_complexity=complexity,
                top_k=decision_data.get("top_k", 5),
                merge_threshold=decision_data.get("merge_threshold", 2),
                needs_decomposition=decision_data.get("needs_decomposition", False),
                parallel_paths=decision_data.get("parallel_paths", 1),
                retrieval_params={"reasoning": decision_data.get("reasoning", "")}
            )

        # 异常：返回默认安全决策
        except Exception as e:
            logger.warning(f"创建路由决策失败: {e}")
            return RouteDecision(
                strategy=RouteStrategy.DIRECT,
                query_complexity=QueryComplexity.SIMPLE
            )

    def adjust_parameters_dynamically(self,
                                    base_decision: RouteDecision,
                                    performance_history: List[Dict[str, Any]]) -> RouteDecision:
        """
        基于历史表现动态调整检索参数（高级优化）
        根据：响应速度、相关性分数 自动优化 top_k、策略、并行路径
        """
        # 无历史数据 → 直接返回原决策
        if not performance_history:
            return base_decision

        try:
            # 计算平均响应时间、平均相关性得分
            avg_response_time = sum(h.get('response_time', 1.0) for h in performance_history) / len(performance_history)
            avg_relevance_score = sum(h.get('relevance_score', 0.5) for h in performance_history) / len(performance_history)

            # 复制基础决策
            adjusted_decision = RouteDecision(
                strategy=base_decision.strategy,
                query_complexity=base_decision.query_complexity,
                top_k=base_decision.top_k,
                merge_threshold=base_decision.merge_threshold,
                needs_decomposition=base_decision.needs_decomposition,
                parallel_paths=base_decision.parallel_paths
            )

            # 相关性低 → 多召回一些文档
            if avg_relevance_score < 0.6:
                adjusted_decision.top_k = min(base_decision.top_k + 2, 10)

            # 响应过慢 → 减少召回数量
            if avg_response_time > 3.0:
                adjusted_decision.top_k = max(base_decision.top_k - 1, 3)

            # 效果好但慢 → 切换并行策略提升速度
            if avg_relevance_score > 0.8 and avg_response_time > 2.0:
                if base_decision.strategy == RouteStrategy.DIRECT:
                    adjusted_decision.strategy = RouteStrategy.PARALLEL
                    adjusted_decision.parallel_paths = 2

            return adjusted_decision

        # 异常：返回原决策
        except Exception as e:
            logger.warning(f"动态参数调整失败: {e}")
            return base_decision


# ==================== 全局单例模式（保证系统高效） ====================
_intelligent_router = None
_router_lock = threading.Lock()

def get_intelligent_router() -> IntelligentRouter:
    """获取全局智能路由器实例
    全局只创建一个，节约内存、避免重复加载模型
    线程安全
    """
    global _intelligent_router
    with _router_lock:
        if _intelligent_router is None:
            _intelligent_router = IntelligentRouter()
    return _intelligent_router


# ==================== 外部便捷调用函数 ====================
def route_query(query: str, context: Dict[str, Any] = None) -> RouteDecision:
    """便捷函数：一行代码完成路由查询
    外部系统直接使用，无需关心内部实现
    """
    router = get_intelligent_router()
    return router.route_query(query, context)