"""
智能路由决策系统
负责分析查询特征并选择最优的处理策略和参数
"""

from typing import Literal, TypedDict, List, Optional, Dict, Any
import os
import json
import logging
import re
import threading

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


from query_understanding.types import QueryComplexity, RouteDecision, RouteStrategy


class IntelligentRouter:
    """
    智能路由器 - 基于查询特征和历史表现选择最优策略
    """

    def __init__(self):
        self.router_model = None
        self._lock = threading.Lock()
        self._init_router_model()

        # 策略选择提示词
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
        """初始化路由决策模型"""
        if API_KEY and MODEL and BASE_URL:
            try:
                self.router_model = init_chat_model(
                    model=MODEL,
                    model_provider="openai",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    temperature=0.1,
                    stream_usage=True,
                )
                logger.info("路由模型初始化成功")
            except Exception as e:
                logger.warning(f"路由模型初始化失败: {e}")
                self.router_model = None

    def _analyze_query_features(self, query: str) -> Dict[str, Any]:
        """分析查询特征"""
        features = {
            'length': len(query),
            'has_specific_terms': any(term in query.lower() for term in ['什么', '如何', '为什么', '请', '告诉']),
            'has_concrete_entities': self._detect_concrete_entities(query),
            'is_conceptual': self._is_conceptual_query(query),
            'is_procedural': self._is_procedural_query(query),
            'has_multiple_questions': query.count('？') + query.count('?') > 1,
            'is_compound': self._is_compound_query(query)
        }
        return features

    def _detect_concrete_entities(self, query: str) -> bool:
        """检测具体实体"""
        concrete_indicators = ['年', '月', '日', '时间', '地点', '人名', '公司', '产品']
        return any(indicator in query for indicator in concrete_indicators)

    def _is_conceptual_query(self, query: str) -> bool:
        """判断是否为概念性查询"""
        conceptual_indicators = ['定义', '概念', '含义', '是什么', '什么意思', '解释']
        return any(indicator in query for indicator in conceptual_indicators)

    def _is_procedural_query(self, query: str) -> bool:
        """判断是否为过程性查询"""
        procedural_indicators = ['步骤', '流程', '方法', '怎么', '如何', '怎样', '过程']
        return any(indicator in query for indicator in procedural_indicators)

    def _is_compound_query(self, query: str) -> bool:
        """判断是否为复合查询"""
        compound_indicators = ['并且', '还有', '另外', '同时', '以及', '还有']
        return any(indicator in query for indicator in compound_indicators)

    def _rule_based_routing(self, query: str, context: Dict[str, Any]) -> RouteDecision:
        """基于规则的fallback路由策略"""
        features = self._analyze_query_features(query)

        if features['is_compound'] or features['has_multiple_questions']:
            return RouteDecision(
                strategy=RouteStrategy.PARALLEL,
                query_complexity=QueryComplexity.COMPLEX,
                needs_decomposition=True,
                parallel_paths=2,
                top_k=7
            )
        elif features['is_conceptual']:
            return RouteDecision(
                strategy=RouteStrategy.HYDE,
                query_complexity=QueryComplexity.MEDIUM,
                top_k=5
            )
        elif features['has_concrete_entities']:
            return RouteDecision(
                strategy=RouteStrategy.STEP_BACK,
                query_complexity=QueryComplexity.MEDIUM,
                top_k=6
            )
        elif features['is_procedural']:
            return RouteDecision(
                strategy=RouteStrategy.COMPLEX,
                query_complexity=QueryComplexity.MEDIUM,
                top_k=5
            )
        else:
            return RouteDecision(
                strategy=RouteStrategy.DIRECT,
                query_complexity=QueryComplexity.SIMPLE,
                top_k=4
            )

    def route_query(self, query: str, context: Dict[str, Any] = None) -> RouteDecision:
        """
        智能路由决策主函数
        """
        if context is None:
            context = {}

        if not isinstance(context, dict):
            context = {}

        if not self.router_model:
            return self._rule_based_routing(query, context)

        try:
            history = context.get('history', [])
            history_length = len(history) if isinstance(history, list) else 0

            routing_input = self.routing_prompt.format(
                query=query,
                context=json.dumps(context, ensure_ascii=False),
                history_length=history_length
            )

            # 修复：完整异常保护
            response = self.router_model.invoke(routing_input)
            decision_data = self._parse_routing_response(response.content)
            return self._create_route_decision(decision_data)

        except Exception as e:
            logger.warning(f"智能路由决策失败，使用规则路由: {e}")
            return self._rule_based_routing(query, context)

    def _parse_routing_response(self, response_content: str) -> Dict[str, Any]:
        """解析路由模型响应"""
        try:
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"解析路由响应失败: {e}")

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
        """创建路由决策对象"""
        try:
            strategy_str = decision_data.get("strategy", "direct")
            complexity_str = decision_data.get("complexity", "simple")

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

            strategy = strategy_map.get(strategy_str, RouteStrategy.DIRECT)
            complexity = complexity_map.get(complexity_str, QueryComplexity.SIMPLE)

            return RouteDecision(
                strategy=strategy,
                query_complexity=complexity,
                top_k=decision_data.get("top_k", 5),
                merge_threshold=decision_data.get("merge_threshold", 2),
                needs_decomposition=decision_data.get("needs_decomposition", False),
                parallel_paths=decision_data.get("parallel_paths", 1),
                retrieval_params={"reasoning": decision_data.get("reasoning", "")}
            )

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
        基于历史表现动态调整参数
        """
        if not performance_history:
            return base_decision

        try:
            avg_response_time = sum(h.get('response_time', 1.0) for h in performance_history) / len(performance_history)
            avg_relevance_score = sum(h.get('relevance_score', 0.5) for h in performance_history) / len(performance_history)

            adjusted_decision = RouteDecision(
                strategy=base_decision.strategy,
                query_complexity=base_decision.query_complexity,
                top_k=base_decision.top_k,
                merge_threshold=base_decision.merge_threshold,
                needs_decomposition=base_decision.needs_decomposition,
                parallel_paths=base_decision.parallel_paths
            )

            if avg_relevance_score < 0.6:
                adjusted_decision.top_k = min(base_decision.top_k + 2, 10)

            if avg_response_time > 3.0:
                adjusted_decision.top_k = max(base_decision.top_k - 1, 3)

            if avg_relevance_score > 0.8 and avg_response_time > 2.0:
                if base_decision.strategy == RouteStrategy.DIRECT:
                    adjusted_decision.strategy = RouteStrategy.PARALLEL
                    adjusted_decision.parallel_paths = 2

            return adjusted_decision

        except Exception as e:
            logger.warning(f"动态参数调整失败: {e}")
            return base_decision


# 修复：线程安全单例
_intelligent_router = None
_router_lock = threading.Lock()

def get_intelligent_router() -> IntelligentRouter:
    """获取全局智能路由器实例"""
    global _intelligent_router
    with _router_lock:
        if _intelligent_router is None:
            _intelligent_router = IntelligentRouter()
    return _intelligent_router


def route_query(query: str, context: Dict[str, Any] = None) -> RouteDecision:
    """便捷函数：路由查询"""
    router = get_intelligent_router()
    return router.route_query(query, context)