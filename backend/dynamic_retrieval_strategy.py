"""
动态检索策略系统
基于查询特征动态选择最优检索策略和参数
核心功能：
1. 分析用户查询的特征、复杂度、意图
2. 根据问题复杂度自动选择检索策略（向量/关键词/混合/多阶段）
3. 动态配置检索参数（top_k、阈值、权重、是否重排序）
4. 记录性能数据，支持策略优化
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging
import re
from enum import Enum
import threading
from datetime import datetime

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 导入类型定义：查询分析、复杂度、检索配置、检索策略等
from query_understanding.types import QueryAnalysis, QueryComplexity, RetrievalConfig, RetrievalStage, RetrievalStrategy

load_dotenv()
logger = logging.getLogger(__name__)

# 从环境变量读取模型配置
API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class DynamicRetrievalStrategy:
    """
    动态检索策略管理器
    根据查询复杂度、领域、意图自动选择最合适的检索方式
    禁用LLM，纯规则实现，速度极快
    """

    def __init__(self):
        # 策略模型（已禁用LLM，使用规则策略）
        self.strategy_model = None
        # 线程锁：保证多线程安全
        self._lock = threading.Lock()

        # 初始化模型（已禁用LLM，避免开销）
        self._init_strategy_model()

        # 策略规则映射：不同复杂度 → 不同策略函数
        self.strategy_rules = {
            "simple": self._get_simple_strategy,
            "medium": self._get_moderate_strategy,
            "complex": self._get_complex_strategy,
        }

        # 运行时参数覆盖（动态调整）
        self.runtime_overrides = {}

        # 性能历史记录（用于优化策略）
        self.performance_history = []
        self.max_history_size = 100

    def _init_strategy_model(self):
        """初始化策略模型
        已禁用LLM，使用规则选择，性能提升显著
        """
        self.strategy_model = None
        logger.info("动态检索策略：使用规则选择（已禁用LLM选择）")

    def analyze_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """
        对外接口：分析查询，返回结构化查询分析结果
        :param query: 用户问题
        :param context: 上下文（对话历史等）
        :return: QueryAnalysis 包含：复杂度、领域、意图、关键词密度等
        """
        if context is None:
            context = {}

        # 提取查询特征
        features = self._extract_query_features(query, context)
        # 判断复杂度
        complexity = self._determine_complexity(features)
        # 判断领域（技术/商业/教育等）
        domain = self._identify_domain(query)
        # 判断意图（事实/流程/对比/分析）
        intent_type = self._identify_intent(query)

        # 返回完整查询分析对象
        return QueryAnalysis(
            complexity=complexity,
            domain=domain,
            intent_type=intent_type,
            entity_count=features['entity_count'],
            keyword_density=features['keyword_density'],
            ambiguity_score=features['ambiguity_score'],
            context_dependency=features['context_dependency']
        )

    def _extract_query_features(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取查询核心特征（内部使用）"""
        features = {}
        features['length'] = len(query)

        # 支持中文分词：提取中文+英文单词
        words = re.findall(r'[\w\u4e00-\u9fff]+', query)
        features['word_count'] = max(len(words), 1)

        # 实体数量：什么/哪个/哪里/何时/谁/多少/为什么/如何
        entity_indicators = ['什么', '哪个', '哪里', '何时', '谁', '多少', '为什么', '如何']
        features['entity_count'] = sum(1 for indicator in entity_indicators if indicator in query)

        # 提取关键词
        keywords = self._extract_keywords(query)
        features['keyword_density'] = len(keywords) / features['word_count']

        # 歧义性评分（模糊词越多分数越高）
        features['ambiguity_score'] = self._calculate_ambiguity(query)
        # 上下文依赖评分（依赖历史则越高）
        features['context_dependency'] = self._calculate_context_dependency(query, context)

        return features

    def _extract_keywords(self, query: str) -> List[str]:
        """提取中文关键词，过滤停用词"""
        stop_words = {'的', '了', '和', '与', '或', '在', '是', '有', '我', '你', '他', '她', '它'}
        words = re.findall(r'[\w\u4e00-\u9fff]+', query)
        # 只保留长度>1且非停用词的关键词
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

    def _calculate_ambiguity(self, query: str) -> float:
        """计算歧义性：模糊词越多，分数越高（0~1）"""
        ambiguity_indicators = ['可能', '大概', '也许', '不确定', '模糊', '不清楚']
        count = sum(1 for w in ambiguity_indicators if w in query)
        return min(count / 3.0, 1.0)

    def _calculate_context_dependency(self, query: str, context: Dict[str, Any]) -> float:
        """计算上下文依赖度：是否依赖历史对话（0~1）"""
        if not context or 'history' not in context:
            return 0.0
        indicators = ['刚才', '之前', '刚才说的', '刚才提到的', '继续', '接着']
        count = sum(1 for w in indicators if w in query)
        return min(count / 3.0, 1.0)

    def _determine_complexity(self, features: Dict[str, Any]) -> QueryComplexity:
        """根据特征自动计算问题复杂度"""
        score = 0
        # 长度加分
        if features['length'] > 100:
            score += 2
        elif features['length'] > 50:
            score += 1

        # 实体数量、歧义、上下文依赖 都会提升复杂度
        score += features['entity_count'] * 0.5
        score += features['ambiguity_score'] * 2
        score += features['context_dependency'] * 1.5

        # 分数判定复杂度
        if score >= 4:
            return QueryComplexity.COMPLEX
        elif score >= 2:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE

    def _identify_domain(self, query: str) -> str:
        """识别问题领域：技术/商业/科学/教育/通用"""
        domains = {
            'technology': ['技术', '编程', '代码', '软件', '系统', '算法'],
            'business': ['商业', '市场', '经济', '金融', '投资', '公司'],
            'science': ['科学', '研究', '实验', '理论', '原理', '发现'],
            'education': ['学习', '教育', '课程', '知识', '教学', '考试'],
            'general': ['一般', '普通', '常识', '基本', '简单']
        }
        q = query.lower()
        for domain, keywords in domains.items():
            if any(kw in q for kw in keywords):
                return domain
        return 'general'

    def _identify_intent(self, query: str) -> str:
        """识别用户意图：事实/流程/对比/分析/综合"""
        intents = {
            'factual': ['是什么', '什么是', '定义', '概念', '含义'],
            'procedural': ['怎么做', '如何', '步骤', '方法', '流程', '过程'],
            'comparative': ['比较', '对比', '区别', '差异', '优缺点'],
            'analytical': ['分析', '原因', '为什么', '原理', '机制'],
            'synthetic': ['总结', '概括', '综述', '综合', '整合']
        }
        q = query.lower()
        for intent, patterns in intents.items():
            if any(p in q for p in patterns):
                return intent
        return 'factual'

    def _get_strategy_key(self, complexity: QueryComplexity) -> str:
        """将复杂度枚举转为策略key：simple/medium/complex"""
        if complexity == QueryComplexity.SIMPLE:
            return "simple"
        if complexity == QueryComplexity.MEDIUM:
            return "medium"
        return "complex"

    def select_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any] = None) -> RetrievalConfig:
        """
        核心函数：选择最优检索策略
        根据复杂度自动选择 简单/中等/复杂 检索方案
        返回 RetrievalConfig 检索配置
        """
        if performance_context is None:
            performance_context = {}

        complexity_key = self._get_strategy_key(query_analysis.complexity)

        selected_config = None
        # LLM已禁用，直接走规则策略
        if self.strategy_model:
            try:
                selected_config = self._intelligent_strategy_selection(query_analysis, performance_context)
            except Exception as e:
                logger.warning(f"智能策略选择失败: {e}")

        # 规则策略选择
        if selected_config is None:
            strategy_func = self.strategy_rules.get(complexity_key, self._get_simple_strategy)
            selected_config = strategy_func(query_analysis, performance_context)

        # 应用运行时覆盖参数
        return self._apply_runtime_overrides(selected_config)

    def _intelligent_strategy_selection(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """LLM策略选择（已禁用，仅保留兼容）"""
        try:
            # 提示词（已不会执行）
            strategy_prompt = f"""请基于以下查询特征选择最优检索策略：
    查询复杂度: {query_analysis.complexity.value}
    领域: {query_analysis.domain}
    意图类型: {query_analysis.intent_type}
    实体数量: {query_analysis.entity_count}
    关键词密度: {query_analysis.keyword_density:.2f}
    歧义性评分: {query_analysis.ambiguity_score:.2f}
    上下文依赖度: {query_analysis.context_dependency:.2f}

    可用策略：dense_only、sparse_only、hybrid、adaptive、multi_stage
    返回JSON格式策略配置
    """
            response = self.strategy_model.invoke(strategy_prompt)
            strategy_data = self._parse_strategy_response(response.content)

            strategy_map = {
                "dense_only": RetrievalStrategy.DENSE_ONLY,
                "sparse_only": RetrievalStrategy.SPARSE_ONLY,
                "hybrid": RetrievalStrategy.HYBRID,
                "adaptive": RetrievalStrategy.ADAPTIVE,
                "multi_stage": RetrievalStrategy.MULTI_STAGE
            }
            strategy = strategy_map.get(strategy_data.get("strategy"), RetrievalStrategy.HYBRID)

            return RetrievalConfig(
                strategy=strategy,
                top_k=strategy_data.get("top_k", 5),
                threshold=strategy_data.get("threshold", 0.3),
                use_rerank=strategy_data.get("use_rerank", True),
                hybrid_weights=strategy_data.get("hybrid_weights")
            )
        except Exception as e:
            logger.error(f"智能策略异常: {e}")
            return self._get_simple_strategy(query_analysis, performance_context)

    def _parse_strategy_response(self, response_content: str) -> Dict[str, Any]:
        """解析LLM返回的JSON（兼容保留）"""
        try:
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"策略解析失败: {e}")
        return self._get_default_strategy_config()

    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """默认策略配置"""
        return {
            "strategy": "hybrid",
            "top_k": 5,
            "threshold": 0.3,
            "use_rerank": True,
            "hybrid_weights": {"dense": 0.7, "sparse": 0.3},
            "reasoning": "默认混合策略"
        }

    def _get_simple_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """简单问题检索策略：快、轻量"""
        return RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=3,
            threshold=0.4,
            use_rerank=False,
            hybrid_weights={"dense": 0.6, "sparse": 0.4}
        )

    def _get_moderate_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """中等问题策略：平衡速度与精度"""
        if query_analysis.intent_type == 'factual':
            strategy = RetrievalStrategy.HYBRID
            weights = {"dense": 0.5, "sparse": 0.5}
        elif query_analysis.intent_type == 'procedural':
            strategy = RetrievalStrategy.DENSE_ONLY
            weights = {"dense": 1.0, "sparse": 0.0}
        else:
            strategy = RetrievalStrategy.HYBRID
            weights = {"dense": 0.7, "sparse": 0.3}

        return RetrievalConfig(
            strategy=strategy,
            top_k=5,
            threshold=0.3,
            use_rerank=True,
            hybrid_weights=weights
        )

    def _get_complex_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """复杂问题策略：多阶段、高精度、重排序"""
        return RetrievalConfig(
            strategy=RetrievalStrategy.MULTI_STAGE,
            top_k=8,
            threshold=0.2,
            use_rerank=True,
            hybrid_weights={"dense": 0.7, "sparse": 0.3},
            stage_configs={
                RetrievalStage.INITIAL: {"top_k": 10, "strategy": "hybrid"},
                RetrievalStage.EXPANSION: {"top_k": 15, "strategy": "dense_only"},
                RetrievalStage.FINAL: {"top_k": 5, "strategy": "hybrid"}
            }
        )

    def set_runtime_override(self, key: str, value: Any):
        """运行时动态覆盖参数（线程安全）"""
        with self._lock:
            self.runtime_overrides[key] = value

    def get_runtime_overrides(self) -> Dict[str, Any]:
        """获取覆盖参数"""
        with self._lock:
            return dict(self.runtime_overrides)

    def _apply_runtime_overrides(self, retrieval_config: RetrievalConfig) -> RetrievalConfig:
        """应用运行时参数覆盖"""
        overrides = self.get_runtime_overrides()
        if not overrides:
            return retrieval_config

        hybrid_weights = overrides.get("hybrid_weights") or dict(retrieval_config.hybrid_weights or {})
        stage_configs = dict(retrieval_config.stage_configs or {})
        return RetrievalConfig(
            strategy=retrieval_config.strategy,
            top_k=int(overrides.get("top_k", retrieval_config.top_k)),
            threshold=float(overrides.get("threshold", retrieval_config.threshold)),
            use_rerank=bool(overrides.get("use_rerank", retrieval_config.use_rerank)),
            hybrid_weights=hybrid_weights,
            stage_configs=stage_configs,
        )

    def update_performance(self, query_analysis: QueryAnalysis, retrieval_config: RetrievalConfig, performance_metrics: Dict[str, Any]):
        """记录策略性能（线程安全），用于后续优化"""
        with self._lock:
            record = {
                "query_analysis": {
                    "complexity": query_analysis.complexity.value,
                    "domain": query_analysis.domain,
                    "intent_type": query_analysis.intent_type
                },
                "strategy": retrieval_config.strategy.value,
                "config": {
                    "top_k": retrieval_config.top_k,
                    "threshold": retrieval_config.threshold,
                    "use_rerank": retrieval_config.use_rerank,
                    "hybrid_weights": retrieval_config.hybrid_weights,
                },
                "performance": performance_metrics,
                "timestamp": datetime.now().isoformat()
            }
            self.performance_history.append(record)
            # 限制历史记录大小
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]

    def get_performance_insights(self) -> Dict[str, Any]:
        """获取策略性能统计，用于分析最优策略"""
        if not self.performance_history:
            return {}

        strategy_perf = {}
        for r in self.performance_history:
            s = r["strategy"]
            if s not in strategy_perf:
                strategy_perf[s] = []
            rel = r["performance"].get("relevance_score", 0.5)
            rt = r["performance"].get("response_time", 1.0)
            strategy_perf[s].append(rel / max(rt, 0.1))

        insights = {}
        for s, scores in strategy_perf.items():
            insights[s] = {
                "average_score": sum(scores) / len(scores),
                "sample_count": len(scores),
                "best_score": max(scores),
                "worst_score": min(scores)
            }
        return insights


# ===================== 全局单例（线程安全） =====================
_dynamic_retrieval_strategy = None
_strategy_lock = threading.Lock()

def get_dynamic_retrieval_strategy() -> DynamicRetrievalStrategy:
    """获取全局唯一的动态检索策略实例"""
    global _dynamic_retrieval_strategy
    with _strategy_lock:
        if _dynamic_retrieval_strategy is None:
            _dynamic_retrieval_strategy = DynamicRetrievalStrategy()
    return _dynamic_retrieval_strategy


def analyze_and_select_strategy(query: str, context: Dict[str, Any] = None) -> tuple[QueryAnalysis, RetrievalConfig]:
    """便捷调用函数：一行完成分析+策略选择"""
    manager = get_dynamic_retrieval_strategy()
    analysis = manager.analyze_query(query, context)
    config = manager.select_strategy(analysis)
    return analysis, config