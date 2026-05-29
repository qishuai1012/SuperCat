"""
动态检索策略系统
基于查询特征动态选择最优检索策略和参数
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging
import re
from enum import Enum
import threading  # 修复线程安全
from datetime import datetime  # 修复动态导入

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from query_understanding.types import QueryAnalysis, QueryComplexity, RetrievalConfig, RetrievalStage, RetrievalStrategy

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")




class DynamicRetrievalStrategy:
    """
    动态检索策略管理器
    基于查询特征、历史表现和实时反馈动态调整检索策略
    """

    def __init__(self):
        self.strategy_model = None
        self._lock = threading.Lock()  # 修复：全局锁

        self._init_strategy_model()

        # 策略选择规则
        self.strategy_rules = {
            "simple": self._get_simple_strategy,
            "medium": self._get_moderate_strategy,
            "complex": self._get_complex_strategy,
        }

        self.runtime_overrides = {}

        # 性能历史
        self.performance_history = []
        self.max_history_size = 100

    def _init_strategy_model(self):
        """初始化策略选择模型（已禁用LLM策略选择，使用规则选择）"""
        # 禁用 LLM-based 策略选择，避免额外 API 调用（37秒开销）
        # 规则选择已足够且瞬间完成
        self.strategy_model = None
        logger.info("动态检索策略：使用规则选择（已禁用LLM选择）")

    def analyze_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """
        分析查询特征

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            查询分析结果
        """
        if context is None:
            context = {}

        features = self._extract_query_features(query, context)
        complexity = self._determine_complexity(features)
        domain = self._identify_domain(query)
        intent_type = self._identify_intent(query)

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
        """提取查询特征"""
        features = {}
        features['length'] = len(query)

        # ===================== 修复：支持中文分词 =====================
        words = re.findall(r'[\w\u4e00-\u9fff]+', query)
        features['word_count'] = max(len(words), 1)

        entity_indicators = ['什么', '哪个', '哪里', '何时', '谁', '多少', '为什么', '如何']
        features['entity_count'] = sum(1 for indicator in entity_indicators if indicator in query)

        keywords = self._extract_keywords(query)
        features['keyword_density'] = len(keywords) / features['word_count']
        features['ambiguity_score'] = self._calculate_ambiguity(query)
        features['context_dependency'] = self._calculate_context_dependency(query, context)

        return features

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词（支持中文）"""
        stop_words = {'的', '了', '和', '与', '或', '在', '是', '有', '我', '你', '他', '她', '它'}
        words = re.findall(r'[\w\u4e00-\u9fff]+', query)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

    def _calculate_ambiguity(self, query: str) -> float:
        """计算歧义性评分（修复逻辑）"""
        ambiguity_indicators = ['可能', '大概', '也许', '不确定', '模糊', '不清楚']
        count = sum(1 for w in ambiguity_indicators if w in query)
        return min(count / 3.0, 1.0)

    def _calculate_context_dependency(self, query: str, context: Dict[str, Any]) -> float:
        """计算上下文依赖度（修复逻辑）"""
        if not context or 'history' not in context:
            return 0.0
        indicators = ['刚才', '之前', '刚才说的', '刚才提到的', '继续', '接着']
        count = sum(1 for w in indicators if w in query)
        return min(count / 3.0, 1.0)

    def _determine_complexity(self, features: Dict[str, Any]) -> QueryComplexity:
        """确定查询复杂度"""
        score = 0
        if features['length'] > 100:
            score += 2
        elif features['length'] > 50:
            score += 1

        score += features['entity_count'] * 0.5
        score += features['ambiguity_score'] * 2
        score += features['context_dependency'] * 1.5

        if score >= 4:
            return QueryComplexity.COMPLEX
        elif score >= 2:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE

    def _identify_domain(self, query: str) -> str:
        """识别查询领域"""
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
        """识别查询意图"""
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
        if complexity == QueryComplexity.SIMPLE:
            return "simple"
        if complexity == QueryComplexity.MEDIUM:
            return "medium"
        return "complex"

    def select_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any] = None) -> RetrievalConfig:
        """选择最优检索策略"""
        if performance_context is None:
            performance_context = {}

        complexity_key = self._get_strategy_key(query_analysis.complexity)

        selected_config = None
        if self.strategy_model:
            try:
                selected_config = self._intelligent_strategy_selection(query_analysis, performance_context)
            except Exception as e:
                logger.warning(f"智能策略选择失败: {e}")

        if selected_config is None:
            strategy_func = self.strategy_rules.get(complexity_key, self._get_simple_strategy)
            selected_config = strategy_func(query_analysis, performance_context)

        return self._apply_runtime_overrides(selected_config)

    def _intelligent_strategy_selection(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """智能策略选择（增加异常捕获）"""
        try:
            strategy_prompt = f"""请基于以下查询特征选择最优检索策略：

查询复杂度: {query_analysis.complexity.value}
领域: {query_analysis.domain}
意图类型: {query_analysis.intent_type}
实体数量: {query_analysis.entity_count}
关键词密度: {query_analysis.keyword_density:.2f}
歧义性评分: {query_analysis.ambiguity_score:.2f}
上下文依赖度: {query_analysis.context_dependency:.2f}

历史性能: {json.dumps(performance_context, ensure_ascii=False)}

可用策略：
1. dense_only - 仅稠密向量（适合语义查询）
2. sparse_only - 仅稀疏向量（适合关键词匹配）
3. hybrid - 混合检索（平衡语义和关键词）
4. adaptive - 自适应策略（根据反馈动态调整）
5. multi_stage - 多阶段检索（复杂查询）

请返回JSON格式的策略配置：
{{
    "strategy": "选择的策略",
    "top_k": 检索数量,
    "threshold": 相关性阈值,
    "use_rerank": true/false,
    "hybrid_weights": {{"dense": 权重, "sparse": 权重}},
    "reasoning": "选择理由"
}}
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
        """解析策略响应"""
        try:
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"策略解析失败: {e}")
        return self._get_default_strategy_config()

    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """获取默认策略配置"""
        return {
            "strategy": "hybrid",
            "top_k": 5,
            "threshold": 0.3,
            "use_rerank": True,
            "hybrid_weights": {"dense": 0.7, "sparse": 0.3},
            "reasoning": "默认混合策略"
        }

    def _get_simple_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """简单查询策略"""
        return RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=3,
            threshold=0.4,
            use_rerank=False,
            hybrid_weights={"dense": 0.6, "sparse": 0.4}
        )

    def _get_moderate_strategy(self, query_analysis: QueryAnalysis, performance_context: Dict[str, Any]) -> RetrievalConfig:
        """中等复杂度查询策略"""
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
        """复杂查询策略"""
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
        with self._lock:
            self.runtime_overrides[key] = value

    def get_runtime_overrides(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.runtime_overrides)

    def _apply_runtime_overrides(self, retrieval_config: RetrievalConfig) -> RetrievalConfig:
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
        """更新性能历史（线程安全）"""
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
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]

    def get_performance_insights(self) -> Dict[str, Any]:
        """获取性能洞察"""
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


# ===================== 修复：线程安全单例 =====================
_dynamic_retrieval_strategy = None
_strategy_lock = threading.Lock()

def get_dynamic_retrieval_strategy() -> DynamicRetrievalStrategy:
    """获取全局动态检索策略管理器"""
    global _dynamic_retrieval_strategy
    with _strategy_lock:
        if _dynamic_retrieval_strategy is None:
            _dynamic_retrieval_strategy = DynamicRetrievalStrategy()
    return _dynamic_retrieval_strategy


def analyze_and_select_strategy(query: str, context: Dict[str, Any] = None) -> tuple[QueryAnalysis, RetrievalConfig]:
    """便捷函数：分析查询并选择策略"""
    manager = get_dynamic_retrieval_strategy()
    analysis = manager.analyze_query(query, context)
    config = manager.select_strategy(analysis)
    return analysis, config