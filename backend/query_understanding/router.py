from typing import List, Optional, Dict, Any
import os
import json
import logging
import re
import threading

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from query_understanding.types import QueryComplexity, RouteDecision, RouteStrategy

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class IntelligentRouter:
    def __init__(self):
        self.router_model = None
        self._lock = threading.Lock()
        self._init_router_model()
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

    def _analyze_query_features(self, query: str) -> Dict[str, Any]:
        return {
            'length': len(query),
            'has_specific_terms': any(term in query.lower() for term in ['什么', '如何', '为什么', '请', '告诉']),
            'has_concrete_entities': any(i in query for i in ['年', '月', '日', '时间', '地点', '人名', '公司', '产品']),
            'is_conceptual': any(i in query for i in ['定义', '概念', '含义', '是什么', '什么意思', '解释']),
            'is_procedural': any(i in query for i in ['步骤', '流程', '方法', '怎么', '如何', '怎样', '过程']),
            'has_multiple_questions': query.count('？') + query.count('?') > 1,
            'is_compound': any(i in query for i in ['并且', '还有', '另外', '同时', '以及']),
        }

    def _rule_based_routing(self, query: str, context: Dict[str, Any]) -> RouteDecision:
        f = self._analyze_query_features(query)
        if f['is_compound'] or f['has_multiple_questions']:
            return RouteDecision(strategy=RouteStrategy.PARALLEL, query_complexity=QueryComplexity.COMPLEX, needs_decomposition=True, parallel_paths=2, top_k=7)
        elif f['is_conceptual']:
            return RouteDecision(strategy=RouteStrategy.HYDE, query_complexity=QueryComplexity.MEDIUM, top_k=5)
        elif f['has_concrete_entities']:
            return RouteDecision(strategy=RouteStrategy.STEP_BACK, query_complexity=QueryComplexity.MEDIUM, top_k=6)
        elif f['is_procedural']:
            return RouteDecision(strategy=RouteStrategy.COMPLEX, query_complexity=QueryComplexity.MEDIUM, top_k=5)
        else:
            return RouteDecision(strategy=RouteStrategy.DIRECT, query_complexity=QueryComplexity.SIMPLE, top_k=4)

    def route_query(self, query: str, context: Dict[str, Any] = None) -> RouteDecision:
        if not isinstance(context, dict):
            context = {}
        if not self.router_model:
            return self._rule_based_routing(query, context)
        try:
            history_length = len(context.get('history', []) or [])
            routing_input = self.routing_prompt.format(
                query=query,
                context=json.dumps(context, ensure_ascii=False),
                history_length=history_length,
            )
            response = self.router_model.invoke(routing_input)
            return self._create_route_decision(self._parse_routing_response(response.content))
        except Exception as e:
            logger.warning(f"智能路由决策失败，使用规则路由: {e}")
            return self._rule_based_routing(query, context)

    def _parse_routing_response(self, content: str) -> Dict[str, Any]:
        try:
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return {"strategy": "direct", "complexity": "simple", "top_k": 5, "merge_threshold": 2, "needs_decomposition": False, "parallel_paths": 1}

    def _create_route_decision(self, data: Dict[str, Any]) -> RouteDecision:
        try:
            strategy = {"direct": RouteStrategy.DIRECT, "step_back": RouteStrategy.STEP_BACK, "hyde": RouteStrategy.HYDE, "complex": RouteStrategy.COMPLEX, "parallel": RouteStrategy.PARALLEL}.get(data.get("strategy", "direct"), RouteStrategy.DIRECT)
            complexity = {"simple": QueryComplexity.SIMPLE, "moderate": QueryComplexity.MEDIUM, "medium": QueryComplexity.MEDIUM, "complex": QueryComplexity.COMPLEX, "complex_light": QueryComplexity.COMPLEX_LIGHT, "complex_heavy": QueryComplexity.COMPLEX_HEAVY}.get(data.get("complexity", "simple"), QueryComplexity.SIMPLE)
            return RouteDecision(strategy=strategy, query_complexity=complexity, top_k=data.get("top_k", 5), merge_threshold=data.get("merge_threshold", 2), needs_decomposition=data.get("needs_decomposition", False), parallel_paths=data.get("parallel_paths", 1), retrieval_params={"reasoning": data.get("reasoning", "")})
        except Exception as e:
            logger.warning(f"创建路由决策失败: {e}")
            return RouteDecision(strategy=RouteStrategy.DIRECT, query_complexity=QueryComplexity.SIMPLE)

    def adjust_parameters_dynamically(self, base_decision: RouteDecision, performance_history: List[Dict[str, Any]]) -> RouteDecision:
        if not performance_history:
            return base_decision
        try:
            avg_time = sum(h.get('response_time', 1.0) for h in performance_history) / len(performance_history)
            avg_rel = sum(h.get('relevance_score', 0.5) for h in performance_history) / len(performance_history)
            d = RouteDecision(strategy=base_decision.strategy, query_complexity=base_decision.query_complexity, top_k=base_decision.top_k, merge_threshold=base_decision.merge_threshold, needs_decomposition=base_decision.needs_decomposition, parallel_paths=base_decision.parallel_paths)
            if avg_rel < 0.6:
                d.top_k = min(d.top_k + 2, 10)
            if avg_time > 3.0:
                d.top_k = max(d.top_k - 1, 3)
            if avg_rel > 0.8 and avg_time > 2.0 and base_decision.strategy == RouteStrategy.DIRECT:
                d.strategy = RouteStrategy.PARALLEL
                d.parallel_paths = 2
            return d
        except Exception as e:
            logger.warning(f"动态参数调整失败: {e}")
            return base_decision


_intelligent_router = None
_router_lock = threading.Lock()


def get_intelligent_router() -> IntelligentRouter:
    global _intelligent_router
    with _router_lock:
        if _intelligent_router is None:
            _intelligent_router = IntelligentRouter()
    return _intelligent_router


def route_query(query: str, context: Dict[str, Any] = None) -> RouteDecision:
    return get_intelligent_router().route_query(query, context)
