from dynamic_retrieval_strategy import get_dynamic_retrieval_strategy
from performance_config import get_performance_config
from query_understanding.complexity import get_complexity_analyzer
from query_understanding.router import get_intelligent_router
from query_understanding.types import QueryComplexity, QueryUnderstandingResult, RouteDecision, RouteStrategy
from smart_cache import get_smart_cache


class QueryUnderstandingService:
    def __init__(self, router=None):
        self.router = router or get_intelligent_router()
        self.complexity_analyzer = get_complexity_analyzer(model_path="../model/m3e-base")
        self.retrieval_strategy = get_dynamic_retrieval_strategy()
        self.smart_cache = get_smart_cache()

    def _get_execution_class(self, complexity: QueryComplexity) -> str:
        if complexity in (QueryComplexity.SIMPLE, QueryComplexity.MEDIUM):
            return "simple"
        return "complex"

    def _get_expansion_hint(self, route_decision) -> str | None:
        if not route_decision:
            return None
        strategy = getattr(route_decision, "strategy", None)
        if strategy == RouteStrategy.STEP_BACK:
            return "step_back"
        if strategy == RouteStrategy.HYDE:
            return "hyde"
        if strategy in (RouteStrategy.COMPLEX, RouteStrategy.PARALLEL):
            return "complex"
        if strategy == RouteStrategy.DIRECT:
            return "direct"
        return None

    def _restore_route_decision(self, cached: dict | None):
        if not cached:
            return None
        try:
            return RouteDecision(
                strategy=RouteStrategy(cached.get("strategy", "direct")),
                query_complexity=QueryComplexity(cached.get("query_complexity", "medium")),
                top_k=int(cached.get("top_k", 5)),
                merge_threshold=int(cached.get("merge_threshold", 2)),
                agent_type=cached.get("agent_type", "default"),
                needs_decomposition=bool(cached.get("needs_decomposition", False)),
                parallel_paths=int(cached.get("parallel_paths", 1)),
                retrieval_params=cached.get("retrieval_params") or {},
            )
        except Exception:
            return None

    def _get_route_decision(self, query: str, context: dict, complexity: QueryComplexity):
        if not self.router or complexity == QueryComplexity.SIMPLE:
            return None

        strategy_config = get_performance_config().get_strategy(complexity)
        cache_enabled = strategy_config.enable_cache and complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX_LIGHT)
        if cache_enabled:
            cached = self._restore_route_decision(self.smart_cache.get_route_decision(query))
            if cached is not None:
                return cached

        decision = self.router.route_query(query, context)
        if decision and cache_enabled:
            self.smart_cache.set_route_decision(query, decision, ttl=strategy_config.cache_ttl)
        return decision

    def analyze_for_chat(self, user_text: str, user_id: str, session_id: str, history: list) -> QueryUnderstandingResult:
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "history": history,
        }
        complexity = self.complexity_analyzer.analyze(user_text, context)
        route_decision = self._get_route_decision(user_text, context, complexity)
        return QueryUnderstandingResult(
            complexity=complexity,
            execution_class=self._get_execution_class(complexity),
            route_decision=route_decision,
            expansion_hint=self._get_expansion_hint(route_decision),
        )

    def analyze_for_retrieval(self, query: str, context: dict | None = None) -> QueryUnderstandingResult:
        context = context or {}
        complexity = self.complexity_analyzer.analyze(query, context)
        route_decision = self._get_route_decision(query, context, complexity)
        query_analysis = self.retrieval_strategy.analyze_query(query, context)
        query_analysis.complexity = complexity
        retrieval_config = self.retrieval_strategy.select_strategy(query_analysis, performance_context=context)
        return QueryUnderstandingResult(
            complexity=complexity,
            execution_class=self._get_execution_class(complexity),
            route_decision=route_decision,
            expansion_hint=self._get_expansion_hint(route_decision),
            query_analysis=query_analysis,
            retrieval_config=retrieval_config,
        )


_query_understanding_service = None


def get_query_understanding_service(router=None) -> QueryUnderstandingService:
    global _query_understanding_service
    if _query_understanding_service is None or (router is not None and _query_understanding_service.router is not router):
        _query_understanding_service = QueryUnderstandingService(router=router)
    return _query_understanding_service
