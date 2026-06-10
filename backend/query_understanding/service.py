# 导入依赖：动态检索策略、性能配置、问题复杂度分析、智能路由、类型定义、智能缓存
from query_understanding.retrieval_policy import get_dynamic_retrieval_strategy
from monitoring.performance_config import get_performance_config
from query_understanding.complexity import get_complexity_analyzer
from query_understanding.router import get_intelligent_router
from query_understanding.types import QueryComplexity, QueryUnderstandingResult, RouteDecision, RouteStrategy
from storage.cache import get_smart_cache

"""
服务功能：查询理解服务（RAG 系统的“大脑”）
职责：
1. 分析用户问题复杂度（简单 / 中等 / 复杂）
2. 智能路由：决定使用哪种检索策略（step_back / hyde / complex / direct）
3. 提供动态检索配置
4. 提供缓存加速
5. 输出 RAG 执行指令（expansion_hint）
"""
class QueryUnderstandingService:

    # 初始化服务：加载路由、复杂度分析、检索策略、  缓存
    def __init__(self, router=None):
        # 路由模型：决定用哪种查询扩展策略
        self.router = router or get_intelligent_router()
        
        import os
        # 加载本地 embedding 模型路径，用于问题复杂度分析
        _model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "m3e-base")
        
        # 问题复杂度分析器（简单/中等/复杂）
        self.complexity_analyzer = get_complexity_analyzer(model_path=_model_path)
        
        # 动态检索策略：根据问题自动选择检索方式
        self.retrieval_strategy = get_dynamic_retrieval_strategy()
        
        # 智能缓存：缓存路由结果，提升性能
        self.smart_cache = get_smart_cache()

    # 根据问题复杂度，返回执行级别：simple / complex
    # 简单/中等 → simple（快速、轻量）
    # 复杂 → complex（多路、强检索、重计算）
    def _get_execution_class(self, complexity: QueryComplexity) -> str:
        if complexity in (QueryComplexity.SIMPLE, QueryComplexity.MEDIUM):
            return "simple"
        return "complex"

    # 把路由策略转换成 RAG 能识别的扩展指令
    # 输出：step_back / hyde / complex / direct / None
    def _get_expansion_hint(self, route_decision) -> str | None:
        if not route_decision:
            return None
        
        # 获取路由策略
        strategy = getattr(route_decision, "strategy", None)
        
        # 转换成 RAG 识别的字符串指令
        if strategy == RouteStrategy.STEP_BACK:
            return "step_back"
        if strategy == RouteStrategy.HYDE:
            return "hyde"
        if strategy in (RouteStrategy.COMPLEX, RouteStrategy.PARALLEL):
            return "complex"
        if strategy == RouteStrategy.DIRECT:
            return "direct"
        
        return None

    # 从缓存中恢复路由决策（提速用）
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

    # 获取路由决策：核心逻辑
    # 1. 简单问题不路由
    # 2. 复杂问题走路由模型
    # 3. 开启缓存则读缓存
    def _get_route_decision(self, query: str, context: dict, complexity: QueryComplexity):
        # 无路由模型 或 简单问题 → 不做路由
        if not self.router or complexity == QueryComplexity.SIMPLE:
            return None

        # 获取性能策略配置
        strategy_config = get_performance_config().get_strategy(complexity)
        
        # 只有中等/轻度复杂问题开启缓存
        cache_enabled = strategy_config.enable_cache and complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX_LIGHT)
        
        # 缓存命中 → 直接返回
        if cache_enabled:
            cached = self._restore_route_decision(self.smart_cache.get_route_decision(query))
            if cached is not None:
                return cached

        # 路由模型分析，得到策略
        decision = self.router.route_query(query, context)
        
        # 缓存路由结果
        if decision and cache_enabled:
            self.smart_cache.set_route_decision(query, decision, ttl=strategy_config.cache_ttl)
            
        return decision

    # 对外接口：聊天场景使用（带用户、会话、历史）
    # 返回：复杂度 + 执行级别 + 路由决策 + expansion_hint
    def analyze_for_chat(self, user_text: str, user_id: str, session_id: str, history: list) -> QueryUnderstandingResult:
        # 上下文：用户ID、会话ID、历史记录
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "history": history,
        }
        
        # 分析问题复杂度
        complexity = self.complexity_analyzer.analyze(user_text, context)
        
        # 路由决策（策略选择）
        route_decision = self._get_route_decision(user_text, context, complexity)
        
        # 返回完整分析结果
        return QueryUnderstandingResult(
            complexity=complexity,
            execution_class=self._get_execution_class(complexity),
            route_decision=route_decision,
            expansion_hint=self._get_expansion_hint(route_decision),
        )

    # 对外接口：检索场景使用（不带聊天上下文）
    # 输出更完整的检索配置：query_analysis + retrieval_config
    def analyze_for_retrieval(self, query: str, context: dict | None = None) -> QueryUnderstandingResult:
        context = context or {}
        
        # 分析复杂度
        complexity = self.complexity_analyzer.analyze(query, context)
        
        # 路由决策
        route_decision = self._get_route_decision(query, context, complexity)
        
        # 查询分析（关键词、意图、分块等）
        query_analysis = self.retrieval_strategy.analyze_query(query, context)
        query_analysis.complexity = complexity
        
        # 选择检索策略配置
        retrieval_config = self.retrieval_strategy.select_strategy(query_analysis, performance_context=context)
        
        return QueryUnderstandingResult(
            complexity=complexity,
            execution_class=self._get_execution_class(complexity),
            route_decision=route_decision,
            expansion_hint=self._get_expansion_hint(route_decision),
            query_analysis=query_analysis,
            retrieval_config=retrieval_config,
        )


# 全局单例：整个系统只创建一个服务实例，避免重复加载模型
_query_understanding_service = None

# 获取查询理解服务（单例模式）
def get_query_understanding_service(router=None) -> QueryUnderstandingService:
    global _query_understanding_service
    if _query_understanding_service is None or (router is not None and _query_understanding_service.router is not router):
        _query_understanding_service = QueryUnderstandingService(router=router)
    return _query_understanding_service