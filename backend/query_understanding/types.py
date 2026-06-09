# 导入Python内置工具：
# dataclass：数据类装饰器，用于快速定义存储数据的类（自动生成构造方法等）
from dataclasses import dataclass

# Enum：枚举类，用于定义固定的可选值（让代码更规范、不易写错）
from enum import Enum

# 类型注解工具：
# Any：任意类型
# Dict：字典类型
# Optional：表示该参数可以是指定类型 或 None
from typing import Any, Dict, Optional


# ==============================================
# 枚举类：定义查询问题的复杂度等级（固定选项）
# 作用：标记用户问题是简单、中等、还是复杂问题
# ==============================================
class QueryComplexity(Enum):
    # 简单问题：直接检索即可回答
    SIMPLE = "simple"
    # 中等问题：需要简单理解
    MEDIUM = "medium"
    # 复杂问题：需要多步骤推理/多文档综合
    COMPLEX = "complex"
    # 轻度复杂：少量推理、少量文档
    COMPLEX_LIGHT = "complex_light"
    # 重度复杂：多跳推理、大量文档、多步骤处理
    COMPLEX_HEAVY = "complex_heavy"


# ==============================================
# 枚举类：定义路由策略（系统该如何处理这个查询）
# 作用：决定AI系统走哪条执行路线
# ==============================================
class RouteStrategy(Enum):
    # 直接检索：简单问题直接搜答案
    DIRECT = "direct"
    # 后退抽象：复杂问题先抽象化，再检索
    STEP_BACK = "step_back"
    # 先生成假设文档再检索：适合模糊问题
    HYDE = "hyde"
    # 复杂处理：启用多步骤、多文档综合
    COMPLEX = "complex"
    # 并行多路：拆分成多个子问题并行检索
    PARALLEL = "parallel"


# ==============================================
# 数据类：路由决策结果
# 存储AI对当前查询做出的所有路由决策
# ==============================================
@dataclass
class RouteDecision:
    # 路由策略（必选）：使用哪种处理方式
    strategy: RouteStrategy
    # 查询复杂度（必选）：问题难度等级
    query_complexity: QueryComplexity
    # 检索返回的文档数量，默认5条
    top_k: int = 5
    # 多路检索后，合并结果的阈值，默认2
    merge_threshold: int = 2
    # 智能体类型，默认default
    agent_type: str = "default"
    # 是否需要将问题拆分成子问题，默认不需要
    needs_decomposition: bool = False
    # 并行检索的路径数量，默认1路
    parallel_paths: int = 1
    # 检索的扩展参数，字典格式，默认为None
    retrieval_params: Dict[str, Any] = None

    # 类初始化后自动执行的方法
    def __post_init__(self):
        # 如果检索参数为None，初始化为空字典
        if self.retrieval_params is None:
            self.retrieval_params = {}
        
        # 限制top_k在1~20之间，保证合法范围
        self.top_k = max(1, min(self.top_k, 20))
        # 合并阈值最小为1
        self.merge_threshold = max(1, self.merge_threshold)
        # 并行路径限制在1~5之间
        self.parallel_paths = max(1, min(self.parallel_paths, 5))


# ==============================================
# 枚举类：检索策略
# 定义从知识库中搜索文档的方式
# ==============================================
class RetrievalStrategy(Enum):
    # 仅使用稠密向量检索（语义搜索）
    DENSE_ONLY = "dense_only"
    # 仅使用稀疏检索（关键词搜索）
    SPARSE_ONLY = "sparse_only"
    # 混合检索：向量+关键词结合
    HYBRID = "hybrid"
    # 自适应检索：根据问题自动选择检索方式
    ADAPTIVE = "adaptive"
    # 多阶段检索：分多轮/多步骤搜索
    MULTI_STAGE = "multi_stage"


# ==============================================
# 枚举类：检索阶段
# 多阶段检索时，标记当前处于哪一步
# ==============================================
class RetrievalStage(Enum):
    # 初始检索：第一轮粗搜
    INITIAL = "initial"
    # 扩展检索：补充更多相关文档
    EXPANSION = "expansion"
    # 验证检索：验证答案准确性
    VERIFICATION = "verification"
    # 最终检索：返回最终结果
    FINAL = "final"


# ==============================================
# 数据类：检索配置
# 存储检索模块的所有参数与规则
# ==============================================
@dataclass
class RetrievalConfig:
    # 检索策略（必选）
    strategy: RetrievalStrategy
    # 检索返回文档数量（必选）
    top_k: int
    # 相似度分数阈值，低于该值的结果会被过滤（必选）
    threshold: float
    # 是否使用重排序模型，默认开启
    use_rerank: bool = True
    # 混合检索权重：稠密/稀疏的权重比例，默认None
    hybrid_weights: Dict[str, float] = None
    # 各检索阶段的详细配置，默认None
    stage_configs: Dict[RetrievalStage, Dict[str, Any]] = None

    # 初始化后自动校验、修正参数
    def __post_init__(self):
        # 默认混合权重：稠密向量70%，稀疏关键词30%
        if self.hybrid_weights is None:
            self.hybrid_weights = {"dense": 0.7, "sparse": 0.3}
        
        # 默认阶段配置为空字典
        if self.stage_configs is None:
            self.stage_configs = {}

        # 权重归一化：确保权重总和=1
        total = sum(self.hybrid_weights.values())
        if abs(total - 1.0) > 0.01 and total:
            for key in self.hybrid_weights:
                self.hybrid_weights[key] /= total

        # top_k最小为1
        self.top_k = max(1, self.top_k)
        # 阈值限制在0~1之间
        self.threshold = max(0.0, min(1.0, self.threshold))


# ==============================================
# 数据类：查询分析
# AI对用户问题的深度分析结果
# ==============================================
@dataclass
class QueryAnalysis:
    # 问题复杂度
    complexity: QueryComplexity
    # 问题所属领域，如：医疗、法律、技术
    domain: str
    # 用户意图类型，如：询问、解释、对比、排查
    intent_type: str
    # 问题中实体数量（人名、地名、专有名词）
    entity_count: int
    # 关键词密度：关键词占比
    keyword_density: float
    # 歧义分数：越高说明问题越模糊
    ambiguity_score: float
    # 上下文依赖程度：是否依赖历史对话
    context_dependency: float


# ==============================================
# 数据类：查询理解最终结果
# 整个查询理解模块的输出结果（总包装类）
# ==============================================
@dataclass
class QueryUnderstandingResult:
    # 问题复杂度（必选）
    complexity: QueryComplexity
    # 执行类别：标记该问题应该走哪种执行流程
    execution_class: str
    # 路由决策对象（可选，可为None）
    route_decision: Optional[RouteDecision] = None
    # 查询扩展提示（可选）
    expansion_hint: Optional[str] = None
    # 详细的查询分析（可选）
    query_analysis: Optional[QueryAnalysis] = None
    # 检索配置（可选）
    retrieval_config: Optional[RetrievalConfig] = None
    # 额外元数据（可选）
    metadata: Dict[str, Any] = None

    # 初始化：如果元数据为None，设为空字典
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}