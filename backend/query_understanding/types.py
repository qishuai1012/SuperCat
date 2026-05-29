from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    COMPLEX_LIGHT = "complex_light"
    COMPLEX_HEAVY = "complex_heavy"


class RouteStrategy(Enum):
    DIRECT = "direct"
    STEP_BACK = "step_back"
    HYDE = "hyde"
    COMPLEX = "complex"
    PARALLEL = "parallel"


@dataclass
class RouteDecision:
    strategy: RouteStrategy
    query_complexity: QueryComplexity
    top_k: int = 5
    merge_threshold: int = 2
    agent_type: str = "default"
    needs_decomposition: bool = False
    parallel_paths: int = 1
    retrieval_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.retrieval_params is None:
            self.retrieval_params = {}
        self.top_k = max(1, min(self.top_k, 20))
        self.merge_threshold = max(1, self.merge_threshold)
        self.parallel_paths = max(1, min(self.parallel_paths, 5))


class RetrievalStrategy(Enum):
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    MULTI_STAGE = "multi_stage"


class RetrievalStage(Enum):
    INITIAL = "initial"
    EXPANSION = "expansion"
    VERIFICATION = "verification"
    FINAL = "final"


@dataclass
class RetrievalConfig:
    strategy: RetrievalStrategy
    top_k: int
    threshold: float
    use_rerank: bool = True
    hybrid_weights: Dict[str, float] = None
    stage_configs: Dict[RetrievalStage, Dict[str, Any]] = None

    def __post_init__(self):
        if self.hybrid_weights is None:
            self.hybrid_weights = {"dense": 0.7, "sparse": 0.3}
        if self.stage_configs is None:
            self.stage_configs = {}

        total = sum(self.hybrid_weights.values())
        if abs(total - 1.0) > 0.01 and total:
            for key in self.hybrid_weights:
                self.hybrid_weights[key] /= total

        self.top_k = max(1, self.top_k)
        self.threshold = max(0.0, min(1.0, self.threshold))


@dataclass
class QueryAnalysis:
    complexity: QueryComplexity
    domain: str
    intent_type: str
    entity_count: int
    keyword_density: float
    ambiguity_score: float
    context_dependency: float


@dataclass
class QueryUnderstandingResult:
    complexity: QueryComplexity
    execution_class: str
    route_decision: Optional[RouteDecision] = None
    expansion_hint: Optional[str] = None
    query_analysis: Optional[QueryAnalysis] = None
    retrieval_config: Optional[RetrievalConfig] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
