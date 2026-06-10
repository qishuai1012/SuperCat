"""策略基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class QueryContext:
    """查询上下文"""
    user_text: str
    user_id: str
    session_id: str
    history: list
    complexity: str
    route_decision: Any = None
    execution_class: str | None = None
    expansion_hint: str | None = None
    retrieval_text: str | None = None  # 改写后用于检索，None 时退回 user_text

    @property
    def query_for_retrieval(self) -> str:
        return self.retrieval_text or self.user_text
    started_at: float | None = None
    query_id: str | None = None


@dataclass
class ExecutionResult:
    """执行结果"""
    response: str
    rag_trace: Dict = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'response': self.response,
            'rag_trace': self.rag_trace,
            **self.metadata
        }


class ExecutionStrategy(ABC):
    """执行策略基类"""

    @abstractmethod
    async def can_handle(self, context: QueryContext) -> bool:
        """判断是否能处理该查询"""
        pass

    @abstractmethod
    async def execute(self, context: QueryContext) -> ExecutionResult:
        """执行查询"""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """获取优先级（数字越小优先级越高）"""
        pass
