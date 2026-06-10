"""复杂查询策略"""
import logging
from core.multi_agent_orchestrator import RequestLevelOrchestrationError
from tools import reset_tool_call_guards
from .base import ExecutionStrategy, QueryContext, ExecutionResult

logger = logging.getLogger(__name__)


class ComplexQueryStrategy(ExecutionStrategy):
    """复杂查询策略：使用多智能体协调"""

    execution_class = "complex"

    def __init__(self, orchestrator, agent_factory, storage):
        self.orchestrator = orchestrator
        self.agent_factory = agent_factory
        self.storage = storage

    async def can_handle(self, context: QueryContext) -> bool:
        if context.execution_class:
            return context.execution_class == self.execution_class
        return context.complexity in ("complex_light", "complex_heavy")

    async def execute(self, context: QueryContext) -> ExecutionResult:
        reset_tool_call_guards()
        try:
            result = await self.orchestrator.coordinate_task(
                context.query_for_retrieval,
                {
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "history": context.history
                },
                query_complexity=context.complexity
            )

            request_error = result.get("request_error")
            if request_error:
                raise RequestLevelOrchestrationError(request_error)

            return ExecutionResult(
                response=result.get("final_answer", ""),
                rag_trace=result.get("rag_trace"),
                metadata={
                    "strategy": "complex",
                    "orchestrator_used": True,
                    "route_decision": context.route_decision,
                    "execution_metadata": result.get("execution_metadata"),
                    "fallback_reason": result.get("fallback_reason"),
                    "verification": result.get("verification"),
                    "planning": result.get("planning"),
                    "subtask_results": result.get("subtask_results"),
                    "orchestration_success": bool(result.get("success", False)),
                    "orchestration_error": result.get("error"),
                }
            )
        except RequestLevelOrchestrationError:
            raise
        except Exception as e:
            logger.error(f"多智能体执行失败: {e}")
            raise

    def get_priority(self) -> int:
        return 3
