import logging
import time
import uuid

from agent.strategies.base import QueryContext
from performance_config import get_performance_config
from performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


class QueryContextBuilder:
    def __init__(self, storage, router, config):
        self.storage = storage
        self.router = router
        self.config = config

    async def build(self, user_text: str, user_id: str, session_id: str) -> QueryContext:
        query_id = str(uuid.uuid4())
        monitor_metrics = None
        build_step = None
        if get_performance_config().enable_performance_monitoring:
            monitor_metrics = get_performance_monitor().start_query(query_id, user_text, user_id, session_id)
            build_step = monitor_metrics.add_step("build_context")

        history = self.storage.load(user_id, session_id)
        history_data = [{"type": getattr(m, "type", "unknown"), "content": str(m.content)} for m in history]

        from query_understanding.service import get_query_understanding_service

        understanding = get_query_understanding_service(router=self.router).analyze_for_chat(
            user_text=user_text,
            user_id=user_id,
            session_id=session_id,
            history=history_data,
        )

        if monitor_metrics:
            monitor_metrics.complexity = understanding.complexity.value
            monitor_metrics.strategy_used = understanding.execution_class
        if build_step:
            build_step.finish()

        return QueryContext(
            user_text=user_text,
            user_id=user_id,
            session_id=session_id,
            history=history_data,
            complexity=understanding.complexity.value,
            route_decision=understanding.route_decision,
            execution_class=understanding.execution_class,
            expansion_hint=understanding.expansion_hint,
            started_at=time.perf_counter(),
            query_id=query_id,
        )
