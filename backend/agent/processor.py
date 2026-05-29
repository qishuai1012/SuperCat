"""Agentic RAG核心处理器"""

from chat.orchestrator import ChatOrchestrator


class AgenticRAGProcessor:
    """Agentic RAG处理器 - 策略模式重构版"""

    def __init__(self, strategies, storage, router, reflection, config):
        self.orchestrator = ChatOrchestrator(
            strategies=strategies,
            storage=storage,
            router=router,
            reflection=reflection,
            config=config,
        )

    async def process_query(self, user_text: str, user_id: str, session_id: str) -> dict:
        """处理查询"""
        return await self.orchestrator.process_query(user_text, user_id, session_id)

    async def process_query_stream(self, user_text: str, user_id: str, session_id: str):
        """流式处理查询"""
        async for event in self.orchestrator.process_query_stream(user_text, user_id, session_id):
            yield event
