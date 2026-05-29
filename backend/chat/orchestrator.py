import asyncio
import logging

from chat.context_builder import QueryContextBuilder
from chat.persistence import ConversationPersistence
from chat.postprocessor import PostProcessor
from chat.result_builder import ResultBuilder
from chat.streaming import StreamProcessor
from chat.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    def __init__(self, strategies, storage, router, reflection, config):
        self.context_builder = QueryContextBuilder(storage=storage, router=router, config=config)
        self.strategy_selector = StrategySelector(strategies)
        self.postprocessor = PostProcessor(reflection=reflection, config=config)
        self.persistence = ConversationPersistence(storage=storage)
        self.result_builder = ResultBuilder()
        self.stream_processor = StreamProcessor(
            strategy_selector=self.strategy_selector,
            postprocessor=self.postprocessor,
            result_builder=self.result_builder,
            persistence=self.persistence,
        )

    async def process_query(self, user_text: str, user_id: str, session_id: str) -> dict:
        context = await self.context_builder.build(user_text, user_id, session_id)
        strategy = await self.strategy_selector.select(context)
        logger.info(f"选择策略: {strategy.__class__.__name__}")

        result = await strategy.execute(context)
        result = await self.result_builder.finalize_result(result, context, self.postprocessor)
        payload = self.result_builder.build_response_payload(result)
        await self.persistence.persist_turn(context, result, payload)
        return payload

    async def process_query_stream(self, user_text: str, user_id: str, session_id: str):
        context = await self.context_builder.build(user_text, user_id, session_id)
        strategy = await self.strategy_selector.select(context)
        logger.info(f"选择策略(流式): {strategy.__class__.__name__}")

        step_queue = asyncio.Queue()
        async for event in self.stream_processor.process_with_context(context, step_queue):
            yield event
