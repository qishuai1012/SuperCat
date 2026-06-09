from langchain_core.messages import AIMessageChunk
import asyncio

from agent.strategies.base import ExecutionResult
from tools import get_last_rag_context, reset_tool_call_guards, set_rag_options, set_rag_step_queue


class StreamProcessor:
    def __init__(self, strategy_selector, postprocessor, result_builder, persistence):
        self.strategy_selector = strategy_selector
        self.postprocessor = postprocessor
        self.result_builder = result_builder
        self.persistence = persistence

    async def process(self, user_text: str, context_builder):
        raise NotImplementedError

    async def process_with_context(self, context, step_queue):
        set_rag_step_queue(step_queue)
        output_queue = asyncio.Queue()

        async def _drain_rag_steps():
            """持续把 rag step 转发到 output_queue，直到收到 None 哨兵"""
            while True:
                step = await step_queue.get()
                if step is None:
                    break
                await output_queue.put({"type": "rag_step", "step": step})

        drain_task = asyncio.create_task(_drain_rag_steps())
        try:
            strategy = await self.strategy_selector.select(context)

            if strategy.get_priority() == 1:
                messages = await strategy.prepare_messages(context)
                get_last_rag_context(clear=True)
                reset_tool_call_guards()
                set_rag_options(expansion_hint=context.expansion_hint)

                agent, _ = strategy.agent_factory.create_or_get()
                response_parts = []
                kb_no_result_detected = False

                async def _run_agent():
                    async for chunk in agent.astream({"messages": messages}, config={"recursion_limit": 8}):
                        await output_queue.put(chunk)
                    await output_queue.put(None)  # agent 结束哨兵

                agent_task = asyncio.create_task(_run_agent())

                while True:
                    item = await output_queue.get()
                    if item is None:
                        break
                    if isinstance(item, dict) and item.get("type") == "rag_step":
                        yield item
                        continue
                    chunk = item
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        if text:
                            response_parts.append(text)
                            kb_no_result_detected = kb_no_result_detected or strategy._contains_kb_no_result_signal(text)
                            if not kb_no_result_detected:
                                yield {"type": "content", "content": text}
                    elif isinstance(chunk, dict):
                        agent_data = chunk.get("agent") or chunk.get("model") or {}
                        msgs = agent_data.get("messages", []) if isinstance(agent_data, dict) else []
                        if msgs:
                            last = msgs[-1]
                            raw = getattr(last, "content", "")
                            if isinstance(raw, list):
                                text = " ".join(
                                    p.get("text", "") if isinstance(p, dict) else str(p)
                                    for p in raw
                                ).strip()
                            else:
                                text = str(raw).strip()
                            if text and text not in response_parts:
                                kb_no_result_detected = kb_no_result_detected or strategy._contains_kb_no_result_signal(text)
                                response_parts.append(text)
                                if not kb_no_result_detected:
                                    for i in range(0, len(text), 40):
                                        yield {"type": "content", "content": text[i:i+40]}

                await agent_task
                # 停止 drain 协程
                await step_queue.put(None)
                await drain_task
                # 排空剩余 rag steps
                while not output_queue.empty():
                    item = output_queue.get_nowait()
                    if isinstance(item, dict) and item.get("type") == "rag_step":
                        yield item

                rag_context = dict(get_last_rag_context(clear=True) or {})
                if kb_no_result_detected:
                    rag_context["kb_no_result"] = True
                normalized_response = strategy._normalize_response("".join(response_parts), rag_context)
                if rag_context.get("kb_no_result"):
                    for i in range(0, len(normalized_response), 40):
                        yield {"type": "content", "content": normalized_response[i:i + 40]}
                result = ExecutionResult(
                    response=normalized_response,
                    rag_trace=rag_context.get("rag_trace"),
                    metadata={
                        "strategy": "simple",
                        "route_decision": context.route_decision,
                        "kb_no_result": bool(rag_context.get("kb_no_result")),
                        "kb_no_result_reason": rag_context.get("kb_no_result_reason"),
                        "kb_no_result_explanation": rag_context.get("kb_no_result_explanation"),
                        "kb_no_result_suggestions": rag_context.get("kb_no_result_suggestions"),
                    },
                )
            else:
                await step_queue.put(None)
                await drain_task
                result = await strategy.execute(context)
                for i in range(0, len(result.response), 40):
                    yield {"type": "content", "content": result.response[i:i + 40]}

            result = await self.result_builder.finalize_result(result, context, self.postprocessor)
            finalized = self.result_builder.build_response_payload(result)
            if finalized.get("rag_trace") is not None:
                yield {"type": "trace", "rag_trace": finalized.get("rag_trace")}
            yield {
                "type": "_done",
                "response": finalized.get("response", ""),
                "rag_trace": finalized.get("rag_trace"),
                "agentic_info": finalized.get("agentic_info"),
                "kb_no_result": bool(finalized.get("kb_no_result", False)),
            }
        except Exception:
            if not drain_task.done():
                await step_queue.put(None)
                drain_task.cancel()
            raise
        finally:
            set_rag_step_queue(None)

        asyncio.create_task(self.persistence.persist_after_stream(context, result, finalized))
