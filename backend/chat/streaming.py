from langchain_core.messages import AIMessageChunk

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
                async for chunk in agent.astream({"messages": messages}, config={"recursion_limit": 8}):
                    if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
                        response_parts.append(chunk.content)
                        kb_no_result_detected = kb_no_result_detected or strategy._contains_kb_no_result_signal(chunk.content)
                        rag_context_snapshot = dict(get_last_rag_context(clear=False) or {})
                        if kb_no_result_detected:
                            rag_context_snapshot["kb_no_result"] = True
                        if not rag_context_snapshot.get("kb_no_result"):
                            yield {"type": "content", "content": chunk.content}
                    elif isinstance(chunk, dict):
                        kb_no_result_detected = kb_no_result_detected or strategy._contains_kb_no_result_signal(chunk)
                        model_data = chunk.get("model", {})
                        messages_list = model_data.get("messages", []) if isinstance(model_data, dict) else []
                        if messages_list:
                            ai_msg = messages_list[-1]
                            content = getattr(ai_msg, "content", "")
                            if content:
                                content = str(content)
                                response_parts.append(content)
                                rag_context_snapshot = dict(get_last_rag_context(clear=False) or {})
                                if kb_no_result_detected:
                                    rag_context_snapshot["kb_no_result"] = True
                                if not rag_context_snapshot.get("kb_no_result"):
                                    for i in range(0, len(content), 40):
                                        yield {"type": "content", "content": content[i:i + 40]}

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
        finally:
            set_rag_step_queue(None)

        import asyncio
        asyncio.create_task(self.persistence.persist_after_stream(context, result, finalized))
