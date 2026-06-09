"""简单查询策略"""
import logging
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from .base import ExecutionStrategy, QueryContext, ExecutionResult
from tools import (
    KB_NO_RESULT_MESSAGE,
    KB_NO_RESULT_SENTINEL,
    get_last_rag_context,
    reset_tool_call_guards,
    set_rag_options,
)

logger = logging.getLogger(__name__)


class SimpleQueryStrategy(ExecutionStrategy):
    """简单查询策略：直接调用Agent"""

    execution_class = "simple"

    def __init__(self, agent_factory, storage):
        self.agent_factory = agent_factory
        self.storage = storage

    def _contains_kb_no_result_signal(self, value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().startswith(f"{KB_NO_RESULT_SENTINEL}:")
        if isinstance(value, list):
            return any(self._contains_kb_no_result_signal(item) for item in value)
        if isinstance(value, dict):
            return any(self._contains_kb_no_result_signal(item) for item in value.values())
        content = getattr(value, "content", None)
        if content is not None:
            return self._contains_kb_no_result_signal(content)
        return False

    async def can_handle(self, context: QueryContext) -> bool:
        if context.execution_class:
            return context.execution_class == self.execution_class
        return context.complexity in ("simple", "medium")

    async def execute(self, context: QueryContext) -> ExecutionResult:
        messages = await self.prepare_messages(context)

        # 清理状态
        get_last_rag_context(clear=True)
        reset_tool_call_guards()
        set_rag_options(expansion_hint=context.expansion_hint)

        # 执行
        agent, _ = self.agent_factory.create_or_get()
        response_parts = []
        kb_no_result_detected = False
        async for chunk in agent.astream({"messages": messages}, config={"recursion_limit": 8}):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                response_parts.append(chunk.content)
                kb_no_result_detected = kb_no_result_detected or self._contains_kb_no_result_signal(chunk.content)
            elif isinstance(chunk, dict):
                kb_no_result_detected = kb_no_result_detected or self._contains_kb_no_result_signal(chunk)
                model_data = chunk.get("model", {})
                messages_list = model_data.get("messages", []) if isinstance(model_data, dict) else []
                if messages_list:
                    ai_msg = messages_list[-1]
                    content = getattr(ai_msg, "content", "")
                    if content:
                        response_parts.append(str(content))

        rag_context = dict(get_last_rag_context(clear=True) or {})
        if kb_no_result_detected:
            rag_context["kb_no_result"] = True
        response = self._normalize_response("".join(response_parts), rag_context)

        return ExecutionResult(
            response=response,
            rag_trace=rag_context.get("rag_trace"),
            metadata={
                "strategy": "simple",
                "route_decision": context.route_decision,
                "kb_no_result": bool(rag_context.get("kb_no_result")),
                "kb_no_result_reason": rag_context.get("kb_no_result_reason"),
                "kb_no_result_explanation": rag_context.get("kb_no_result_explanation"),
                "kb_no_result_suggestions": rag_context.get("kb_no_result_suggestions"),
            }
        )

    def get_priority(self) -> int:
        return 1

    def _extract_response(self, result):
        if isinstance(result, dict):
            response = result.get("output", "")
            if not response and result.get("messages"):
                msg = result["messages"][-1]
                response = getattr(msg, "content", str(msg))
        elif hasattr(result, "content"):
            response = result.content
        else:
            response = str(result)
        return response or "无法生成回答"

    async def _summarize_messages(self, messages):
        try:
            _, model = self.agent_factory.create_or_get()
            text = "\n".join(f"{'用户' if m.type=='human' else 'AI'}: {m.content}" for m in messages[:30])
            prompt = f"请总结以下对话的核心内容：\n{text}\n总结："
            result = await model.ainvoke(prompt)
            return result.content
        except Exception as e:
            logger.warning(f"摘要生成失败: {e}")
            return "对话过长，已自动摘要"

    async def prepare_messages(self, context):
        """为执行前准备消息（同步/流式共用）"""
        messages = self.storage.load(context.user_id, context.session_id)

        # 注入记忆
        try:
            from memory_optimizer import retrieve_memories
            memories = retrieve_memories(context.user_text, {
                "user_id": context.user_id,
                "session_id": context.session_id
            })
            if memories:
                memory_texts = [f"- {m.content.strip()}" for m in memories[:5]]
                messages.insert(0, SystemMessage(
                    content="历史记忆：\n" + "\n".join(memory_texts)
                ))
        except Exception as e:
            logger.warning(f"记忆注入失败: {e}")

        # 长对话摘要
        if len(messages) > 50:
            summary = await self._summarize_messages(messages)
            messages = [SystemMessage(content=f"对话摘要：{summary}")] + messages[-10:]

        messages.append(HumanMessage(content=context.user_text))
        return messages

    def _build_kb_no_result_response(self, rag_context: dict) -> str:
        explanation = (rag_context.get("kb_no_result_explanation") or "").strip()
        suggestions = list(rag_context.get("kb_no_result_suggestions") or [])[:3]
        lines = [KB_NO_RESULT_MESSAGE]

        if explanation:
            lines.extend(["", f"原因：{explanation}"])

        if suggestions:
            lines.extend(["", "建议你这样重试："])
            lines.extend([f"{idx}. {item}" for idx, item in enumerate(suggestions, 1)])

        return "\n".join(lines)

    def _normalize_response(self, response: str, rag_context: dict = None) -> str:
        rag_context = rag_context or {}
        if rag_context.get("kb_no_result"):
            return self._build_kb_no_result_response(rag_context)

        response = (response or "").strip()
        if response.startswith(f"{KB_NO_RESULT_SENTINEL}:"):
            return self._build_kb_no_result_response(rag_context)
        return response or "无法生成回答"
