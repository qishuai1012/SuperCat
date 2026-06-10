import asyncio
from dataclasses import asdict, is_dataclass
from enum import Enum
import logging
import time
import uuid

logger = logging.getLogger(__name__)


class ConversationPersistence:
    def __init__(self, storage):
        self.storage = storage

    def _serialize_for_json(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return self._serialize_for_json(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): self._serialize_for_json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_for_json(item) for item in value]
        return str(value)

    def _persist_memory(self, context, result, payload: dict | None = None) -> None:
        from core.memory_optimizer import add_memory, get_memory_optimizer

        tags = ["dialogue", context.session_id, context.complexity]
        strategy = result.metadata.get("strategy")
        if strategy:
            tags.append(strategy)

        add_memory(
            content=f"用户：{context.user_text}\n回答：{result.response}",
            importance=3,
            tags=tags,
        )

        if result.metadata.get("kb_no_result"):
            explanation = result.metadata.get("kb_no_result_explanation") or ""
            suggestions = result.metadata.get("kb_no_result_suggestions") or []
            note_parts = [f"知识库未命中：{context.user_text}"]
            if explanation:
                note_parts.append(f"原因：{explanation}")
            if suggestions:
                note_parts.append("建议：" + "；".join(str(item) for item in suggestions[:3]))
            add_memory(
                content="\n".join(note_parts),
                importance=2,
                tags=["kb_no_result", context.session_id, context.complexity],
            )

        get_memory_optimizer().run_memory_maintenance()

    def _collect_learning_metrics(self, context, result, payload: dict | None = None) -> None:
        from monitoring.learning_system import PerformanceMetrics, get_online_learning_system

        started_at = context.started_at or time.perf_counter()
        response_time = max(time.perf_counter() - started_at, 0.0)
        strategy = str(result.metadata.get("strategy") or context.execution_class or "unknown")
        rag_trace = payload.get("rag_trace") if payload else result.rag_trace
        kb_no_result = bool(result.metadata.get("kb_no_result"))
        success = bool(result.response) and not bool(result.metadata.get("orchestration_error"))
        relevance_score = 0.2 if kb_no_result else 0.8
        user_satisfaction = 0.3 if kb_no_result else (0.7 if success else 0.4)
        retrieved_chunks = []
        if isinstance(rag_trace, dict):
            retrieved_chunks = rag_trace.get("retrieved_chunks") or rag_trace.get("expanded_retrieved_chunks") or rag_trace.get("initial_retrieved_chunks") or []
        retrieval_eval = {
            "status": "runtime_observation",
            "query": context.user_text,
            "kb_no_result": kb_no_result,
            "retrieved_count": len(retrieved_chunks),
            "retrieved_ids": [
                item.get("chunk_id") or item.get("parent_chunk_id") or item.get("root_chunk_id") or f"{item.get('filename', '')}:{item.get('page_number', '')}"
                for item in retrieved_chunks
                if isinstance(item, dict)
            ],
            "multi_query_enabled": bool((rag_trace or {}).get("multi_query_enabled")) if isinstance(rag_trace, dict) else False,
            "multi_query_variants": list((rag_trace or {}).get("multi_query_variants") or []) if isinstance(rag_trace, dict) else [],
            "retrieval_mode": (rag_trace or {}).get("retrieval_mode") if isinstance(rag_trace, dict) else None,
            "score_type": (rag_trace or {}).get("score_type") if isinstance(rag_trace, dict) else None,
            "rerank_applied": bool((rag_trace or {}).get("rerank_applied")) if isinstance(rag_trace, dict) else False,
            "rerank_fallback_used": bool((rag_trace or {}).get("rerank_fallback_used")) if isinstance(rag_trace, dict) else False,
            "auto_merge_applied": bool((rag_trace or {}).get("auto_merge_applied")) if isinstance(rag_trace, dict) else False,
            "candidate_count": (rag_trace or {}).get("candidate_count") if isinstance(rag_trace, dict) else None,
            "retained_count": (rag_trace or {}).get("retained_count") if isinstance(rag_trace, dict) else None,
        }

        learning_metadata = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "complexity": context.complexity,
            "execution_class": context.execution_class,
            "kb_no_result": kb_no_result,
            "route_decision": self._serialize_for_json(result.metadata.get("route_decision")),
        }

        if rag_trace is not None:
            learning_metadata["rag_trace"] = self._serialize_for_json(rag_trace)

        metrics = PerformanceMetrics(
            query_id=context.query_id or str(uuid.uuid4()),
            response_time=response_time,
            relevance_score=relevance_score,
            user_satisfaction=user_satisfaction,
            strategy_used=strategy,
            config_used=learning_metadata,
            success=success,
            error_message=str(result.metadata.get("orchestration_error") or ""),
            retrieval_eval=retrieval_eval,
        )
        get_online_learning_system().collect_performance_metrics(metrics)

    def _collect_observability_metrics(self, context, result, payload: dict | None = None) -> None:
        from monitoring.performance_config import get_performance_config
        from monitoring.performance_monitor import get_performance_monitor

        if not get_performance_config().enable_performance_monitoring or not context.query_id:
            return

        monitor = get_performance_monitor()
        metrics = monitor.get_metrics(context.query_id)
        if metrics is None:
            return

        rag_trace = payload.get("rag_trace") if payload else result.rag_trace
        strategy = str(result.metadata.get("strategy") or context.execution_class or "unknown")
        metrics.complexity = context.complexity
        metrics.strategy_used = strategy
        metrics.api_calls = 1

        finalize_step = metrics.add_step("finalize_response")
        finalize_step.metadata.update({
            "kb_no_result": bool(result.metadata.get("kb_no_result")),
            "orchestration_success": bool(result.metadata.get("orchestration_success", True)),
            "rag_trace_present": rag_trace is not None,
        })
        finalize_step.finish(success=not bool(result.metadata.get("orchestration_error")), error=str(result.metadata.get("orchestration_error") or "") or None)

        monitor.finish_query(context.query_id, success=not bool(result.metadata.get("orchestration_error")))

    async def persist_turn(self, context, result, payload: dict | None = None) -> None:
        def _persist_sync():
            from langchain_core.messages import AIMessage, HumanMessage

            try:
                self._persist_memory(context, result, payload)
            except Exception as e:
                logger.warning(f"记忆保存失败: {e}")

            try:
                self._collect_learning_metrics(context, result, payload)
            except Exception as e:
                logger.warning(f"学习指标保存失败: {e}")

            try:
                self._collect_observability_metrics(context, result, payload)
            except Exception as e:
                logger.warning(f"可观测性指标保存失败: {e}")

            try:
                from monitoring.adaptive_tuning import get_adaptive_tuner
                asyncio.run(get_adaptive_tuner().optimize_if_needed())
            except Exception as e:
                logger.warning(f"自适应调优执行失败: {e}")

            messages = self.storage.load(context.user_id, context.session_id)
            messages.append(HumanMessage(content=context.user_text))
            messages.append(AIMessage(content=result.response or ""))

            rag_trace = payload.get("rag_trace") if payload else result.rag_trace
            extra_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
            self.storage.save(context.user_id, context.session_id, messages, extra_message_data=extra_data)

        try:
            await asyncio.to_thread(_persist_sync)
        except Exception as e:
            logger.warning(f"对话收尾持久化失败: {e}")

    async def persist_after_stream(self, context, result, payload: dict | None = None) -> None:
        await self.persist_turn(context, result, payload)
