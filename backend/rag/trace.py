from typing import Any, List


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)



def merge_rag_trace(trace: dict | None = None, **updates: Any) -> dict:
    merged = dict(trace or {})
    merged.update(updates)
    return merged



def build_retrieval_trace(
    query: str,
    docs: List[dict],
    retrieval_meta: dict | None = None,
    retrieval_stage: str = "initial",
    expanded_query: str | None = None,
    trace: dict | None = None,
) -> dict:
    retrieval_meta = retrieval_meta or {}
    expanded_query = expanded_query or query
    stage_updates = {
        "initial_retrieved_chunks": docs,
    } if retrieval_stage == "initial" else {
        "expanded_retrieved_chunks": docs,
    }
    score_values = [float(doc.get("fusion_score", doc.get("score", 0.0)) or 0.0) for doc in docs]
    score_stats = {
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
        "score_avg": (sum(score_values) / len(score_values)) if score_values else None,
    }
    trace = trace or {}
    stage_durations_ms = dict(trace.get("stage_durations_ms") or {})
    if retrieval_meta.get("retrieval_duration_ms") is not None:
        stage_durations_ms[f"{retrieval_stage}_retrieval"] = retrieval_meta.get("retrieval_duration_ms")
    model_versions = dict(trace.get("model_versions") or {})
    if retrieval_meta.get("rerank_model"):
        model_versions["rerank_model"] = retrieval_meta.get("rerank_model")
    hit_reason = retrieval_meta.get("hit_reason")
    if not hit_reason:
        if retrieval_meta.get("kb_no_result") or retrieval_meta.get("no_relevant_docs"):
            hit_reason = "no_relevant_docs"
        elif retrieval_meta.get("rerank_applied"):
            hit_reason = "reranked_top_chunks"
        elif retrieval_meta.get("multi_query_enabled"):
            hit_reason = "multi_query_fusion"
        else:
            hit_reason = "top_similarity_chunks"
    diagnostic_summary = retrieval_meta.get("diagnostic_summary") or (
        f"stage={retrieval_stage}, mode={retrieval_meta.get('retrieval_mode')}, docs={len(docs)}, hit_reason={hit_reason}"
    )
    return merge_rag_trace(
        trace,
        tool_used=True,
        tool_name="search_knowledge_base",
        query=query,
        expanded_query=expanded_query,
        retrieved_chunks=docs,
        retrieval_stage=retrieval_stage,
        query_expansion_owner=retrieval_meta.get("query_expansion_owner", "rag_pipeline"),
        hyde_generated_count=retrieval_meta.get("hyde_generated_count", 0),
        step_back_generated=retrieval_meta.get("step_back_generated", False),
        expanded_retrieval_count=retrieval_meta.get("expanded_retrieval_count", 1),
        rerank_requested=retrieval_meta.get("rerank_requested"),
        rerank_available=retrieval_meta.get("rerank_available"),
        rerank_enabled=retrieval_meta.get("rerank_enabled"),
        rerank_applied=retrieval_meta.get("rerank_applied"),
        rerank_fallback_used=retrieval_meta.get("rerank_fallback_used"),
        rerank_skip_reason=retrieval_meta.get("rerank_skip_reason"),
        rerank_model=retrieval_meta.get("rerank_model"),
        rerank_endpoint=retrieval_meta.get("rerank_endpoint"),
        rerank_error=retrieval_meta.get("rerank_error"),
        retrieval_mode=retrieval_meta.get("retrieval_mode"),
        candidate_k=retrieval_meta.get("candidate_k"),
        candidate_count=retrieval_meta.get("candidate_count"),
        leaf_retrieve_level=retrieval_meta.get("leaf_retrieve_level"),
        auto_merge_enabled=retrieval_meta.get("auto_merge_enabled"),
        auto_merge_applied=retrieval_meta.get("auto_merge_applied"),
        auto_merge_threshold=retrieval_meta.get("auto_merge_threshold"),
        auto_merge_replaced_chunks=retrieval_meta.get("auto_merge_replaced_chunks"),
        auto_merge_steps=retrieval_meta.get("auto_merge_steps"),
        no_relevant_docs=bool(retrieval_meta.get("no_relevant_docs", False)),
        kb_no_result=bool(retrieval_meta.get("kb_no_result", retrieval_meta.get("no_relevant_docs", False))),
        reason=retrieval_meta.get("reason"),
        score_type=retrieval_meta.get("score_type") or retrieval_meta.get("filter_type"),
        filter_applied=retrieval_meta.get("filter_applied"),
        filter_threshold=retrieval_meta.get("filter_threshold"),
        filter_type=retrieval_meta.get("filter_type"),
        filter_score_types=retrieval_meta.get("filter_score_types"),
        original_count=retrieval_meta.get("original_count"),
        filtered_count=retrieval_meta.get("filtered_count"),
        retained_count=retrieval_meta.get("retained_count"),
        dynamic_strategy=retrieval_meta.get("dynamic_strategy"),
        multi_query_enabled=retrieval_meta.get("multi_query_enabled"),
        multi_query_variants=retrieval_meta.get("multi_query_variants"),
        multi_query_docs_total=retrieval_meta.get("multi_query_docs_total"),
        multi_query_docs_unique=retrieval_meta.get("multi_query_docs_unique"),
        multi_query_docs_returned=retrieval_meta.get("multi_query_docs_returned"),
        multi_query_hits=retrieval_meta.get("multi_query_hits"),
        stage_durations_ms=stage_durations_ms or None,
        retrieval_duration_ms=retrieval_meta.get("retrieval_duration_ms"),
        rewrite_duration_ms=retrieval_meta.get("rewrite_duration_ms"),
        grading_duration_ms=retrieval_meta.get("grading_duration_ms"),
        rerank_duration_ms=retrieval_meta.get("rerank_duration_ms"),
        model_versions=model_versions or None,
        hit_reason=hit_reason,
        diagnostic_summary=diagnostic_summary,
        **score_stats,
        **stage_updates,
    )


__all__ = ["_format_docs", "merge_rag_trace", "build_retrieval_trace"]
