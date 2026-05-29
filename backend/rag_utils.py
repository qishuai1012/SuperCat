from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import os
import json
import hashlib
import logging
import requests
from dotenv import load_dotenv

from milvus_client import MilvusManager
from embedding import embedding_service as _embedding_service
from parent_chunk_store import ParentChunkStore
from performance_config import get_performance_config
from query_understanding.service import get_query_understanding_service
from query_understanding.types import QueryComplexity, RetrievalStrategy
from smart_cache import get_smart_cache

from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)

# 这是 RAG 检索核心引擎
# 作用：用户问问题 → 去向量库检索 → 重排 → 自动合并小文档 → 返回最相关资料

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()
_smart_cache = get_smart_cache()

_stepback_model = None

# 服务启动一次性初始化 Milvus 集合（修复：每次检索重复init）
_milvus_initialized = False

def _ensure_milvus_initialized():
    """确保Milvus集合已初始化"""
    global _milvus_initialized
    if not _milvus_initialized:
        try:
            _milvus_manager.init_collection()
            _milvus_initialized = True
        except Exception as e:
            logger.warning("Milvus 集合初始化失败: %s", e)
            raise


def _chunk_key(doc: dict) -> tuple:
    return (doc.get("chunk_id") or "", doc.get("filename") or "", doc.get("page_number") or "", doc.get("text") or "")


def _doc_identifier(doc: dict) -> str:
    stable_source = "|".join([
        str(doc.get("chunk_id") or ""),
        str(doc.get("parent_chunk_id") or ""),
        str(doc.get("root_chunk_id") or ""),
        str(doc.get("filename") or ""),
        str(doc.get("page_number") or ""),
        str(doc.get("text") or "")[:200],
    ])
    return hashlib.sha1(stable_source.encode("utf-8")).hexdigest()[:16]


def _rank_fusion_score(doc: dict, position: int) -> float:
    base_score = float(doc.get("score") or 0.0)
    rerank_score = float(doc.get("rerank_score") or 0.0)
    rrf_rank = int(doc.get("rrf_rank") or position or 1)
    return base_score + rerank_score + (1.0 / (60 + rrf_rank))


def _merge_query_results(result_sets: List[Dict[str, Any]], top_k: int) -> tuple[list[dict], Dict[str, Any]]:
    fused: Dict[tuple, dict] = {}
    query_hits: Counter = Counter()
    for result in result_sets:
        docs = result.get("docs", []) or []
        variant = result.get("variant", "unknown")
        for pos, doc in enumerate(docs, 1):
            key = _chunk_key(doc)
            query_hits[variant] += 1
            current = fused.get(key)
            candidate = dict(doc)
            candidate["fused_query_variant"] = variant
            candidate["fusion_components"] = {
                "base_score": float(candidate.get("score") or 0.0),
                "rerank_score": float(candidate.get("rerank_score") or 0.0),
                "rrf_rank": int(candidate.get("rrf_rank") or pos or 1),
            }
            candidate["fusion_components"]["rrf_bonus"] = round(1.0 / (60 + candidate["fusion_components"]["rrf_rank"]), 8)
            candidate["fusion_score"] = _rank_fusion_score(candidate, pos)
            source_info = {
                "variant": variant,
                "doc_id": _doc_identifier(candidate),
                "position": pos,
                "score": candidate["fusion_score"],
            }
            if current is None or candidate["fusion_score"] > current.get("fusion_score", 0.0):
                merged_sources = list(current.get("fused_sources") or []) if current is not None else []
                merged_sources.append(source_info)
                merged_variants = list(current.get("fused_from_variants") or []) if current is not None else []
                merged_variants.append(variant)
                candidate["fused_from_variants"] = list(dict.fromkeys(merged_variants))
                candidate["fused_sources"] = merged_sources
                fused[key] = candidate
            else:
                current["fused_from_variants"] = list(dict.fromkeys((current.get("fused_from_variants") or []) + [variant]))
                current["fused_sources"] = list(current.get("fused_sources") or []) + [source_info]
    fused_docs = sorted(
        fused.values(),
        key=lambda item: (
            float(item.get("fusion_score", item.get("score", 0.0)) or 0.0),
            len(item.get("fused_from_variants") or []),
            -int(item.get("fusion_components", {}).get("rrf_rank") or item.get("rrf_rank") or 0),
        ),
        reverse=True,
    )[:top_k]
    for idx, item in enumerate(fused_docs, 1):
        item["fusion_rank"] = idx
    return fused_docs, {
        "multi_query_enabled": len(result_sets) > 1,
        "multi_query_variants": [item.get("variant") for item in result_sets],
        "multi_query_docs_total": sum(len(item.get("docs", []) or []) for item in result_sets),
        "multi_query_docs_unique": len(fused),
        "multi_query_docs_returned": len(fused_docs),
        "multi_query_hits": dict(query_hits),
    }

#获取重排序接口地址
def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"

#自动合并文档   小块数量 ≥ 阈值 → 自动替换成父块（更大、更完整）
def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    # 创建一个分组字典：key=父块ID，value=属于这个父块的所有小块
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    # 筛选出：需要合并的父块ID
    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    # 去父块存储里，批量查询这些父块的完整内容
    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    #parent_map 存储父块的id和内容
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    #最终合并好的块列表
    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        child_score = float(doc.get("score", 0.0) or 0.0)
        parent_score = float(parent_doc.get("score", 0.0) or 0.0)
        child_fusion_score = float(doc.get("fusion_score", child_score) or child_score)
        parent_fusion_score = float(parent_doc.get("fusion_score", parent_score) or parent_score)
        parent_doc["score"] = max(parent_score, child_score)
        parent_doc["fusion_score"] = max(parent_fusion_score, child_fusion_score)
        parent_doc["rerank_score"] = float(doc.get("rerank_score", parent_doc.get("rerank_score", 0.0)) or 0.0)
        parent_doc["rrf_rank"] = int(doc.get("rrf_rank") or parent_doc.get("rrf_rank") or 0)
        parent_doc["fused_from_variants"] = list(doc.get("fused_from_variants") or [])
        parent_doc["fused_sources"] = list(doc.get("fused_sources") or [])
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        parent_doc["merged_child_ids"] = [child.get("chunk_id") for child in groups[parent_id] if child.get("chunk_id")]
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count

#两层合并（L3→L2 → L2→L1）
def _auto_merge_documents(docs: List[dict], top_k: int, auto_merge_enabled: bool) -> Tuple[List[dict], Dict[str, Any]]:
    if not auto_merge_enabled or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": auto_merge_enabled,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 两段自动合并：L3->L2，再 L2->L1。
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("fusion_score", item.get("score", 0.0)), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": auto_merge_enabled,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }

 # 重排序（让最相关的排前面）
def _rerank_documents(query: str, docs: List[dict], top_k: int, rerank_requested: bool) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i, "rerank_source_score": doc.get("score", 0.0)} for i, doc in enumerate(docs, 1)]
    rerank_available = bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST)
    meta: Dict[str, Any] = {
        "rerank_requested": bool(rerank_requested),
        "rerank_available": rerank_available,
        "rerank_enabled": bool(rerank_requested and rerank_available),
        "rerank_applied": False,
        "rerank_skip_reason": None,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
        "rerank_fallback_used": False,
    }
    if not docs_with_rank:
        meta["rerank_skip_reason"] = "no_docs"
        return docs_with_rank[:top_k], meta
    if not rerank_requested:
        meta["rerank_skip_reason"] = "strategy_disabled"
        return docs_with_rank[:top_k], meta
    if not rerank_available:
        meta["rerank_skip_reason"] = "capability_unavailable"
        return docs_with_rank[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            meta["rerank_fallback_used"] = True
            return docs_with_rank[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        used_indexes = set()
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                used_indexes.add(idx)
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                doc["fusion_score"] = float(doc.get("score") or 0.0) + float(doc.get("rerank_score") or 0.0)
                reranked.append(doc)

        if reranked:
            meta["rerank_applied"] = True
            remaining = [dict(doc) for idx, doc in enumerate(docs_with_rank) if idx not in used_indexes]
            for doc in remaining:
                doc["fusion_score"] = float(doc.get("score") or 0.0)
            merged_reranked = reranked + remaining
            merged_reranked.sort(key=lambda item: float(item.get("fusion_score", item.get("score", 0.0)) or 0.0), reverse=True)
            return merged_reranked[:top_k], meta

    except requests.RequestException as e:
        meta["rerank_error"] = f"request_err: {str(e)}"
        meta["rerank_fallback_used"] = True
        logger.exception("重排序接口请求异常")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = f"parse_err: {str(e)}"
        meta["rerank_fallback_used"] = True
        logger.exception("重排序结果解析异常")
    return docs_with_rank[:top_k], meta

# 退步查询模型（StepBack）
def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""

#生成假设性文档（HyDE）
def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""

# 扩展查询（StepBack + 原始问题）
def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }

def _build_retrieval_cache_key_payload(query: str, query_analysis: Any, retrieval_config: Any, top_k: int, auto_merge_enabled: bool) -> Dict[str, Any]:
    normalized_query = get_smart_cache()._normalize_query_text(query)
    return {
        "query": query,
        "canonical_query": normalized_query,
        "top_k": top_k,
        "complexity": getattr(query_analysis.complexity, "value", str(getattr(query_analysis, "complexity", ""))),
        "domain": getattr(query_analysis, "domain", None),
        "intent_type": getattr(query_analysis, "intent_type", None),
        "strategy": getattr(retrieval_config.strategy, "value", str(getattr(retrieval_config, "strategy", ""))),
        "use_rerank": bool(getattr(retrieval_config, "use_rerank", False)),
        "auto_merge_enabled": auto_merge_enabled,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
        "rrf_threshold": float(os.getenv("RRF_SCORE_THRESHOLD", "0.022")),
        "raw_similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.65")),
    }


def build_retrieval_judgement(query: str, docs: List[dict], relevant_ids: List[str], k: int | None = None, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta = meta or {}
    retrieved_ids = [_doc_identifier(doc) for doc in docs]
    effective_k = int(k or len(retrieved_ids) or 0)
    return {
        "query": query,
        "k": effective_k,
        "relevant_ids": [str(item) for item in relevant_ids if item is not None],
        "retrieved_ids": retrieved_ids,
        "retrieved_count": len(retrieved_ids),
        "retrieval_mode": meta.get("retrieval_mode"),
        "multi_query_enabled": bool(meta.get("multi_query_enabled")),
        "multi_query_variants": list(meta.get("multi_query_variants") or []),
        "rerank_applied": bool(meta.get("rerank_applied")),
        "rerank_fallback_used": bool(meta.get("rerank_fallback_used")),
        "auto_merge_applied": bool(meta.get("auto_merge_applied")),
    }


def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    understanding = get_query_understanding_service().analyze_for_retrieval(query)
    query_analysis = understanding.query_analysis
    retrieval_config = understanding.retrieval_config
    top_k = retrieval_config.top_k
    auto_merge_enabled = retrieval_config.strategy in (
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.ADAPTIVE,
        RetrievalStrategy.MULTI_STAGE,
    )

    strategy_config = get_performance_config().get_strategy(query_analysis.complexity)
    cache_enabled = strategy_config.enable_cache and query_analysis.complexity in (QueryComplexity.SIMPLE, QueryComplexity.MEDIUM)
    cache_key_payload = _build_retrieval_cache_key_payload(query, query_analysis, retrieval_config, top_k, auto_merge_enabled)
    if cache_enabled:
        cached_result = _smart_cache.get_retrieval_result_by_key(cache_key_payload)
        if cached_result is None:
            cached_result = _smart_cache.get_semantic_retrieval_result_by_key(cache_key_payload)
            cache_mode = "semantic"
        else:
            cache_mode = "structured"
        if cached_result is not None:
            cached_meta = dict(cached_result.get("meta") or {})
            cached_meta["cache_hit"] = True
            cached_meta["cache_key_mode"] = cache_mode
            return {
                "docs": cached_result.get("docs") or [],
                "meta": cached_meta,
            }

    # ========== 相似度阈值配置 ==========
    RRF_THRESHOLD = float(os.getenv("RRF_SCORE_THRESHOLD", "0.022"))
    RAW_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
    # =====================================

    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    _ensure_milvus_initialized()

    try:
        logger.info(
            f"🔍 检索开始: 查询复杂度={query_analysis.complexity.value}, "
            f"扩展所有权=rag_pipeline, rerank_requested={retrieval_config.use_rerank}"
        )

        query_variants = [{"variant": "original", "query": query}]
        if query_analysis.complexity != QueryComplexity.SIMPLE:
            step_back = step_back_expand(query)
            if step_back.get("step_back_question") or step_back.get("step_back_answer"):
                query_variants.append({
                    "variant": "step_back",
                    "query": step_back.get("expanded_query") or query,
                    "step_back_question": step_back.get("step_back_question", ""),
                    "step_back_answer": step_back.get("step_back_answer", ""),
                })
            hyde_doc = generate_hypothetical_document(query)
            if hyde_doc:
                query_variants.append({"variant": "hyde", "query": hyde_doc, "hypothetical_doc": hyde_doc})

        if query_analysis.complexity in (QueryComplexity.MEDIUM, QueryComplexity.COMPLEX_LIGHT, QueryComplexity.COMPLEX_HEAVY):
            synonym_query = query
            if query_analysis.intent_type:
                synonym_query = f"{query}\n\n同义表达: {query_analysis.intent_type}"
            query_variants.append({"variant": "synonyms", "query": synonym_query})

        result_sets: List[Dict[str, Any]] = []
        for variant in query_variants:
            search_query = variant["query"]
            variant_name = variant["variant"]
            try:
                dense_embeddings = _embedding_service.get_embeddings([search_query])
                dense_embedding = dense_embeddings[0]
                sparse_embedding = _embedding_service.get_sparse_embedding(search_query)
                retrieved = _milvus_manager.hybrid_retrieve(
                    dense_embedding=dense_embedding,
                    sparse_embedding=sparse_embedding,
                    top_k=max(1, candidate_k // len(query_variants)),
                    filter_expr=filter_expr,
                )
                result_sets.append({"variant": variant_name, "docs": retrieved})
            except Exception as e:
                logger.warning(f"检索查询 '{search_query[:50]}...' 失败: {e}")
                try:
                    dense_embeddings = _embedding_service.get_embeddings([search_query])
                    dense_embedding = dense_embeddings[0]
                    dense_results = _milvus_manager.dense_retrieve(
                        dense_embedding=dense_embedding,
                        top_k=max(1, candidate_k // len(query_variants)),
                        filter_expr=filter_expr,
                    )
                    result_sets.append({"variant": variant_name, "docs": dense_results})
                    logger.info(f"查询 '{search_query[:50]}...' 降级到密集检索成功")
                except Exception as e2:
                    logger.error(f"查询 '{search_query[:50]}...' 的降级检索也失败: {e2}")
                    result_sets.append({"variant": variant_name, "docs": []})

        fused_docs, fusion_meta = _merge_query_results(result_sets, top_k=max(top_k * 2, top_k))

        seen = set()
        unique_results = []
        for result in fused_docs:
            key = _chunk_key(result)
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(result)

        unique_results.sort(key=lambda x: x.get("fusion_score", x.get("score", 0.0)), reverse=True)

        def filter_by_similarity(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
            if not docs:
                return docs, {"filtered_count": 0, "filter_applied": False}

            score_types = {doc.get("score_type", "raw_similarity") for doc in docs}
            effective_score_type = "rrf" if "rrf" in score_types else "raw_similarity"
            threshold = RRF_THRESHOLD if effective_score_type == "rrf" else RAW_SIMILARITY_THRESHOLD

            filtered_docs = [d for d in docs if d.get("fusion_score", d.get("score", 0)) >= threshold]
            filtered_count = len(docs) - len(filtered_docs)

            logger.info(
                f"📊 相似度过滤: 原始{len(docs)}个, "
                f"阈值={threshold:.3f}({effective_score_type}), "
                f"过滤{filtered_count}个, 保留{len(filtered_docs)}个"
            )

            return filtered_docs, {
                "filter_applied": True,
                "filter_threshold": threshold,
                "filter_type": effective_score_type,
                "filter_score_types": sorted(score_types),
                "original_count": len(docs),
                "filtered_count": filtered_count,
                "retained_count": len(filtered_docs)
            }

        filtered_results, filter_meta = filter_by_similarity(unique_results)

        if not filtered_results:
            logger.warning("⚠️ 相似度过滤后无文档，知识库可能没有相关内容")
            result = {
                "docs": [],
                "meta": {
                    "retrieval_mode": "multi_query_fused",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "no_relevant_docs": True,
                    "reason": "all_docs_below_similarity_threshold",
                    "query_expansion_owner": "rag_pipeline",
                    "hyde_generated_count": sum(1 for item in query_variants if item["variant"] == "hyde"),
                    "step_back_generated": any(item["variant"] == "step_back" for item in query_variants),
                    "expanded_retrieval_count": len(query_variants),
                    **fusion_meta,
                    **filter_meta,
                    "rerank_requested": bool(retrieval_config.use_rerank),
                    "rerank_available": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_enabled": False,
                    "rerank_applied": False,
                    "rerank_skip_reason": "no_relevant_docs_before_rerank",
                    "auto_merge_enabled": auto_merge_enabled,
                    "auto_merge_applied": False,
                }
            }
            if cache_enabled:
                _smart_cache.set_retrieval_result_by_key(cache_key_payload, result, ttl=strategy_config.cache_ttl)
                _smart_cache.set_semantic_retrieval_result_by_key(cache_key_payload, result, ttl=strategy_config.cache_ttl)
            return result

        reranked, rerank_meta = _rerank_documents(
            query=query,
            docs=filtered_results,
            top_k=top_k,
            rerank_requested=retrieval_config.use_rerank,
        )
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k, auto_merge_enabled=auto_merge_enabled)

        rerank_meta["retrieval_mode"] = "multi_query_fused"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta["score_type"] = filter_meta.get("filter_type")
        rerank_meta["query_expansion_owner"] = "rag_pipeline"
        rerank_meta["hyde_generated_count"] = sum(1 for item in query_variants if item["variant"] == "hyde")
        rerank_meta["step_back_generated"] = any(item["variant"] == "step_back" for item in query_variants)
        rerank_meta["expanded_retrieval_count"] = len(query_variants)
        rerank_meta["dynamic_strategy"] = {
            "strategy": retrieval_config.strategy.value,
            "query_complexity": query_analysis.complexity.value,
            "domain": query_analysis.domain,
            "intent_type": query_analysis.intent_type
        }
        rerank_meta.update(fusion_meta)
        rerank_meta.update(filter_meta)
        rerank_meta.update(merge_meta)
        result = {"docs": merged_docs, "meta": rerank_meta}
        if cache_enabled:
            _smart_cache.set_retrieval_result_by_key(cache_key_payload, result, ttl=strategy_config.cache_ttl)
            _smart_cache.set_semantic_retrieval_result_by_key(cache_key_payload, result, ttl=strategy_config.cache_ttl)
        return result
    except Exception as e:
        logger.warning(f"混合检索失败，尝试降级到密集检索: {e}")
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(
                query=query,
                docs=retrieved,
                top_k=top_k,
                rerank_requested=retrieval_config.use_rerank,
            )
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k, auto_merge_enabled=auto_merge_enabled)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update({
                "score_type": "raw_similarity",
                "query_expansion_owner": "rag_pipeline",
                "hyde_generated_count": 0,
                "step_back_generated": False,
                "expanded_retrieval_count": 1,
                "dynamic_strategy": {
                    "strategy": retrieval_config.strategy.value,
                    "query_complexity": query_analysis.complexity.value,
                    "domain": query_analysis.domain,
                    "intent_type": query_analysis.intent_type,
                },
            })
            rerank_meta.update(merge_meta)
            result = {"docs": merged_docs, "meta": rerank_meta}
            if cache_enabled:
                _smart_cache.set_retrieval_result_by_key(cache_key_payload, result, ttl=strategy_config.cache_ttl)
                _smart_cache.set_semantic_retrieval_result_by_key(cache_key_payload, result, ttl=strategy_config.cache_ttl)
            return result
        except Exception as e2:
            logger.error(f"密集检索也失败: {e2}")
            return {
                "docs": [],
                "meta": {
                    "rerank_requested": bool(retrieval_config.use_rerank),
                    "rerank_available": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_enabled": bool(retrieval_config.use_rerank and RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_applied": False,
                    "rerank_skip_reason": "retrieve_failed",
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "query_expansion_owner": "rag_pipeline",
                    "hyde_generated_count": 0,
                    "step_back_generated": False,
                    "expanded_retrieval_count": 0,
                    "auto_merge_enabled": auto_merge_enabled,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }