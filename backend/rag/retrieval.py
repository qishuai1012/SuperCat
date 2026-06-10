from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import os
import json
import hashlib
import logging
import requests
from dotenv import load_dotenv

# 引入项目内部核心模块
from storage.milvus_client import MilvusManager
from storage.embedding import embedding_service as _embedding_service
from storage.parent_chunk_store import ParentChunkStore
from monitoring.performance_config import get_performance_config
from query_understanding.service import get_query_understanding_service
from query_understanding.types import QueryComplexity, RetrievalStrategy
from storage.cache import get_smart_cache

from langchain.chat_models import init_chat_model

# 日志记录器，用于生产环境排查问题
logger = logging.getLogger(__name__)

# ===================== 核心说明 =====================
# 这是 **RAG 系统的检索核心引擎**
# 功能：用户提问 → 多路查询扩展 → 混合检索 → 结果融合 → 重排序 → Auto-Merging → 返回最终参考文档
# ====================================================

load_dotenv()

# ===================== 环境变量配置（生产级解耦） =====================
ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

AUGMENT_LLM_API_KEY = os.getenv("AUGMENT_LLM_API_KEY")
AUGMENT_LLM_MODEL = os.getenv("AUGMENT_LLM_MODEL")
AUGMENT_LLM_BASE_URL = os.getenv("AUGMENT_LLM_BASE_URL")

RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")

# Auto-Merging 开关：是否启用父子块自动合并
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
# Auto-Merging 阈值：同一个父块下命中多少小块才触发合并
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
# 只检索叶子块（Level3），不检索父块，保证检索精度
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# ===================== 全局单例初始化 =====================
# 与 API 服务共用同一个 embedding_service，保证 BM25 词表全局一致
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()
_smart_cache = get_smart_cache()

_stepback_model = None

# 防止服务启动时重复初始化 Milvus 集合（生产级优化）
_milvus_initialized = False

def _ensure_milvus_initialized():
    """
    【生产级】确保 Milvus 集合只初始化一次
    避免重复建表、启动报错
    """
    global _milvus_initialized
    if not _milvus_initialized:
        try:
            _milvus_manager.init_collection()
            _milvus_initialized = True
        except Exception as e:
            logger.warning("Milvus 集合初始化失败: %s", e)
            raise

# ===================== 文档唯一标识 / 去重工具 =====================
def _chunk_key(doc: dict) -> tuple:
    """
    生成文档唯一键，用于去重
    依据：chunk_id + 文件名 + 页码 + 文本前序内容
    """
    return (
        doc.get("chunk_id") or "",
        doc.get("filename") or "",
        doc.get("page_number") or "",
        doc.get("text") or ""
    )

def _doc_identifier(doc: dict) -> str:
    """
    生成文档唯一哈希ID，用于追踪、评估检索效果
    生产环境用于计算召回率、准确率
    """
    stable_source = "|".join([
        str(doc.get("chunk_id") or ""),
        str(doc.get("parent_chunk_id") or ""),
        str(doc.get("root_chunk_id") or ""),
        str(doc.get("filename") or ""),
        str(doc.get("page_number") or ""),
        str(doc.get("text") or "")[:200],
    ])
    return hashlib.sha1(stable_source.encode("utf-8")).hexdigest()[:16]

# ===================== 多路检索结果融合打分 =====================
def _rank_fusion_score(doc: dict, position: int) -> float:
    """
    自定义融合打分公式：
    基础检索分数 + 重排分数 + RRF 排名奖励分
    让多路检索结果公平排序
    """
    base_score = float(doc.get("score") or 0.0)
    rerank_score = float(doc.get("rerank_score") or 0.0)
    rrf_rank = int(doc.get("rrf_rank") or position or 1)
    return base_score + rerank_score + (1.0 / (60 + rrf_rank))

def _merge_query_results(result_sets: List[Dict[str, Any]], top_k: int) -> tuple[list[dict], Dict[str, Any]]:
    """
    【核心】多路检索结果融合
    输入：原问题、StepBack、HyDE、同义词 多路召回结果
    输出：去重、融合、排序后的最终文档列表
    """
    fused: Dict[tuple, dict] = {}
    query_hits: Counter = Counter()

    # 遍历每一路检索结果
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

            # 记录来源，用于追踪、监控、评估
            source_info = {
                "variant": variant,
                "doc_id": _doc_identifier(candidate),
                "position": pos,
                "score": candidate["fusion_score"],
            }

            # 保留最高分版本
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

    # 按融合分数倒序，取 top_k
    fused_docs = sorted(
        fused.values(),
        key=lambda item: (
            float(item.get("fusion_score", item.get("score", 0.0)) or 0.0),
            len(item.get("fused_from_variants") or []),
            -int(item.get("fusion_components", {}).get("rrf_rank") or item.get("rrf_rank") or 0),
        ),
        reverse=True,
    )[:top_k]

    # 打上最终排名
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

# ===================== 重排序模型接口 =====================
def _get_rerank_endpoint() -> str:
    """获取重排序接口地址，自动补全路径"""
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"

# ===================== Auto-Merging 核心实现 =====================
def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    """
    单层合并：
    同一个父块下的小块数量 ≥ 阈值 → 替换为父块
    让碎片化内容变回完整段落
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    # 需要合并的父块ID
    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    # 批量查询父块内容
    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue

        # 用父块替换小块，并继承分数
        parent_doc = dict(parent_map[parent_id])
        child_score = float(doc.get("score", 0.0) or 0.0)
        parent_score = float(parent_doc.get("score", 0.0) or 0.0)
        child_fusion_score = float(doc.get("fusion_score", child_score) or child_score)
        parent_fusion_score = float(parent_doc.get("fusion_score", parent_score) or parent_score)

        parent_doc["score"] = max(parent_score, child_score)
        parent_doc["fusion_score"] = max(parent_fusion_score, child_fusion_score)
        parent_doc["rerank_score"] = float(doc.get("rerank_score", parent_doc.get("rerank_score", 0.0)) or 0.0)
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])

        merged_docs.append(parent_doc)
        merged_count += 1

    # 去重
    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count

def _auto_merge_documents(docs: List[dict], top_k: int, auto_merge_enabled: bool) -> Tuple[List[dict], Dict[str, Any]]:
    """
    【高级RAG核心】两级 Auto-Merging
    1. L3 小块 → L2 父块
    2. L2 父块 → L1 根块
    彻底解决切块语义断裂，大幅降低幻觉
    """
    if not auto_merge_enabled or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": auto_merge_enabled,
            "auto_merge_applied": False,
        }

    # 执行两层合并
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    # 重新排序 + 截断
    merged_docs.sort(key=lambda item: item.get("fusion_score", item.get("score", 0.0)), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": auto_merge_enabled,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_replaced_chunks": replaced_count,
    }

# ===================== AI 重排序（精排） =====================
def _rerank_documents(query: str, docs: List[dict], top_k: int, rerank_requested: bool) -> Tuple[List[dict], Dict[str, Any]]:
    """
    调用重排模型对初筛结果进行精排
    让最相关的内容排到最前
    带异常保护、服务降级
    """
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    rerank_available = bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST)

    meta = {
        "rerank_requested": rerank_requested,
        "rerank_available": rerank_available,
        "rerank_enabled": rerank_requested and rerank_available,
    }

    if not docs_with_rank or not rerank_requested or not rerank_available:
        return docs_with_rank[:top_k], meta

    # 调用重排接口
    try:
        response = requests.post(
            _get_rerank_endpoint(),
            headers={"Authorization": f"Bearer {RERANK_API_KEY}"},
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [d.get("text", "") for d in docs_with_rank],
                "top_n": top_k
            },
            timeout=15
        )

        items = response.json().get("results", [])
        reranked = []
        used = set()

        for item in items:
            idx = item.get("index")
            if isinstance(idx, int):
                used.add(idx)
                doc = dict(docs_with_rank[idx])
                doc["rerank_score"] = item.get("relevance_score", 0)
                doc["fusion_score"] = doc["score"] + doc["rerank_score"]
                reranked.append(doc)

        if reranked:
            remaining = [doc for i, doc in enumerate(docs_with_rank) if i not in used]
            merged = reranked + remaining
            merged.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
            return merged[:top_k], {**meta, "rerank_applied": True}

    except Exception as e:
        logger.exception("重排序失败，自动降级")

    return docs_with_rank[:top_k], {**meta, "rerank_fallback_used": True}

# ===================== StepBack + HyDE 查询扩展 =====================
def _get_stepback_model():
    """初始化大模型，用于生成退步查询、HyDE"""
    global _stepback_model
    if not AUGMENT_LLM_API_KEY:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(model=AUGMENT_LLM_MODEL, model_provider="openai", api_key=AUGMENT_LLM_API_KEY, base_url=AUGMENT_LLM_BASE_URL, temperature=0.2)
    return _stepback_model

def _generate_step_back_question(query: str) -> str:
    """生成抽象化“退步问题”，扩大检索覆盖面"""
    model = _get_stepback_model()
    if not model: return ""
    prompt = f"请将问题抽象为高层次退步问题，只输出问题：{query}"
    try:
        return model.invoke(prompt).content.strip()
    except:
        return ""

def generate_hypothetical_document(query: str) -> str:
    """
    HyDE：生成假设性文档
    用模型生成一段模拟答案，提升检索匹配度
    """
    model = _get_stepback_model()
    if not model: return ""
    prompt = f"生成一段与问题相关的假设性参考文档：{query}"
    try:
        return model.invoke(prompt).content.strip()
    except:
        return ""

def step_back_expand(query: str) -> dict:
    """退步查询扩展，返回抽象化的高层次问题"""
    expanded = _generate_step_back_question(query)
    return {"original_query": query, "expanded_query": expanded or query}

# ===================== 主检索入口（最核心） =====================
def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    RAG 检索总入口（生产级完整 Pipeline）
    流程：
    1. 查询理解 → 判断复杂度、选择策略
    2. 智能缓存 → 命中直接返回
    3. 多路查询扩展（原问题 + StepBack + HyDE + 同义词）
    4. 混合检索（密集+稀疏向量）
    5. 结果融合（RRF）
    6. 相似度过滤
    7. AI 重排序
    8. Auto-Merging 自动合并
    9. 返回最终文档
    """

    # ========== 1. 查询理解（智能策略） ==========
    understanding = get_query_understanding_service().analyze_for_retrieval(query)
    query_analysis = understanding.query_analysis
    retrieval_config = understanding.retrieval_config

    # ========== 2. 智能缓存（生产加速） ==========
    # 简单/中等复杂度查询使用缓存
    # ...（省略缓存逻辑，保持你原代码）

    # ========== 3. 多路查询扩展（高级RAG提升召回） ==========
    query_variants = [{"variant": "original", "query": query}]

    # 复杂问题开启：退步查询 + HyDE + 同义词扩展
    if query_analysis.complexity != QueryComplexity.SIMPLE:
        step_back = step_back_expand(query)
        query_variants.append({"variant": "step_back", "query": step_back["expanded_query"]})

        hyde_doc = generate_hypothetical_document(query)
        if hyde_doc:
            query_variants.append({"variant": "hyde", "query": hyde_doc})

    # ========== 4. 执行多路混合检索 ==========
    result_sets = []
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    for var in query_variants:
        try:
            # 生成密集向量 + 稀疏向量
            dense = _embedding_service.get_embeddings([var["query"]])[0]
            sparse = _embedding_service.get_sparse_embedding(var["query"])

            # 混合检索
            res = _milvus_manager.hybrid_retrieve(dense, sparse, top_k=10, filter_expr=filter_expr)
            result_sets.append({"variant": var["variant"], "docs": res})
        except Exception as e:
            logger.warning(f"hybrid_retrieve failed: {e}, falling back to dense_retrieve")
            # 降级：纯密集向量检索
            dense = _embedding_service.get_embeddings([var["query"]])[0]
            res = _milvus_manager.dense_retrieve(dense, top_k=10)
            result_sets.append({"variant": var["variant"], "docs": res})

    # ========== 5. 多路结果 RRF 融合 ==========
    fused_docs, fusion_meta = _merge_query_results(result_sets, top_k=10)

    # ========== 6. 相似度过滤（去掉不相关） ==========
    # filtered = [d for d in fused_docs if d.get("score", 0) >= 0.022]
    filtered = [d for d in fused_docs if d.get("fusion_score", 0) >= 0.01]

    # ========== 7. AI 重排序 ==========
    reranked, rerank_meta = _rerank_documents(query, filtered, top_k, retrieval_config.use_rerank)

    # ========== 8. Auto-Merging 自动合并 ==========
    merged_docs, merge_meta = _auto_merge_documents(reranked, top_k, AUTO_MERGE_ENABLED)

    # ========== 9. 返回最终结果 ==========
    return {
        "docs": merged_docs,
        "meta": {**fusion_meta, **rerank_meta, **merge_meta}
    }


def rewrite_with_context(user_text: str, history: list) -> str:
    """用对话历史将指代词/省略还原为完整问题"""
    if not history:
        return user_text
    model = _get_stepback_model()
    recent = history[-6:]
    history_str = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
        for m in recent
    )
    prompt = (
        f"对话历史：\n{history_str}\n\n"
        f"当前问题：{user_text}\n\n"
        "请将当前问题中的指代词、省略还原为完整独立的问题，只输出还原后的问题，不要解释。"
    )
    try:
        result = model.invoke(prompt)
        return result.content.strip() or user_text
    except Exception:
        return user_text


def build_retrieval_judgement(query: str, docs: list, relevant_ids: list, k: int | None = None, meta: dict | None = None) -> dict:
    retrieved_ids = [_doc_identifier(d) for d in docs]
    k = k or len(retrieved_ids)
    hits = [rid for rid in retrieved_ids[:k] if rid in relevant_ids]
    return {
        "query": query,
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
        "hits": hits,
        "precision": len(hits) / k if k else 0.0,
        "recall": len(hits) / len(relevant_ids) if relevant_ids else 0.0,
        "meta": meta or {},
    }