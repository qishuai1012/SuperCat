from typing import Literal, TypedDict, List, Optional
import json
import os
import re
import time

import logging
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from performance_config import get_performance_config
from query_understanding.complexity import get_complexity_analyzer
from rag.expansion import generate_hypothetical_document, step_back_expand
from rag.retriever import retrieve_documents
from rag_utils import build_retrieval_judgement, _merge_query_results
from rag.trace import _format_docs, build_retrieval_trace, merge_rag_trace
from tools import emit_rag_step, should_skip_grading

logger = logging.getLogger(__name__)
load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL", MODEL)  # 使用主模型作为默认值

_grader_model = None
_router_model = None


def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )
    return _grader_model


def _get_router_model():
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )
    return _router_model


GRADE_PROMPT = (
    "You are an expert grader. Your task is to assess whether a retrieved document is relevant to a user question.\n\n"
    "Retrieved document:\n{context}\n\n"
    "User question:\n{question}\n\n"
    "Your job: Output ONLY a valid JSON object with a single field 'binary_score'.\n\n"
    "Rules:\n"
    "- If the document contains keywords or semantic meaning related to the user question, set binary_score to 'yes'.\n"
    "- If the document is not relevant to the user question, set binary_score to 'no'.\n"
    "- Output ONLY the JSON object. Do NOT add any explanations, comments, or additional text.\n\n"
    "Example valid output:\n"
    "{{\"binary_score\": \"yes\"}}"
    "or\n"
    "{{\"binary_score\": \"no\"}}"
    "\n\n"
    "Your output:")


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """Choose a query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    skip_grading: Optional[bool]
    rag_trace: Optional[dict]


def _get_strategy_config_for_question(question: str):
    complexity = get_complexity_analyzer().analyze(question)
    return get_performance_config().get_strategy(complexity), complexity


def _decide_grading(question: str, docs: List[dict], skip_grading_signal: bool, retrieval_trace: dict | None = None) -> dict:
    retrieval_trace = retrieval_trace or {}
    strategy_config, complexity = _get_strategy_config_for_question(question)
    docs_count = len(docs)

    decision = {
        "grade_config_enabled": bool(strategy_config.enable_document_grading),
        "grade_docs_count": docs_count,
        "grade_complexity": complexity.value,
        "grade_skip_heuristic": bool(skip_grading_signal),
        "grade_skip_reason": None,
        "grade_should_run": True,
        "grade_doc_quality_threshold": strategy_config.doc_quality_threshold,
        "grade_min_docs_for_skip": strategy_config.min_docs_for_skip_grading,
    }

    if docs_count == 0:
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "no_docs"
        return decision

    if not strategy_config.enable_document_grading:
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "config_disabled"
        return decision

    if retrieval_trace.get("no_relevant_docs"):
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "retrieval_reported_no_relevant_docs"
        return decision

    if skip_grading_signal and docs_count >= strategy_config.min_docs_for_skip_grading:
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "heuristic_skip_with_enough_docs"
        return decision

    return decision


def _parse_grade_response(raw_content: str) -> tuple[str, str]:
    text = (raw_content or "").strip()
    if not text:
        raise ValueError("empty grader response")

    try:
        payload = json.loads(text)
        score = str(payload.get("binary_score", "")).strip().lower()
        if score in ("yes", "no"):
            return score, "json"
    except Exception:
        pass

    json_match = re.search(r"\{[\s\S]*?\}", text)
    if json_match:
        payload = json.loads(json_match.group())
        score = str(payload.get("binary_score", "")).strip().lower()
        if score in ("yes", "no"):
            return score, "json_extract"

    normalized = text.lower()
    if normalized in ("yes", '"yes"', "'yes'"):
        return "yes", "literal"
    if normalized in ("no", '"no"', "'no'"):
        return "no", "literal"

    raise ValueError(f"unrecognized grader response: {text[:80]}")


def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]
    logger.info(f"🔵 retrieve_initial节点开始执行")
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
    started_at = time.perf_counter()
    retrieved = retrieve_documents(query, top_k=5)
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(f"🔵 retrieve_initial节点执行完成")
    results = retrieved.get("docs", [])
    retrieve_meta = dict(retrieved.get("meta", {}) or {})
    retrieve_meta["retrieval_duration_ms"] = duration_ms
    retrieve_meta["model_versions"] = {"grade_model": GRADE_MODEL, "rewrite_router_model": MODEL, "generation_model": MODEL}
    context = _format_docs(results)
    emit_rag_step(
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    emit_rag_step("✅", f"检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    rag_trace = build_retrieval_trace(
        query=query,
        docs=results,
        retrieval_meta=retrieve_meta,
        retrieval_stage="initial",
        expanded_query=query,
        trace=state.get("rag_trace"),
    )
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    started_at = time.perf_counter()
    docs = state.get("docs") or []
    question = state["question"]
    skip_grading_signal = bool(state.get("skip_grading") or should_skip_grading(question))
    rag_trace = state.get("rag_trace", {}) or {}
    decision = _decide_grading(
        question=question,
        docs=docs,
        skip_grading_signal=skip_grading_signal,
        retrieval_trace=rag_trace,
    )
    emit_rag_step("📊", "正在评估文档相关性...")

    base_trace = {
        "grade_config_enabled": decision["grade_config_enabled"],
        "grade_docs_count": decision["grade_docs_count"],
        "grade_complexity": decision["grade_complexity"],
        "grade_skip_heuristic": decision["grade_skip_heuristic"],
        "grade_skip_reason": decision["grade_skip_reason"],
        "grade_doc_quality_threshold": decision["grade_doc_quality_threshold"],
        "grade_min_docs_for_skip": decision["grade_min_docs_for_skip"],
        "grading_duration_ms": None,
    }

    if not docs:
        logger.info("No documents retrieved; returning KB no-result route without grading or rewrite")
        rag_trace = merge_rag_trace(
            rag_trace,
            **{**base_trace, "grading_duration_ms": round((time.perf_counter() - started_at) * 1000, 2)},
            grade_score="no_docs",
            grade_route="generate_answer",
            rewrite_needed=False,
            grade_skipped=True,
            grade_parser_mode="not_run",
            kb_no_result=True,
        )
        emit_rag_step("⚠️", "未检索到相关文档", "直接返回知识库无结果")
        return {"route": "generate_answer", "rag_trace": rag_trace}

    if not decision["grade_should_run"]:
        logger.info(f"Skipping document grading: {decision['grade_skip_reason']}")
        rag_trace = merge_rag_trace(
            rag_trace,
            **{**base_trace, "grading_duration_ms": round((time.perf_counter() - started_at) * 1000, 2)},
            grade_score="skipped",
            grade_route="generate_answer",
            rewrite_needed=False,
            grade_skipped=True,
            grade_parser_mode="not_run",
            kb_no_result=bool(rag_trace.get("kb_no_result", False)),
        )
        emit_rag_step("⚡", "跳过文档评分", f"原因: {decision['grade_skip_reason']}")
        return {"route": "generate_answer", "rag_trace": rag_trace}

    grader = _get_grader_model()
    logger.info(
        f"Grader model: {GRADE_MODEL}, API key present: {bool(API_KEY)}, "
        f"skip_signal={skip_grading_signal}, complexity={decision['grade_complexity']}"
    )

    if not grader:
        rag_trace = merge_rag_trace(
            rag_trace,
            **{
                **base_trace,
                "grade_skip_reason": "grader_unavailable",
                "grading_duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
            grade_score="unknown",
            grade_route="rewrite_question",
            rewrite_needed=True,
            grade_skipped=False,
            grade_parser_mode="not_run",
            kb_no_result=False,
        )
        return {"route": "rewrite_question", "rag_trace": rag_trace}

    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    score = "no"
    parser_mode = "fallback_no"

    try:
        response = grader.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        if response and hasattr(response, "binary_score"):
            raw_score = str(response.binary_score).strip().lower()
            if raw_score == "binary_score":
                raise ValueError("模型返回了字段名作为值")
            if raw_score in ("yes", "no"):
                score = raw_score
                parser_mode = "structured"
            else:
                raise ValueError(f"unexpected structured score: {raw_score}")
    except Exception:
        try:
            raw_response = grader.invoke([{"role": "user", "content": prompt}])
            content = str(raw_response.content if hasattr(raw_response, "content") else raw_response)
            score, parser_mode = _parse_grade_response(content)
        except Exception:
            score = "no"
            parser_mode = "fallback_no"

    route = "generate_answer" if score == "yes" else "rewrite_question"
    if route == "generate_answer":
        emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score}")

    rag_trace = merge_rag_trace(
        rag_trace,
        **{
            **base_trace,
            "grade_skip_reason": None,
            "grading_duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
        grade_score=score,
        grade_route=route,
        rewrite_needed=route == "rewrite_question",
        grade_skipped=False,
        grade_parser_mode=parser_mode,
        kb_no_result=False,
    )
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    started_at = time.perf_counter()
    question = state["question"]
    emit_rag_step("✏️", "正在重写查询...")

    strategy = state.get("expansion_type")
    router = _get_router_model()
    rag_trace = state.get("rag_trace", {}) or {}
    source = None

    if strategy:
        if rag_trace.get("route_expansion_hint"):
            source = "expansion_hint"
        elif rag_trace.get("legacy_route_strategy"):
            source = "legacy_route"

    if not strategy and router:
        prompt = (
            "请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
            f"用户问题：{question}"
        )
        try:
            decision = router.with_structured_output(RewriteStrategy).invoke(
                [{"role": "user", "content": prompt}]
            )
            strategy = decision.strategy
            source = "rag_router"
        except Exception:
            strategy = "step_back"
            source = "default"

    if not strategy:
        strategy = "step_back"
        source = "default"

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE 假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)

    rag_trace = merge_rag_trace(
        rag_trace,
        rewrite_strategy=strategy,
        rewrite_duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        rewrite_query=expanded_query,
        query_expansion_owner="rag_pipeline",
        hyde_generated_count=1 if hypothetical_doc else 0,
        step_back_generated=bool(step_back_question or step_back_answer),
        rewrite_strategy_source=source or "default",
        kb_no_result=False,
    )

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    started_at = time.perf_counter()
    strategy = state.get("expansion_type") or "step_back"
    emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")

    result_sets: List[Dict[str, Any]] = []
    rerank_requested_any = False
    rerank_available_any = False
    rerank_applied_any = False
    rerank_enabled_any = False
    rerank_model = None
    rerank_endpoint = None
    rerank_errors = []
    rerank_skip_reasons = []
    retrieval_mode = None
    candidate_k = None
    leaf_retrieve_level = None
    auto_merge_enabled = None
    auto_merge_applied = False
    auto_merge_threshold = None
    auto_merge_replaced_chunks = 0
    auto_merge_steps = 0

    if strategy in ("hyde", "complex"):
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=5)
        result_sets.append({"variant": "hyde", "docs": retrieved_hyde.get("docs", [])})
        hyde_meta = retrieved_hyde.get("meta", {})
        emit_rag_step(
            "🧱",
            "HyDE 三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_requested_any = rerank_requested_any or bool(hyde_meta.get("rerank_requested"))
        rerank_available_any = rerank_available_any or bool(hyde_meta.get("rerank_available"))
        rerank_applied_any = rerank_applied_any or bool(hyde_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(hyde_meta.get("rerank_enabled"))
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        if hyde_meta.get("rerank_skip_reason"):
            rerank_skip_reasons.append(f"hyde:{hyde_meta.get('rerank_skip_reason')}")
        retrieval_mode = retrieval_mode or hyde_meta.get("retrieval_mode")
        candidate_k = candidate_k or hyde_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or hyde_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else hyde_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(hyde_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or hyde_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(hyde_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(hyde_meta.get("auto_merge_steps") or 0)

    if strategy in ("step_back", "complex", "direct"):
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = retrieve_documents(expanded_query, top_k=5)
        result_sets.append({"variant": "step_back" if strategy != "direct" else "direct", "docs": retrieved_stepback.get("docs", [])})
        step_meta = retrieved_stepback.get("meta", {})
        emit_rag_step(
            "🧱",
            "Step-back 三级检索" if strategy != "direct" else "Direct 三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_requested_any = rerank_requested_any or bool(step_meta.get("rerank_requested"))
        rerank_available_any = rerank_available_any or bool(step_meta.get("rerank_available"))
        rerank_applied_any = rerank_applied_any or bool(step_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(step_meta.get("rerank_enabled"))
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        if step_meta.get("rerank_skip_reason"):
            rerank_skip_reasons.append(f"step_back:{step_meta.get('rerank_skip_reason')}")
        retrieval_mode = retrieval_mode or step_meta.get("retrieval_mode")
        candidate_k = candidate_k or step_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or step_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else step_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(step_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or step_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(step_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(step_meta.get("auto_merge_steps") or 0)

    fused_docs, fusion_meta = _merge_query_results(result_sets, top_k=max(5, len(result_sets) * 5)) if result_sets else ([], {"multi_query_enabled": False, "multi_query_variants": [], "multi_query_docs_total": 0, "multi_query_docs_unique": 0, "multi_query_docs_returned": 0, "multi_query_hits": {}})

    context = _format_docs(fused_docs)
    emit_rag_step("✅", f"扩展检索完成，共 {len(fused_docs)} 个片段")
    retrieval_meta = {
        "query_expansion_owner": "rag_pipeline",
        "retrieval_duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "model_versions": {"grade_model": GRADE_MODEL, "rewrite_router_model": MODEL, "generation_model": MODEL},
        "hyde_generated_count": 1 if state.get("hypothetical_doc") else 0,
        "step_back_generated": bool(state.get("step_back_question") or state.get("step_back_answer")),
        "expanded_retrieval_count": (1 if strategy in ("hyde", "complex") else 0) + (1 if strategy in ("step_back", "complex", "direct") else 0),
        "rerank_requested": rerank_requested_any,
        "rerank_available": rerank_available_any,
        "rerank_enabled": rerank_enabled_any,
        "rerank_applied": rerank_applied_any,
        "rerank_skip_reason": None if rerank_applied_any else "; ".join(rerank_skip_reasons) if rerank_skip_reasons else None,
        "rerank_model": rerank_model,
        "rerank_endpoint": rerank_endpoint,
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "leaf_retrieve_level": leaf_retrieve_level,
        "auto_merge_enabled": auto_merge_enabled,
        "auto_merge_applied": auto_merge_applied,
        "auto_merge_threshold": auto_merge_threshold,
        "auto_merge_replaced_chunks": auto_merge_replaced_chunks,
        "auto_merge_steps": auto_merge_steps,
        "kb_no_result": not fused_docs,
        "no_relevant_docs": not fused_docs,
        **fusion_meta,
    }
    rag_trace = build_retrieval_trace(
        query=state["question"],
        docs=fused_docs,
        retrieval_meta=retrieval_meta,
        retrieval_stage="expanded",
        expanded_query=state.get("expanded_query") or state["question"],
        trace=state.get("rag_trace"),
    )
    rag_trace = merge_rag_trace(
        rag_trace,
        step_back_question=state.get("step_back_question", ""),
        step_back_answer=state.get("step_back_answer", ""),
        hypothetical_doc=state.get("hypothetical_doc", ""),
        expansion_type=strategy,
    )
    return {"docs": fused_docs, "context": context, "rag_trace": rag_trace}


def build_rag_graph():
    # 创建一个状态图，数据格式遵循 RAGState
    graph = StateGraph(RAGState)

    # 注册 4 个核心节点（就是你刚才看懂的那些函数）
    graph.add_node("retrieve_initial", retrieve_initial)    # 1. 初始检索
    graph.add_node("grade_documents", grade_documents_node)# 2. 评估文档
    graph.add_node("rewrite_question", rewrite_question_node)# 3. 重写问题
    graph.add_node("retrieve_expanded", retrieve_expanded)  # 4. 扩展检索（你刚吃透的！）

    # 入口：从第一次检索开始
    graph.set_entry_point("retrieve_initial")

    # 第一步执行完 → 进入评估
    graph.add_edge("retrieve_initial", "grade_documents")

    # --------------------------
    # 最关键：条件分支
    # --------------------------
    graph.add_conditional_edges(
        "grade_documents",          # 从评估节点出发
        lambda state: state["route"], # 看路由决定下一步
        {
            "generate_answer": END,      # 文档好 → 直接结束（后面生成答案）
            "rewrite_question": "rewrite_question", # 文档差 → 重写问题
        },
    )

    # 重写问题 → 执行扩展检索（你吃透的那个函数）
    graph.add_edge("rewrite_question", "retrieve_expanded")

    # 扩展检索执行完 → 结束流程
    graph.add_edge("retrieve_expanded", END)

    return graph.compile()

# 最终生成一个可调用的 RAG 图
rag_graph = build_rag_graph()


def run_rag_graph(question: str, skip_grading: bool = False, expansion_hint: str | None = None) -> dict:
    legacy_strategy = None
    if question.startswith("[ROUTE:"):
        end = question.find("]")
        if end > 0:
            legacy_strategy = question[7:end].strip()
            question = question[end+1:].strip()

    strategy = expansion_hint or legacy_strategy

    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": strategy,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "skip_grading": skip_grading,
        "rag_trace": {
            "route_expansion_hint": expansion_hint,
            "legacy_route_strategy": legacy_strategy,
        },
    })


def evaluate_rag_retrieval(question: str, relevant_ids: List[str], *, k: int | None = None, skip_grading: bool = False, expansion_hint: str | None = None) -> dict:
    from learning_system import get_online_learning_system

    rag_result = run_rag_graph(question, skip_grading=skip_grading, expansion_hint=expansion_hint)
    docs = rag_result.get("docs") or []
    rag_trace = rag_result.get("rag_trace") or {}
    judged_result = build_retrieval_judgement(
        query=question,
        docs=docs,
        relevant_ids=relevant_ids,
        k=k,
        meta=rag_trace if isinstance(rag_trace, dict) else {},
    )
    metrics = get_online_learning_system().evaluate_retrieval_metrics([judged_result])
    return {
        "question": question,
        "judged_result": judged_result,
        "metrics": metrics,
        "rag_trace": rag_trace,
        "docs": docs,
    }