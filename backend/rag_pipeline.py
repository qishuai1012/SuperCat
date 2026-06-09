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

# 配置：性能策略、问题复杂度分析
from performance_config import get_performance_config
from query_understanding.complexity import get_complexity_analyzer
# RAG 模块：检索、查询扩展
from rag.expansion import generate_hypothetical_document, step_back_expand
from rag.retriever import retrieve_documents
# 工具函数：RRF融合、结果评估、文档格式化、追踪日志
from rag_utils import build_retrieval_judgement, _merge_query_results
from rag.trace import _format_docs, build_retrieval_trace, merge_rag_trace
# 前端/日志输出工具
from tools import emit_rag_step, should_skip_grading

# 日志初始化
logger = logging.getLogger(__name__)
load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL", MODEL)  # 使用主模型作为默认值，# 评分模型（默认=主模型）

# 全局单例模型（避免重复加载，生产级标准）
_grader_model = None
_router_model = None

# ----------------------
# 1. 加载 文档评分模型
# 作用：判断检索回来的文档是否与问题相关
# ----------------------
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
            temperature=0,       # 评分必须确定性
        )
    return _grader_model

# ----------------------
# 2. 加载 路由模型
# 作用：自动选择策略：step_back / hyde / complex
# ----------------------
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

# ====================== 评分 Prompt ======================
# 作用：让模型判断文档是否相关，输出 yes / no
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

# ====================== 结构化输出定义 ======================
class GradeDocuments(BaseModel):
    """文档相关性评分：二元判断"""
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """查询重写策略"""
    """Choose a query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]

# ====================== RAG 状态管理 ======================
# 整个工作流共享的状态（LangGraph 必须）
class RAGState(TypedDict):
    question: str                      # 用户原始问题
    query: str                        # 当前查询语句
    context: str                      # 文档拼接后的内容
    docs: List[dict]                  # 检索到的文档
    route: Optional[str]              # 路由：generate_answer / rewrite_question
    expansion_type: Optional[str]     # 扩展类型：step_back/hyde/complex
    expanded_query: Optional[str]     # 扩展后的查询
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]   # HyDE 生成的假设文档
    skip_grading: Optional[bool]      # 是否跳过评分
    rag_trace: Optional[dict]         # 完整追踪日志

# ====================== 策略决策（根据问题复杂度） ======================
def _get_strategy_config_for_question(question: str):
    """
    根据问题复杂度（简单/中等/复杂）自动选择检索策略
    生产级：动态调整，简单问题不浪费性能
    """
    complexity = get_complexity_analyzer().analyze(question)
    return get_performance_config().get_strategy(complexity), complexity

# ====================== 评分决策 ======================
def _decide_grading(question: str, docs: List[dict], skip_grading_signal: bool, retrieval_trace: dict | None = None) -> dict:
    """
    生产级核心逻辑：
    判断是否需要运行文档评分，避免不必要调用模型
    """
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

    # 无文档 → 不评分
    if docs_count == 0:
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "no_docs"
        return decision

     # 配置关闭评分 → 不评分
    if not strategy_config.enable_document_grading:
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "config_disabled"
        return decision

    # 检索器返回无相关文档 → 不评分
    if retrieval_trace.get("no_relevant_docs"):
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "retrieval_reported_no_relevant_docs"
        return decision

    # 启发式跳过：问题简单+文档足够 → 不评分
    if skip_grading_signal and docs_count >= strategy_config.min_docs_for_skip_grading:
        decision["grade_should_run"] = False
        decision["grade_skip_reason"] = "heuristic_skip_with_enough_docs"
        return decision

    return decision

# ====================== 解析评分模型输出 ======================
def _parse_grade_response(raw_content: str) -> tuple[str, str]:
    """
    解析模型返回的 JSON / 纯文本，兼容各种输出格式
    生产级：鲁棒性极强
    """
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
    
    # 2. 正则提取 JSON
    json_match = re.search(r"\{[\s\S]*?\}", text)
    if json_match:
        payload = json.loads(json_match.group())
        score = str(payload.get("binary_score", "")).strip().lower()
        if score in ("yes", "no"):
            return score, "json_extract"

     # 3. 纯文本匹配
    normalized = text.lower()
    if normalized in ("yes", '"yes"', "'yes'"):
        return "yes", "literal"
    if normalized in ("no", '"no"', "'no'"):
        return "no", "literal"

    raise ValueError(f"unrecognized grader response: {text[:80]}")

# ====================== 节点 1：初次检索 ======================
def retrieve_initial(state: RAGState) -> RAGState:
    """
    第一次检索：直接用用户原始问题检索
    """
    query = state["question"]
    logger.info(f"🔵 retrieve_initial节点开始执行")
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
    started_at = time.perf_counter()

    # 调用检索工具
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

# ====================== 节点 2：文档评分（Self-RAG 核心） ======================
def grade_documents_node(state: RAGState) -> RAGState:
    """
    让模型判断：检索到的文档是否足够回答问题
    yes → 生成答案
    no → 重写查询 + 多路检索
    """
    started_at = time.perf_counter()
    docs = state.get("docs") or []
    question = state["question"]
    skip_grading_signal = bool(state.get("skip_grading") or should_skip_grading(question))
    rag_trace = state.get("rag_trace", {}) or {}

    # 判断是否需要评分
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

    # 无文档 → 直接结束
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

    # 配置跳过评分 → 直接生成答案
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

    # 获取评分模型
    grader = _get_grader_model()
    logger.info(
        f"Grader model: {GRADE_MODEL}, API key present: {bool(API_KEY)}, "
        f"skip_signal={skip_grading_signal}, complexity={decision['grade_complexity']}"
    )

     # 模型不可用 → 重写查询
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
    
    # 执行评分
    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    score = "no"
    parser_mode = "fallback_no"

    # 结构化输出优先
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
        # 兼容普通文本输出
        try:
            raw_response = grader.invoke([{"role": "user", "content": prompt}])
            content = str(raw_response.content if hasattr(raw_response, "content") else raw_response)
            score, parser_mode = _parse_grade_response(content)
        except Exception:
            score = "no"
            parser_mode = "fallback_no"

     # 决策路由
    route = "generate_answer" if score == "yes" else "rewrite_question"
    if route == "generate_answer":
        emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score}")
    
    # 记录追踪日志
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

# ====================== 节点 3：查询重写（智能路由） ======================
def rewrite_question_node(state: RAGState) -> RAGState:
    """
    智能选择查询扩展策略：
    step_back / hyde / complex
    复杂问题自动两路都执行
    """
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

    # 路由模型选择最优策略
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

    # step_back 策略：抽象问题，扩大召回
    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    # hyde 策略：生成假设答案再检索   
    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE 假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)
    
    # 记录日志
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

# ====================== 节点 4：扩展检索（多路检索 + RRF 融合） ======================
def retrieve_expanded(state: RAGState) -> RAGState:
    """
    多路检索核心：
    1. hyde 检索一路
    2. step_back 检索一路
    3. RRF 融合排序
    4. 返回最终文档
    """
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

    # ----------------------
    # 多路 1：HyDE 检索
    # ----------------------
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

    # ----------------------
    # 多路 2：StepBack 检索
    # ----------------------
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

    # ----------------------
    # RRF 多路结果融合
    # ----------------------
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

# ====================== 构建 RAG 工作流（LangGraph） ======================
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

#执行完整的 RAG 工作流，返回最终文档
def run_rag_graph(question: str, skip_grading: bool = False, expansion_hint: str | None = None) -> dict:
    legacy_strategy = None
    if question.startswith("[ROUTE:"):
        end = question.find("]")
        if end > 0:
            legacy_strategy = question[7:end].strip()
            question = question[end+1:].strip()

    strategy = expansion_hint or legacy_strategy

    #调用 LangGraph 执行完整 RAG 流程
    return rag_graph.invoke({
        "question": question,            # 用户问题
        "query": question,               # 当前查询
        "context": "",                   # 文档内容（一开始为空）
        "docs": [],                     # 检索到的文档（一开始为空）
        "route": None,                  # 路由（一开始为空）
        "expansion_type": strategy,     # 策略
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,   
        "hypothetical_doc": None,
        "skip_grading": skip_grading,   # 是否跳过评分
        "rag_trace": {                  # 追踪日志
            "route_expansion_hint": expansion_hint,
            "legacy_route_strategy": legacy_strategy,
        },
    })

#测试 RAG 效果好不好，打分、评估准确率
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