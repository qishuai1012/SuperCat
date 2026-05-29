from pydantic import BaseModel
from typing import Optional, List, Any

#用户登录 / 注册相关
class RegisterRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"
    admin_code: Optional[str] = None

#登录接口接收的数据
class LoginRequest(BaseModel):
    username: str
    password: str

#登录成功返回的数据
class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str

#获取当前登录用户信息
class CurrentUserResponse(BaseModel):
    username: str
    role: str

# 前端发消息给后端的数据格式
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"

#检索到的文档片段格式
class RetrievedChunk(BaseModel):
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class KnowledgePoint(BaseModel):
    concept: str
    definition: str
    importance: Optional[int] = None
    category: Optional[str] = None


class MessageDigest(BaseModel):
    type: str
    content_preview: str


class CompressionMetadata(BaseModel):
    strategy: str
    raw_chars: int = 0
    compact_chars: int = 0
    compression_ratio: float = 1.0
    dropped_fields: List[str] = []
    history_messages_total: Optional[int] = None
    history_messages_retained: Optional[int] = None
    trace_chunks_total: Optional[int] = None
    trace_chunks_retained: Optional[int] = None


class CompactTrace(BaseModel):
    tool_used: Optional[bool] = None
    tool_name: Optional[str] = None
    query: Optional[str] = None
    retrieval_stage: Optional[str] = None
    evidence_chunks: Optional[List[RetrievedChunk]] = None
    reason: Optional[str] = None
    filter_summary: Optional[dict] = None


class ContextBundle(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    history_summary: str = ""
    recent_messages: List[MessageDigest] = []
    key_knowledge: List[KnowledgePoint] = []
    compression_meta: Optional[CompressionMetadata] = None


class SubtaskResultSummary(BaseModel):
    task_id: str
    agent_type: str
    success: bool
    summary: str
    key_points: List[KnowledgePoint] = []
    evidence: List[RetrievedChunk] = []
    issues: List[str] = []
    confidence: Optional[float] = None
    compact_trace: Optional[CompactTrace] = None
    execution_time: Optional[float] = None


class PlanningSummary(BaseModel):
    goal: str
    strategy: Optional[str] = None
    subtasks: List[dict] = []
    risks: List[str] = []
    checks: List[str] = []


class VerificationSummary(BaseModel):
    verdict: str
    issues_found: List[str] = []
    supported_claims: List[str] = []
    unsupported_claims: List[str] = []
    recommended_changes: List[str] = []

#这就是 RAG 执行全过程的记录！
class RagTrace(BaseModel):
    tool_used: bool
    tool_name: str
    query: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rewrite_strategy_source: Optional[str] = None
    rerank_requested: Optional[bool] = None
    rerank_available: Optional[bool] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_skip_reason: Optional[str] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    candidate_k: Optional[int] = None
    candidate_count: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    query_expansion_owner: Optional[str] = None
    hyde_generated_count: Optional[int] = None
    step_back_generated: Optional[bool] = None
    expanded_retrieval_count: Optional[int] = None
    kb_no_result: Optional[bool] = None
    no_relevant_docs: Optional[bool] = None
    reason: Optional[str] = None
    score_type: Optional[str] = None
    filter_applied: Optional[bool] = None
    filter_threshold: Optional[float] = None
    filter_type: Optional[str] = None
    filter_score_types: Optional[List[str]] = None
    original_count: Optional[int] = None
    filtered_count: Optional[int] = None
    retained_count: Optional[int] = None
    dynamic_strategy: Optional[dict] = None
    route_expansion_hint: Optional[str] = None
    legacy_route_strategy: Optional[str] = None
    grade_config_enabled: Optional[bool] = None
    grade_docs_count: Optional[int] = None
    grade_complexity: Optional[str] = None
    grade_skip_heuristic: Optional[bool] = None
    grade_skip_reason: Optional[str] = None
    grade_should_run: Optional[bool] = None
    grade_doc_quality_threshold: Optional[float] = None
    grade_min_docs_for_skip: Optional[int] = None
    grade_skipped: Optional[bool] = None
    grade_parser_mode: Optional[str] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_avg: Optional[float] = None
    multi_query_enabled: Optional[bool] = None
    multi_query_variants: Optional[List[str]] = None
    multi_query_docs_total: Optional[int] = None
    multi_query_docs_unique: Optional[int] = None
    multi_query_docs_returned: Optional[int] = None
    multi_query_hits: Optional[dict] = None
    stage_durations_ms: Optional[dict] = None
    retrieval_duration_ms: Optional[float] = None
    rewrite_duration_ms: Optional[float] = None
    grading_duration_ms: Optional[float] = None
    rerank_duration_ms: Optional[float] = None
    model_versions: Optional[dict] = None
    hit_reason: Optional[str] = None
    diagnostic_summary: Optional[str] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None

#后端返回给前端的数据
class ChatResponse(BaseModel):
    response: str
    rag_trace: Optional[RagTrace] = None

#单条消息的格式
class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None

#获取某个会话的所有历史消息
class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]

#单个会话信息
class SessionInfo(BaseModel):
    session_id: str
    updated_at: str
    message_count: int

#获取用户所有会话列表
class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]

#删除会话后返回
class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str

#知识库文件信息
class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    file_md5: Optional[str] = None
    uploaded_at: Optional[str] = None

#￥返回知识库所有文件列表
class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

#文件上传成功返回
class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str
    file_md5: Optional[str] = None

#删除文件成功返回
class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str
