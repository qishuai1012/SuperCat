# 定义前后端交互的所有数据格式（接口的 “语法手册”）
from pydantic import BaseModel
from typing import Optional, List, Any


# ==============================
# 1. 用户认证相关（登录/注册）
# ==============================

# 注册请求体
class RegisterRequest(BaseModel):
    username: str                # 用户名
    password: str                # 密码
    role: Optional[str] = "user" # 角色：默认普通用户
    admin_code: Optional[str] = None # 管理员邀请码（可选）

# 登录请求体
class LoginRequest(BaseModel):
    username: str
    password: str

# 登录/注册成功返回
class AuthResponse(BaseModel):
    access_token: str       # JWT 身份令牌
    token_type: str = "bearer"
    username: str           # 用户名
    role: str               # 角色

# 获取当前登录用户信息
class CurrentUserResponse(BaseModel):
    username: str
    role: str


# ==============================
# 2. 聊天接口数据格式
# ==============================

# 前端 → 后端：发消息
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"

# 检索到的文档块信息
class RetrievedChunk(BaseModel):
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None


# ------------------------------
# 以下是 RAG 内部追踪格式（高级调试用）
# ------------------------------
# ============================
# 以下所有类 👉 仅供 AI 内部使用
# 不属于接口，不给前端用
# 作用：记录 AI 的思考、推理、上下文、任务
# ============================

# 知识点结构：AI 从文档里提取的核心知识点（结构化记忆）
class KnowledgePoint(BaseModel):
    concept: str                          # 知识点名称（如：向量数据库）
    definition: str                       # 知识点定义/解释
    importance: Optional[int] = None      # 重要程度（1-5）
    category: Optional[str] = None        # 分类（如：RAG、数据库、编程）

# 消息摘要：用于压缩聊天历史，只保留简短关键信息
class MessageDigest(BaseModel):
    type: str                             # 消息类型：user / assistant
    content_preview: str                  # 消息内容摘要（精简版）

# 上下文压缩元数据：记录 AI 如何压缩长对话
class CompressionMetadata(BaseModel):
    strategy: str                         # 压缩策略名称
    raw_chars: int = 0                    # 原始字符数
    compact_chars: int = 0                # 压缩后字符数
    compression_ratio: float = 1.0        # 压缩比例
    dropped_fields: List[str] = []        # 压缩时丢弃了哪些字段
    history_messages_total: Optional[int] = None    # 总历史消息数
    history_messages_retained: Optional[int] = None # 保留的消息数
    trace_chunks_total: Optional[int] = None        # 总文档块数
    trace_chunks_retained: Optional[int] = None     # 保留的文档块数

# 精简追踪信息：AI 执行过程的轻量记录
class CompactTrace(BaseModel):
    tool_used: Optional[bool] = None              # 是否使用了 RAG 工具
    tool_name: Optional[str] = None                # 工具名称
    query: Optional[str] = None                    # 用户的问题
    retrieval_stage: Optional[str] = None          # 检索阶段
    evidence_chunks: Optional[List[RetrievedChunk]] = None  # 参考的文档片段
    reason: Optional[str] = None                   # 选择这些文档的原因
    filter_summary: Optional[dict] = None          # 过滤规则的总结

# 上下文包：给大模型提供的所有上下文（问题+历史+知识点）
class ContextBundle(BaseModel):
    query: str                                     # 用户当前问题
    user_id: Optional[str] = None                  # 用户ID
    session_id: Optional[str] = None               # 会话ID
    history_summary: str = ""                      # 历史对话总结
    recent_messages: List[MessageDigest] = []      # 最近的精简消息
    key_knowledge: List[KnowledgePoint] = []       # 提取的关键知识点
    compression_meta: Optional[CompressionMetadata] = None  # 压缩信息

# 子任务结果总结：多智能体系统中，每个子任务的执行结果
class SubtaskResultSummary(BaseModel):
    task_id: str                                   # 子任务ID
    agent_type: str                                # 执行任务的智能体类型
    success: bool                                  # 是否成功
    summary: str                                   # 任务结果总结
    key_points: List[KnowledgePoint] = []          # 任务提取的知识点
    evidence: List[RetrievedChunk] = []            # 任务使用的证据文档
    issues: List[str] = []                         # 遇到的问题
    confidence: Optional[float] = None             # 置信度
    compact_trace: Optional[CompactTrace] = None   # 执行过程追踪
    execution_time: Optional[float] = None         # 执行耗时

# 规划总结：AI 回答问题前的“作战计划”
class PlanningSummary(BaseModel):
    goal: str                                      # 最终目标
    strategy: Optional[str] = None                 # 整体策略
    subtasks: List[dict] = []                      # 拆分成的子任务列表
    risks: List[str] = []                          # 识别到的风险
    checks: List[str] = []                         # 校验步骤

# 验证总结：AI 回答后的自我校验（防幻觉、纠错）
class VerificationSummary(BaseModel):
    verdict: str                                   # 最终判定：正确/错误/需修改
    issues_found: List[str] = []                   # 发现的问题
    supported_claims: List[str] = []               # 有文档支持的内容
    unsupported_claims: List[str] = []             # 无依据的内容（幻觉）
    recommended_changes: List[str] = []            # 建议修改的内容


# RAG 完整追踪日志（超级详细）
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


# 后端 → 前端：聊天回复
class ChatResponse(BaseModel):
    response: str               # AI 回答内容
    rag_trace: Optional[RagTrace] = None # RAG 追踪信息（可选）


# ==============================
# 3. 会话（聊天历史）格式
# ==============================

# 单条消息
class MessageInfo(BaseModel):
    type: str          # user / assistant
    content: str       # 内容
    timestamp: str     # 时间
    rag_trace: Optional[RagTrace] = None

# 会话的全部消息
class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]

# 单个会话信息
class SessionInfo(BaseModel):
    session_id: str
    updated_at: str
    message_count: int

# 会话列表
class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]

# 删除会话返回
class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


# ==============================
# 4. 知识库文档管理格式
# ==============================

# 文档信息
class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    file_md5: Optional[str] = None
    uploaded_at: Optional[str] = None

# 文档列表
class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

# 上传文档返回
class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str
    file_md5: Optional[str] = None

# 删除文档返回
class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str