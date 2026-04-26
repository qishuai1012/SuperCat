from pydantic import BaseModel
from typing import Optional, List

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
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    candidate_k: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
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
    uploaded_at: Optional[str] = None

#￥返回知识库所有文件列表
class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

#文件上传成功返回
class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str

#删除文件成功返回
class DocumentDeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str
