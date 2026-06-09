from fastapi import APIRouter, Depends, HTTPException

# agent.storage：负责会话数据的存储（内存/文件/数据库）
from agent import storage
# 身份认证：获取当前登录用户
from auth import get_current_user
# 用户数据模型
from models import User
# 接口返回格式（Pydantic 规范）
from schemas import (
    MessageInfo,        # 单条消息结构
    SessionDeleteResponse,  # 删除会话返回结构
    SessionInfo,        # 会话信息结构
    SessionListResponse,    # 会话列表结构
    SessionMessagesResponse # 会话消息列表结构
)

# 创建会话管理 API 路由组
router = APIRouter()


# ==============================
# 1. 获取某个会话的所有消息
# ==============================
@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,                # 要获取的会话 ID
    current_user: User = Depends(get_current_user)  # 当前登录用户
):
    try:
        # 第一步：权限校验！只允许访问自己的会话
        # 先获取当前用户的所有有效会话 ID
        valid_sessions = [s["session_id"] for s in storage.list_session_infos(current_user.username)]
        
        # 如果会话不属于当前用户 → 403 无权访问
        if session_id not in valid_sessions:
            raise HTTPException(status_code=403, detail="无权访问该会话")

        # 第二步：从 storage 中读取该会话的所有消息
        messages = [
            MessageInfo(
                type=msg["type"],          # 消息类型：user / assistant
                content=msg["content"],    # 消息内容
                timestamp=msg["timestamp"],# 时间
                rag_trace=msg.get("rag_trace"),  # RAG 检索追踪信息
            )
            for msg in storage.get_session_messages(current_user.username, session_id)
        ]
        
        # 返回消息列表
        return SessionMessagesResponse(messages=messages)
    
    # 直接抛出已知 HTTP 异常
    except HTTPException:
        raise
    # 其他未知错误 → 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 2. 获取当前用户的所有会话列表
# ==============================
@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    try:
        # 获取当前用户的所有会话信息
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        
        # 按更新时间倒序排列（最新的排在最前面）
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        
        return SessionListResponse(sessions=sessions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 3. 删除指定会话
# ==============================
@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str, 
    current_user: User = Depends(get_current_user)
):
    try:
        # 权限校验：只能删除自己的会话
        valid_sessions = [s["session_id"] for s in storage.list_session_infos(current_user.username)]
        if session_id not in valid_sessions:
            raise HTTPException(status_code=403, detail="无权删除该会话")

        # 执行删除
        deleted = storage.delete_session(current_user.username, session_id)
        
        # 删除失败（会话不存在）
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 返回成功信息
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))