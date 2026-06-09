import json
import re
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from agent import chat_with_agent_async, chat_with_agent_stream
from auth import get_current_user
from models import User
from schemas import ChatRequest, ChatResponse

# ====================== 日志配置（生产必备）======================
logger = logging.getLogger(__name__)

router = APIRouter()

# ====================== 错误码匹配规则（抽离配置）======================
ERROR_PATTERNS = {
    "embedding": (503, "嵌入模型服务暂时不可用，请稍后重试。"),
    "milvus": (503, "向量数据库服务暂时不可用，请稍后重试。"),
    "timeout": (504, "请求超时，请稍后重试。"),
    "connection": (503, "服务连接失败，请检查网络连接后重试。"),
}

# ====================== 通用错误处理函数（解耦！）======================
def handle_chat_exception(e: Exception) -> HTTPException:
    """统一处理聊天接口异常，返回标准 HTTPException"""
    message = str(e)
    logger.error(f"聊天服务异常: {message}", exc_info=True)  # 关键：打日志

    # 1. 提取错误码
    code_match = re.search(r"Error code:\s*(\d{3})", message)
    status_code = 500
    detail = message

    if code_match:
        try:
            status_code = int(code_match.group(1))
        except (ValueError, TypeError):
            status_code = 500

    # 2. 标准 HTTP 错误
    if status_code == 429:
        detail = f"上游模型服务限流/额度不足（429）。原始错误：{message}"
        return HTTPException(status_code=429, detail=detail)

    if status_code in (401, 403):
        return HTTPException(status_code=status_code, detail=message)

    # 3. 业务关键字匹配
    for keyword, (code, msg) in ERROR_PATTERNS.items():
        if keyword in message.lower():
            return HTTPException(status_code=code, detail=msg)

    # 4. 默认 500
    return HTTPException(status_code=500, detail=f"服务器内部错误：{detail}")

# ====================== 普通聊天接口 ======================
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        session_id = request.session_id or "default_session"
        logger.info(
            f"用户 [{current_user.username}] 发起聊天 | session={session_id}"
        )

        resp = await chat_with_agent_async(
            user_text=request.message,
            user_id=current_user.username,
            session_id=session_id
        )

        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)

    except Exception as e:
        raise handle_chat_exception(e) from e

# ====================== 流式聊天接口 ======================
@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            logger.info(
                f"用户 [{current_user.username}] 发起流式聊天 | session={session_id}"
            )

            logger.info(f"调用 chat_with_agent_stream, user_text={request.message!r}, user_id={current_user.username!r}, session_id={session_id!r}")
            async for chunk in chat_with_agent_stream(
                user_text=request.message,
                user_id=current_user.username,
                session_id=session_id
            ):
                yield chunk

        except Exception as e:
            import traceback
            err_msg = str(e)
            logger.error(f"流式聊天异常: {err_msg}\n完整调用栈:\n{traceback.format_exc()}")
            error_data = {"type": "error", "content": err_msg}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )