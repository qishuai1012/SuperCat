import json
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent import chat_with_agent_async, chat_with_agent_stream
from auth import get_current_user
from models import User
from schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = await chat_with_agent_async(request.message, current_user.username, session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        code = None
        if match:
            try:
                code = int(match.group(1))
            except (ValueError, TypeError):
                code = 500

        if code == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                    f"原始错误：{message}"
                ),
            )
        if code in (401, 403):
            raise HTTPException(status_code=code, detail=message)
        if code:
            raise HTTPException(status_code=code, detail=message)

        if "embedding" in message.lower():
            raise HTTPException(status_code=503, detail="嵌入模型服务暂时不可用，请稍后重试。")
        if "milvus" in message.lower():
            raise HTTPException(status_code=503, detail="向量数据库服务暂时不可用，请稍后重试。")
        if "timeout" in message.lower():
            raise HTTPException(status_code=504, detail="请求超时，请稍后重试。")
        if "connection" in message.lower():
            raise HTTPException(status_code=503, detail="服务连接失败，请检查网络连接后重试。")

        raise HTTPException(status_code=500, detail=f"内部服务器错误: {message}")


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
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
