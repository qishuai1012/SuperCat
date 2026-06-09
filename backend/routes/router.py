# 导入 FastAPI 路由核心类
from fastapi import APIRouter

# 导入 4 个业务模块的路由（把分散的接口全部引进来）
# 1. 登录、注册、认证相关路由
from .auth_routes import router as auth_router
# 2. 聊天接口路由（普通对话 + 流式输出）
from .chat_routes import router as chat_router
# 3. 文档管理路由（上传、删除、列出知识库文件）
from .document_routes import router as document_router
# 4. 会话管理路由（查询历史、删除会话）
from .session_routes import router as session_router

# 创建一个【根路由 / 总路由】
router = APIRouter()

# 将 4 个子路由全部挂载到总路由上
# 所有接口都会被整合到一起
router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(document_router)
router.include_router(session_router)