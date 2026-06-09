from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import os
import logging

# ====================== 环境变量配置（离线运行 + 线程安全）======================
# 强制 Hugging Face 离线模式，不联网下载模型/权重
os.environ["HF_HUB_OFFLINE"] = "1"
# 禁止 transformers 联网，仅使用本地模型
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 关闭分词器多线程，避免死锁/报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入 API 路由和数据库初始化工具
import api as api_module
from database import init_db

# ====================== 日志系统配置 ======================
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志输出格式
)

# ====================== 路径配置 ======================
# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
# 前端静态文件目录（存放网页、JS、CSS）
FRONTEND_DIR = BASE_DIR / "frontend"


# ====================== 创建 FastAPI 应用 ======================
def create_app() -> FastAPI:
    # 初始化 FastAPI 实例
    app = FastAPI(title="Cute Cat Bot API")

    # ====================== 服务启动时自动执行 ======================
    @app.on_event("startup")
    async def _startup_init_db():
        """服务启动时：初始化数据库 + 预热 Embedding 模型"""
        # 初始化数据库（建表、创建索引）
        init_db()

        # 懒加载 embedding 服务，避免循环导入
        from embedding import embedding_service
        # 异步预热 embedding 模型（加载到内存，加速第一次请求）
        await asyncio.to_thread(embedding_service.warmup)

    # ====================== 跨域中间件 ======================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            # 允许所有域名访问
        allow_credentials=True,         # 允许携带 Cookie
        allow_methods=["*"],             # 允许所有 HTTP 方法
        allow_headers=["*"],             # 允许所有请求头
    )

    # ====================== 无缓存中间件（开发环境用）======================
    @app.middleware("http")
    async def _no_cache(request, call_next):
        """让前端页面不缓存，修改后立即生效"""
        response = await call_next(request)
        path = request.url.path or ""
        # 对 HTML / JS / CSS 禁用缓存
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # ====================== 注册 API 路由 ======================
    # 把聊天、检索、文件上传等接口注册到 app
    app.include_router(api_module.router)

    # ====================== 挂载前端静态页面 ======================
    # 如果存在前端目录，则将根路径映射为网页
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


# 创建应用实例（供 uvicorn 运行）
app = create_app()

# ====================== 启动命令 ======================
if __name__ == "__main__":
    import uvicorn
    # 启动服务，0.0.0.0 允许局域网访问，端口 8000
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))