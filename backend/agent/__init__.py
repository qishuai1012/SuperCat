"""Agent模块 - 向后兼容的对外接口"""
import asyncio
import json
import logging
from typing import Any, Optional
from .config import AgentConfig
from .storage import ConversationStorage
from .factory import AgentFactory

logger = logging.getLogger(__name__)

# 全局实例
_config: Optional[AgentConfig] = None
_storage: Optional[ConversationStorage] = None
_factory: Optional[AgentFactory] = None
_processor: Optional[Any] = None


def _init_system():
    """初始化系统"""
    global _config, _storage, _factory, _processor

    if _processor is not None:
        return

    # 加载配置
    _config = AgentConfig.from_env()

    # 创建存储
    _storage = ConversationStorage()

    # 创建工厂
    _factory = AgentFactory(
        api_key=_config.api_key,
        model=_config.model,
        base_url=_config.base_url,
        temperature=_config.temperature
    )

    from .processor import AgenticRAGProcessor
    from .strategies.simple_strategy import SimpleQueryStrategy
    from .strategies.complex_strategy import ComplexQueryStrategy

    # 创建策略
    strategies = [
        SimpleQueryStrategy(_factory, _storage),
    ]

    # 如果启用多智能体，添加复杂策略
    if _config.enable_multi_agent:
        try:
            from multi_agent_orchestrator import get_multi_agent_orchestrator
            orchestrator = get_multi_agent_orchestrator()
            strategies.append(ComplexQueryStrategy(orchestrator, _factory, _storage))
        except Exception as e:
            logger.warning(f"多智能体初始化失败: {e}")

    # 创建处理器
    try:
        from intelligent_router import get_intelligent_router
        from reflection_agent import get_reflection_agent
        router = get_intelligent_router()
        reflection = get_reflection_agent()
    except Exception as e:
        logger.warning(f"路由器/反思Agent初始化失败: {e}")
        router = None
        reflection = None

    _processor = AgenticRAGProcessor(
        strategies=strategies,
        storage=_storage,
        router=router,
        reflection=reflection,
        config=_config
    )


# 向后兼容的接口
def get_storage():
    """获取存储实例"""
    _init_system()
    return _storage


def get_processor():
    """获取处理器实例"""
    _init_system()
    return _processor


async def chat_with_agent_async(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """异步对话接口"""
    try:
        processor = get_processor()
        return await processor.process_query(user_text, user_id, session_id)
    except Exception as e:
        logger.exception(f"对话处理失败: {e}")
        raise


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """同步对话接口（向后兼容）"""
    return asyncio.run(chat_with_agent_async(user_text, user_id, session_id))


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """流式对话接口（向后兼容）"""
    processor = get_processor()
    async for event in processor.process_query_stream(user_text, user_id, session_id):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# 延迟初始化storage
class _StorageProxy:
    """Storage代理，延迟初始化"""
    def __getattr__(self, name):
        _init_system()
        return getattr(_storage, name)

storage = _StorageProxy()
