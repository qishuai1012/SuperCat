"""
Agent模块兼容层
保持向后兼容，所有功能已重构到agent/目录
"""

# 同步/流式接口统一使用新模块
from agent import (
    chat_with_agent,
    chat_with_agent_async,
    chat_with_agent_stream,
    storage,
    get_storage,
    get_processor,
    ConversationStorage,
    AgentFactory,
    AgenticRAGProcessor,
)

__all__ = [
    'chat_with_agent',
    'chat_with_agent_async',
    'chat_with_agent_stream',
    'storage',
    'ConversationStorage',
    'AgentFactory',
    'AgenticRAGProcessor',
    'get_storage',
    'get_processor',
]
