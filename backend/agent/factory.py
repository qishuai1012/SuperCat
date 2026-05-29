"""Agent创建工厂"""
import threading
from typing import Tuple
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools import (
    KB_NO_RESULT_MESSAGE,
    KB_NO_RESULT_SENTINEL,
    get_current_weather,
    search_knowledge_base,
    calculator,
    web_search,
)


class AgentFactory:
    """Agent工厂类，管理Agent实例的创建和缓存"""

    def __init__(self, api_key: str, model: str, base_url: str, temperature: float = 0.3):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._cache = {}
        self._lock = threading.Lock()

    def create_or_get(self, cache_key: str = "default") -> Tuple:
        """创建或获取缓存的Agent实例"""
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = self._create_new_agent()
            return self._cache[cache_key]

    def _create_new_agent(self):
        """创建新的Agent实例"""
        model = init_chat_model(
            model=self.model,
            model_provider="openai",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            stream_usage=True,
        )

        agent = create_agent(
            model=model,
            tools=[get_current_weather, search_knowledge_base, calculator, web_search],
            system_prompt=(
                "You are a cute cat bot that loves to help users. "
                "When responding, you may use tools to assist. "
                "Use search_knowledge_base when users ask document/knowledge questions. "
                "Use web_search when users ask about latest news, real-time information, or current events. "
                "Use calculator when users ask math calculation questions. "
                "Use get_current_weather when users ask about weather. "
                "Do not call the same tool repeatedly in one turn. "
                "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer. "
                f"If the tool output starts with '{KB_NO_RESULT_SENTINEL}:', reply with exactly this sentence and nothing else: {KB_NO_RESULT_MESSAGE} "
                "When the knowledge base has no relevant information, do not answer from your own knowledge, do not guess, and do not call additional tools."
            ),
        )

        return agent, model
