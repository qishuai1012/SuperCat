"""
智能缓存层
支持查询相似度匹配、TTL过期、LRU淘汰
"""

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Optional, Any, Dict
from cache import cache
import logging

logger = logging.getLogger(__name__)


class SmartCache:
    """智能缓存管理器"""

    def __init__(self, prefix: str = "smart_cache", cache_version: str = "v2"):
        self.prefix = prefix
        self.cache_version = cache_version

    def _serialize_for_json(self, value: Any):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return self._serialize_for_json(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._serialize_for_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_for_json(item) for item in value]
        return str(value)

    def _normalize_query_text(self, query: str) -> str:
        text = unicodedata.normalize("NFKC", str(query or ""))
        text = re.sub(r"^\s*\[ROUTE:[^\]]+\]\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.casefold()

    def _serialize_payload(self, payload: Dict[str, Any], semantic: bool = False) -> str:
        data = dict(self._serialize_for_json(payload) if payload else {})
        data["cache_version"] = self.cache_version
        if semantic:
            query = data.get("query") or data.get("canonical_query") or ""
            data["canonical_query"] = self._normalize_query_text(query)
        return json.dumps(data, sort_keys=True, ensure_ascii=False)

    def _make_key(self, query: str, cache_type: str) -> str:
        """生成缓存键"""
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:16]
        return f"{self.prefix}:{self.cache_version}:{cache_type}:{query_hash}"

    def get_route_decision(self, query: str) -> Optional[Any]:
        """获取路由决策缓存"""
        key = self._make_key(query, "route")
        return cache.get_json(key)

    def set_route_decision(self, query: str, decision: Any, ttl: int = 3600):
        """缓存路由决策"""
        key = self._make_key(query, "route")
        try:
            # 将路由决策对象转为可序列化的字典
            if hasattr(decision, '__dict__'):
                data = {
                    'strategy': decision.strategy.value if hasattr(decision.strategy, 'value') else str(decision.strategy),
                    'query_complexity': decision.query_complexity.value if hasattr(decision.query_complexity, 'value') else str(decision.query_complexity),
                    'top_k': getattr(decision, 'top_k', 5),
                    'merge_threshold': getattr(decision, 'merge_threshold', 2),
                    'agent_type': getattr(decision, 'agent_type', 'default'),
                    'parallel_paths': getattr(decision, 'parallel_paths', 1),
                    'needs_decomposition': getattr(decision, 'needs_decomposition', False),
                    'retrieval_params': self._serialize_for_json(getattr(decision, 'retrieval_params', {}) or {}),
                    'cached_at': time.time()
                }
            else:
                data = {'raw': str(decision), 'cached_at': time.time()}

            cache.set_json(key, data, ttl=ttl)
            logger.debug(f"路由决策已缓存: {query[:30]}")
        except Exception as e:
            logger.warning(f"缓存路由决策失败: {e}")

    def get_retrieval_result(self, query: str) -> Optional[Dict]:
        """获取检索结果缓存"""
        key = self._make_key(query, "retrieval")
        return cache.get_json(key)

    def get_retrieval_result_by_key(self, key_payload: Dict[str, Any]) -> Optional[Dict]:
        """按结构化键获取检索结果缓存"""
        key = self._make_key(self._serialize_payload(key_payload, semantic=False), "retrieval")
        return cache.get_json(key)

    def get_semantic_retrieval_result_by_key(self, key_payload: Dict[str, Any]) -> Optional[Dict]:
        """按语义键获取检索结果缓存"""
        key = self._make_key(self._serialize_payload(key_payload, semantic=True), "retrieval_semantic")
        return cache.get_json(key)

    def set_retrieval_result(self, query: str, result: Dict, ttl: int = 1800):
        """缓存检索结果"""
        key = self._make_key(query, "retrieval")
        try:
            cache.set_json(key, self._serialize_for_json(result), ttl=ttl)
            logger.debug(f"检索结果已缓存: {query[:30]}")
        except Exception as e:
            logger.warning(f"缓存检索结果失败: {e}")

    def set_retrieval_result_by_key(self, key_payload: Dict[str, Any], result: Dict, ttl: int = 1800):
        """按结构化键缓存检索结果"""
        key = self._make_key(self._serialize_payload(key_payload, semantic=False), "retrieval")
        try:
            cache.set_json(key, self._serialize_for_json(result), ttl=ttl)
            logger.debug("结构化检索结果已缓存")
        except Exception as e:
            logger.warning(f"缓存结构化检索结果失败: {e}")

    def set_semantic_retrieval_result_by_key(self, key_payload: Dict[str, Any], result: Dict, ttl: int = 1800):
        """按语义键缓存检索结果"""
        key = self._make_key(self._serialize_payload(key_payload, semantic=True), "retrieval_semantic")
        try:
            cache.set_json(key, self._serialize_for_json(result), ttl=ttl)
            logger.debug("语义检索结果已缓存")
        except Exception as e:
            logger.warning(f"缓存语义检索结果失败: {e}")

    def invalidate(self, query: str):
        """清除特定查询的所有缓存"""
        for cache_type in ["route", "retrieval", "retrieval_semantic"]:
            key = self._make_key(query, cache_type)
            cache.delete(key)


# 全局实例
_smart_cache = None


def get_smart_cache() -> SmartCache:
    """获取全局智能缓存"""
    global _smart_cache
    if _smart_cache is None:
        _smart_cache = SmartCache()
    return _smart_cache
