import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "supermew")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        try:
            value = self._get_client().get(self._key(key))
            return json.loads(value) if value else None
        except Exception as e:
            logger.debug(f"缓存读取失败 key={key}: {e}")
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            self._get_client().setex(self._key(key), ttl or self.default_ttl, json.dumps(value, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"缓存写入失败 key={key}: {e}")

    def delete(self, key: str) -> None:
        try:
            self._get_client().delete(self._key(key))
        except Exception as e:
            logger.warning(f"缓存删除失败 key={key}: {e}")

    def delete_pattern(self, pattern: str) -> None:
        try:
            keys = self._get_client().keys(self._key(pattern))
            if keys:
                self._get_client().delete(*keys)
        except Exception as e:
            logger.warning(f"缓存批量删除失败 pattern={pattern}: {e}")


cache = RedisCache()


class SmartCache:
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
            return {str(k): self._serialize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_for_json(i) for i in value]
        return str(value)

    def _normalize_query_text(self, query: str) -> str:
        text = unicodedata.normalize("NFKC", str(query or ""))
        text = re.sub(r"^\s*\[ROUTE:[^\]]+\]\s*", "", text)
        return re.sub(r"\s+", " ", text).strip().casefold()

    def _serialize_payload(self, payload: Dict[str, Any], semantic: bool = False) -> str:
        data = dict(self._serialize_for_json(payload) if payload else {})
        data["cache_version"] = self.cache_version
        if semantic:
            query = data.get("query") or data.get("canonical_query") or ""
            data["canonical_query"] = self._normalize_query_text(query)
        return json.dumps(data, sort_keys=True, ensure_ascii=False)

    def _make_key(self, query: str, cache_type: str) -> str:
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:16]
        return f"{self.prefix}:{self.cache_version}:{cache_type}:{query_hash}"

    def get_route_decision(self, query: str) -> Optional[Any]:
        return cache.get_json(self._make_key(query, "route"))

    def set_route_decision(self, query: str, decision: Any, ttl: int = 3600):
        key = self._make_key(query, "route")
        try:
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
                    'cached_at': time.time(),
                }
            else:
                data = {'raw': str(decision), 'cached_at': time.time()}
            cache.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning(f"缓存路由决策失败: {e}")

    def get_retrieval_result(self, query: str) -> Optional[Dict]:
        return cache.get_json(self._make_key(query, "retrieval"))

    def get_retrieval_result_by_key(self, key_payload: Dict[str, Any]) -> Optional[Dict]:
        return cache.get_json(self._make_key(self._serialize_payload(key_payload, semantic=False), "retrieval"))

    def get_semantic_retrieval_result_by_key(self, key_payload: Dict[str, Any]) -> Optional[Dict]:
        return cache.get_json(self._make_key(self._serialize_payload(key_payload, semantic=True), "retrieval_semantic"))

    def set_retrieval_result(self, query: str, result: Dict, ttl: int = 1800):
        try:
            cache.set_json(self._make_key(query, "retrieval"), self._serialize_for_json(result), ttl=ttl)
        except Exception as e:
            logger.warning(f"缓存检索结果失败: {e}")

    def set_retrieval_result_by_key(self, key_payload: Dict[str, Any], result: Dict, ttl: int = 1800):
        try:
            cache.set_json(self._make_key(self._serialize_payload(key_payload, semantic=False), "retrieval"), self._serialize_for_json(result), ttl=ttl)
        except Exception as e:
            logger.warning(f"缓存结构化检索结果失败: {e}")

    def set_semantic_retrieval_result_by_key(self, key_payload: Dict[str, Any], result: Dict, ttl: int = 1800):
        try:
            cache.set_json(self._make_key(self._serialize_payload(key_payload, semantic=True), "retrieval_semantic"), self._serialize_for_json(result), ttl=ttl)
        except Exception as e:
            logger.warning(f"缓存语义检索结果失败: {e}")

    def invalidate(self, query: str):
        for cache_type in ["route", "retrieval", "retrieval_semantic"]:
            cache.delete(self._make_key(query, cache_type))


_smart_cache = None


def get_smart_cache() -> SmartCache:
    global _smart_cache
    if _smart_cache is None:
        _smart_cache = SmartCache()
    return _smart_cache
