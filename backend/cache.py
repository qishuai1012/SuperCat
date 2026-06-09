import json
import logging
import os
from typing import Any, Optional

import redis

# 日志
logger = logging.getLogger(__name__)


# Redis 缓存工具类（全局通用，用于加速 RAG、会话、接口数据）
class RedisCache:
    # 初始化：从环境变量读取配置
    def __init__(self):
        # Redis 连接地址（默认本地）
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        # 缓存 key 前缀，防止与其他项目冲突
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "supermew")
        # 默认缓存过期时间（秒），默认 5 分钟
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        # Redis 客户端（懒加载）
        self._client = None

    # 获取 Redis 客户端（单例，只创建一次）
    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    # 给 key 加上前缀，避免冲突
    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    # ===================== 对外方法 =====================

    # 读取 JSON 格式的缓存
    def get_json(self, key: str) -> Optional[Any]:
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            # 把字符串转回 JSON 对象
            return json.loads(value)
        except Exception as e:
            logger.debug(f"缓存读取失败 key={key}: {e}")
            return None

    # 写入 JSON 格式的缓存（自动设置过期时间）
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            # 对象 → JSON 字符串
            payload = json.dumps(value, ensure_ascii=False)
            # 写入并设置过期
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception as e:
            logger.warning(f"缓存写入失败 key={key}: {e}")

    # 删除指定缓存
    def delete(self, key: str) -> None:
        try:
            self._get_client().delete(self._key(key))
        except Exception as e:
            logger.warning(f"缓存删除失败 key={key}: {e}")

    # 批量删除缓存（按通配符，如 user:*）
    def delete_pattern(self, pattern: str) -> None:
        try:
            # 1. 给传入的 pattern 加上统一前缀
            full_pattern = self._key(pattern)
            
            # 2. 根据通配符查找所有匹配的 key
            keys = self._get_client().keys(full_pattern)
            
            # 3. 如果找到 keys，就一次性全部删除
            if keys:
                self._get_client().delete(*keys)

        # 4. 出错不崩溃，只打警告日志
        except Exception as e:
            logger.warning(f"缓存批量删除失败 pattern={pattern}: {e}")


# 创建全局单例，整个项目共用一个缓存实例
cache = RedisCache()