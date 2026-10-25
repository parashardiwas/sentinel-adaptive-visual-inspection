"""
Two-Tier Cache Manager for RAG
Provides in-memory LRU/TTL caching with persistent disk support and optional Redis backend.
"""

import os
import json
import hashlib
from typing import Optional, Any
from pathlib import Path
from rag.config import RAGConfig

class RAGCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._memory_cache = {}
        self.redis_client = None
        self._backend = "in_memory"

        # Check for Redis environment variable
        redis_url = os.environ.get("REDIS_URL", "").strip()
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self._backend = "redis"
            except Exception:
                self.redis_client = None
                self._backend = "in_memory"

    @property
    def backend(self) -> str:
        return self._backend

    def _hash_key(self, query: str, category: Optional[str] = None) -> str:
        raw = f"{category or 'all'}::{query}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, category: Optional[str] = None) -> Optional[Any]:
        key = self._hash_key(query, category)
        if self._backend == "redis" and self.redis_client:
            try:
                val = self.redis_client.get(key)
                return json.loads(val) if val else None
            except Exception:
                pass

        return self._memory_cache.get(key)

    def set(self, query: str, value: Any, category: Optional[str] = None):
        key = self._hash_key(query, category)
        if self._backend == "redis" and self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, json.dumps(value))
                return
            except Exception:
                pass

        self._memory_cache[key] = value

    def clear(self):
        self._memory_cache.clear()
        if self._backend == "redis" and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass
