"""
Async Redis cache for ContextObject results.

Cache key derivation
--------------------
SHA-256 is computed over a canonical JSON string of the source's connection
details (path, bucket+key, query, URL, etc.).  The source type is included as
a prefix so keys from different source types never collide even if their
connection strings happen to be identical.

  key = "cia:{source_type}:{sha256_hex[:16]}"

Using only the first 16 hex chars (64 bits) keeps keys short while keeping
collision probability negligible for any realistic number of sources.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from redis.asyncio import Redis, from_url

from app.config import settings
from app.models.context import ContextObject
from app.models.sources import DataSource

_KEY_PREFIX = "cia"


class ContextCache:
    def __init__(self) -> None:
        self._redis: Optional[Redis] = None

    async def _get_client(self) -> Redis:
        if self._redis is None:
            self._redis = from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    # ── Key generation ────────────────────────────────────────────────────────

    @staticmethod
    def generate_key(source: DataSource) -> str:
        """
        Deterministic SHA-256 key derived from the source's connection details.
        Only connection-relevant fields are hashed (credentials are excluded).
        """
        source_dict = source.model_dump()
        source_type = source_dict.get("type", "unknown")

        # Remove credential fields — they must not influence the cache key
        # (same data, different creds → same cached result)
        for sensitive in (
            "aws_access_key_id", "aws_secret_access_key",
            "password", "private_key_path",
            "credentials_path", "google_application_credentials",
        ):
            source_dict.pop(sensitive, None)

        canonical = json.dumps(source_dict, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return f"{_KEY_PREFIX}:{source_type}:{digest}"

    # ── Get ───────────────────────────────────────────────────────────────────

    async def get_context(self, key: str) -> Optional[ContextObject]:
        client = await self._get_client()
        raw = await client.get(key)
        if raw is None:
            return None
        return ContextObject.model_validate_json(raw)

    # ── Set ───────────────────────────────────────────────────────────────────

    async def set_context(self, key: str, context: ContextObject) -> None:
        client = await self._get_client()
        serialised = context.model_dump_json()
        await client.set(key, serialised, ex=settings.context_ttl_seconds)
