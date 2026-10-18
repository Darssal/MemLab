from __future__ import annotations

import json

try:
    from redis.asyncio import ConnectionPool, Redis
except Exception:  # pragma: no cover - optional dependency guard
    ConnectionPool = None  # type: ignore
    Redis = None  # type: ignore

from memlab.memory.types import MemoryItem


class RedisStore:
    def __init__(self, redis_url: str, *, max_connections: int = 256) -> None:
        if Redis is None or ConnectionPool is None:
            raise RuntimeError("redis package is required for L1/Redis operations")
        self._pool = ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections,
        )
        self.client = Redis(connection_pool=self._pool)
        self._max_cached_text_chars = 1024

    def _encode_item_payload(self, item: MemoryItem) -> tuple[str, str]:
        ikey = self._item_key(item.agent_id, item.key)
        payload_dict = dict(item.__dict__)
        # L1 cache does not need persisted embeddings; dropping them reduces payload size and latency.
        payload_dict["embedding"] = None
        text_value = str(payload_dict.get("text", ""))
        if len(text_value) > self._max_cached_text_chars:
            payload_dict["text"] = text_value[: self._max_cached_text_chars]
            meta = dict(payload_dict.get("metadata") or {})
            meta["l1_truncated"] = True
            payload_dict["metadata"] = meta
        payload = json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":"))
        return ikey, payload

    @staticmethod
    def _item_key(agent_id: str, key: str) -> str:
        return f"mem:{agent_id}:{key}"

    @staticmethod
    def _recent_key(agent_id: str) -> str:
        return f"mem_recent:{agent_id}"

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def put_item(self, item: MemoryItem, ttl_s: int | None = None) -> None:
        ikey, payload = self._encode_item_payload(item)
        recent_key = self._recent_key(item.agent_id)
        async with self.client.pipeline(transaction=False) as pipe:
            pipe.set(ikey, payload, ex=ttl_s)
            pipe.lpush(recent_key, item.key)
            pipe.ltrim(recent_key, 0, 2047)
            await pipe.execute()

    async def put_items_batch(self, items: list[MemoryItem], ttl_s: int | None = None) -> None:
        if not items:
            return
        async with self.client.pipeline(transaction=False) as pipe:
            for item in items:
                ikey, payload = self._encode_item_payload(item)
                recent_key = self._recent_key(item.agent_id)
                pipe.set(ikey, payload, ex=ttl_s)
                pipe.lpush(recent_key, item.key)
                pipe.ltrim(recent_key, 0, 2047)
            await pipe.execute()

    async def get_item(self, agent_id: str, key: str) -> MemoryItem | None:
        raw = await self.client.get(self._item_key(agent_id, key))
        if not raw:
            return None
        data = json.loads(raw)
        return MemoryItem(**data)

    async def get_recent(self, agent_id: str, limit: int) -> list[MemoryItem]:
        keys = await self.client.lrange(self._recent_key(agent_id), 0, max(limit - 1, 0))
        out: list[MemoryItem] = []
        if not keys:
            return out
        redis_keys = [self._item_key(agent_id, key) for key in keys]
        raw_items = await self.client.mget(redis_keys)
        for raw in raw_items:
            if not raw:
                continue
            out.append(MemoryItem(**json.loads(raw)))
        return out

    async def close(self) -> None:
        await self.client.aclose()
        await self._pool.aclose()
