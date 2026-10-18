from __future__ import annotations

from memlab.backends.redis_store import RedisStore
from memlab.memory.types import MemoryItem


class L1RedisClient:
    def __init__(self, redis_url: str) -> None:
        self.store = RedisStore(redis_url)

    async def ping(self) -> bool:
        return await self.store.ping()

    async def put(self, item: MemoryItem, ttl_s: int | None) -> None:
        await self.store.put_item(item, ttl_s=ttl_s)

    async def put_many(self, items: list[MemoryItem], ttl_s: int | None) -> None:
        await self.store.put_items_batch(items, ttl_s=ttl_s)

    async def get_recent(self, agent_id: str, limit: int) -> list[MemoryItem]:
        return await self.store.get_recent(agent_id, limit)

    async def get_key(self, agent_id: str, key: str) -> MemoryItem | None:
        return await self.store.get_item(agent_id, key)

    async def close(self) -> None:
        await self.store.close()
