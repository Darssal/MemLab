from __future__ import annotations

from memlab.backends.postgres_store import PostgresStore
from memlab.memory.types import MemoryItem


class L2PostgresStore:
    def __init__(self, pg_uri: str) -> None:
        self.store = PostgresStore(pg_uri)

    async def connect(self) -> None:
        await self.store.connect()

    async def ping(self) -> bool:
        return await self.store.ping()

    async def put(self, item: MemoryItem) -> None:
        await self.store.put_item(item)

    async def put_many(self, items: list[MemoryItem]) -> None:
        await self.store.put_items_batch(items)

    async def get_recent(self, agent_id: str, limit: int) -> list[MemoryItem]:
        return await self.store.get_recent(agent_id, limit)

    async def search_text(self, agent_id: str, query: str, limit: int) -> list[MemoryItem]:
        return await self.store.search_text(agent_id, query, limit)

    async def get_by_keys(self, agent_id: str, keys: list[str]) -> list[MemoryItem]:
        return await self.store.get_by_keys(agent_id, keys)

    async def checkpoint(self, agent_id: str, thread_id: str, ts_ms: int, state: dict) -> None:
        await self.store.write_checkpoint(agent_id, thread_id, ts_ms, state)

    async def close(self) -> None:
        await self.store.close()
