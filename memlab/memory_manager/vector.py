from __future__ import annotations

from typing import Any

from memlab.backends.qdrant_store import QdrantStore


class QdrantVectorStore:
    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        vector_size: int = 64,
        shard_number: int = 1,
    ) -> None:
        self.store = QdrantStore(
            qdrant_url,
            collection,
            vector_size=vector_size,
            shard_number=shard_number,
        )

    async def ensure_collection(self) -> None:
        await self.store.ensure_collection()

    async def ping(self) -> bool:
        return await self.store.ping()

    async def upsert(self, point_id: str | int, vector: list[float], payload: dict[str, Any]) -> None:
        await self.store.upsert(point_id=point_id, vector=vector, payload=payload)

    async def upsert_many(self, points: list[dict[str, Any]]) -> None:
        await self.store.upsert_many(points)

    async def search(self, query_vector: list[float], *, limit: int, agent_id: str | None = None) -> list[dict[str, Any]]:
        return await self.store.search(query_vector, limit=limit, agent_id=agent_id)

    async def close(self) -> None:
        await self.store.close()
