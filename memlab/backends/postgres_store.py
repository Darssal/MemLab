from __future__ import annotations

import json
from typing import Any

try:
    import asyncpg
except Exception:  # pragma: no cover - optional dependency guard
    asyncpg = None  # type: ignore

from memlab.memory.types import MemoryItem


UPSERT_MEMORY_SQL = """
INSERT INTO memory_items (
  key, agent_id, kind, text_body, bytes, importance, created_at_ms, last_access_ms, access_count, metadata, embedding
) VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb
)
ON CONFLICT (key, agent_id) DO UPDATE SET
  kind = EXCLUDED.kind,
  text_body = EXCLUDED.text_body,
  bytes = EXCLUDED.bytes,
  importance = EXCLUDED.importance,
  last_access_ms = EXCLUDED.last_access_ms,
  access_count = EXCLUDED.access_count,
  metadata = EXCLUDED.metadata,
  embedding = EXCLUDED.embedding
"""


class PostgresStore:
    def __init__(self, pg_uri: str, *, min_pool_size: int = 1, max_pool_size: int = 32) -> None:
        self.pg_uri = pg_uri
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool: asyncpg.Pool | None = None
        self.backend = "postgres"
        if asyncpg is None:
            raise RuntimeError("asyncpg package is required for L2/Postgres operations")

    async def connect(self) -> None:
        if self.pool is not None:
            return
        self.pool = await asyncpg.create_pool(
            self.pg_uri,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
            command_timeout=30.0,
            ssl=False,
        )

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def ping(self) -> bool:
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            v = await conn.fetchval("SELECT 1")
            return v == 1

    async def put_item(self, item: MemoryItem) -> None:
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            await conn.execute(
                UPSERT_MEMORY_SQL,
                item.key,
                item.agent_id,
                item.kind,
                item.text,
                item.bytes,
                item.importance,
                item.created_at_ms,
                item.last_access_ms,
                item.access_count,
                json.dumps(item.metadata or {}, ensure_ascii=False),
                None,
            )

    async def put_items_batch(self, items: list[MemoryItem]) -> None:
        if not items:
            return
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        args = [
            (
                item.key,
                item.agent_id,
                item.kind,
                item.text,
                item.bytes,
                item.importance,
                item.created_at_ms,
                item.last_access_ms,
                item.access_count,
                json.dumps(item.metadata or {}, ensure_ascii=False),
                None,
            )
            for item in items
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(UPSERT_MEMORY_SQL, args)

    async def get_recent(self, agent_id: str, limit: int) -> list[MemoryItem]:
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key, agent_id, kind, text_body, bytes, importance, created_at_ms, last_access_ms,
                       access_count, metadata, embedding
                FROM memory_items
                WHERE agent_id = $1
                ORDER BY last_access_ms DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )
        return [self._row_to_item(r) for r in rows]

    async def search_text(self, agent_id: str, query: str, limit: int) -> list[MemoryItem]:
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        like = f"%{query}%"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key, agent_id, kind, text_body, bytes, importance, created_at_ms, last_access_ms,
                       access_count, metadata, embedding
                FROM memory_items
                WHERE agent_id = $1 AND text_body ILIKE $2
                ORDER BY last_access_ms DESC
                LIMIT $3
                """,
                agent_id,
                like,
                limit,
            )
        return [self._row_to_item(r) for r in rows]

    async def get_by_keys(self, agent_id: str, keys: list[str]) -> list[MemoryItem]:
        if not keys:
            return []
        deduped_keys = list(dict.fromkeys(keys))
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key, agent_id, kind, text_body, bytes, importance, created_at_ms, last_access_ms,
                       access_count, metadata, embedding
                FROM memory_items
                WHERE agent_id = $1 AND key = ANY($2::text[])
                """,
                agent_id,
                deduped_keys,
            )
        by_key = {r["key"]: self._row_to_item(r) for r in rows}
        return [by_key[k] for k in deduped_keys if k in by_key]

    async def write_checkpoint(self, agent_id: str, thread_id: str, ts_ms: int, state_json: dict[str, Any]) -> None:
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints (agent_id, thread_id, ts_ms, state_json)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                agent_id,
                thread_id,
                ts_ms,
                json.dumps(state_json, ensure_ascii=False),
            )

    @staticmethod
    def _row_to_item(r: Any) -> MemoryItem:
        metadata = r["metadata"] or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        embedding = r["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        return MemoryItem(
            key=r["key"],
            agent_id=r["agent_id"],
            kind=r["kind"],
            text=r["text_body"],
            bytes=int(r["bytes"]),
            importance=float(r["importance"]),
            created_at_ms=int(r["created_at_ms"]),
            last_access_ms=int(r["last_access_ms"]),
            access_count=int(r["access_count"]),
            metadata=metadata,
            embedding=embedding,
        )
