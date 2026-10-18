from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from memlab.backends.object_store import LocalObjectStore
from memlab.backends.postgres_store import PostgresStore
from memlab.backends.qdrant_store import QdrantStore
from memlab.backends.redis_store import RedisStore
from memlab.metrics.stats import summarize_latency
from memlab.tools.embedding import hash_embedding


def _parse_concurrency_levels(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        value = max(1, int(txt))
        if value not in out:
            out.append(value)
    return out or [1]


async def _bench_redis(redis_url: str, n: int, concurrency: int) -> dict[str, float]:
    store = RedisStore(redis_url)
    lat: list[float] = [0.0 for _ in range(n)]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(i: int) -> None:
        async with sem:
            t0 = time.perf_counter()
            await store.client.set(f"bench:{concurrency}:{i}", f"value:{i}", ex=60)
            _ = await store.client.get(f"bench:{concurrency}:{i}")
            lat[i] = (time.perf_counter() - t0) * 1000.0

    await asyncio.gather(*[_one(i) for i in range(n)])
    await store.close()
    return summarize_latency(lat)


async def _bench_postgres(pg_uri: str, n: int, concurrency: int) -> dict[str, float]:
    store = PostgresStore(pg_uri)
    await store.connect()
    lat: list[float] = [0.0 for _ in range(n)]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(i: int) -> None:
        async with sem:
            t0 = time.perf_counter()
            async with store.pool.acquire() as conn:  # type: ignore[union-attr]
                _ = await conn.fetchval("SELECT $1::int + 1", i)
            lat[i] = (time.perf_counter() - t0) * 1000.0

    await asyncio.gather(*[_one(i) for i in range(n)])
    await store.close()
    return summarize_latency(lat)


async def _bench_qdrant(qdrant_url: str, collection: str, n: int, concurrency: int) -> dict[str, float]:
    store = QdrantStore(qdrant_url, collection, vector_size=64)
    await store.ensure_collection()
    for i in range(max(32, n)):
        emb = hash_embedding(f"doc:{i}")
        await store.upsert(i, emb, {"agent_id": "bench", "key": f"k{i}", "kind": "bench"})
    lat: list[float] = [0.0 for _ in range(n)]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(i: int) -> None:
        async with sem:
            q = hash_embedding(f"query:{concurrency}:{i}")
            t0 = time.perf_counter()
            _ = await store.search(q, limit=8, agent_id="bench")
            lat[i] = (time.perf_counter() - t0) * 1000.0

    await asyncio.gather(*[_one(i) for i in range(n)])
    await store.close()
    return summarize_latency(lat)


async def _bench_l3(root_dir: str, n: int, concurrency: int) -> dict[str, float]:
    store = LocalObjectStore(root_dir=root_dir)
    await store.ensure_ready()
    lat: list[float] = [0.0 for _ in range(n)]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(i: int) -> None:
        async with sem:
            text = f"payload-{concurrency}-{i}-" + ("x" * 1024)
            t0 = time.perf_counter()
            pointer = await store.put(agent_id="bench", key=f"obj-{concurrency}-{i}", text=text, metadata={"kind": "bench"})
            _ = await store.get(pointer)
            lat[i] = (time.perf_counter() - t0) * 1000.0

    await asyncio.gather(*[_one(i) for i in range(n)])
    return summarize_latency(lat)


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-url", type=str, default="redis://127.0.0.1:6379/0")
    ap.add_argument("--pg-uri", type=str, default=os.getenv("PG_URI", ""))
    ap.add_argument("--qdrant-url", type=str, default="http://127.0.0.1:6333")
    ap.add_argument("--qdrant-collection", type=str, default="memlab_docs")
    ap.add_argument("--l3-root", type=str, default="storage/l3_microbench")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency-levels", type=str, default="1,16,64")
    ap.add_argument("--out", type=str, default="logs/microbench.json")
    args = ap.parse_args()

    levels = _parse_concurrency_levels(args.concurrency_levels)
    result: dict[str, object] = {"concurrency_levels": levels}
    redis_by_level: dict[str, dict[str, float]] = {}
    postgres_by_level: dict[str, dict[str, float]] = {}
    qdrant_by_level: dict[str, dict[str, float]] = {}
    l3_by_level: dict[str, dict[str, float]] = {}
    for level in levels:
        redis_by_level[f"c{level}"] = await _bench_redis(args.redis_url, args.requests, level)
        postgres_by_level[f"c{level}"] = await _bench_postgres(args.pg_uri, args.requests, level)
        qdrant_by_level[f"c{level}"] = await _bench_qdrant(args.qdrant_url, args.qdrant_collection, args.requests, level)
        l3_by_level[f"c{level}"] = await _bench_l3(args.l3_root, args.requests, level)
    result["redis"] = redis_by_level
    result["postgres"] = postgres_by_level
    result["qdrant"] = qdrant_by_level
    result["l3_object_store"] = l3_by_level
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
