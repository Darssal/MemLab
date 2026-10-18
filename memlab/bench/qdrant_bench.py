from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

from memlab.backends.qdrant_store import QdrantStore
from memlab.metrics import summarize_latency
from memlab.tools.embedding import hash_embedding


async def _run(args: argparse.Namespace) -> None:
    store = QdrantStore(args.qdrant_url, args.collection, vector_size=args.dim)
    await store.ensure_collection()
    rng = random.Random(args.seed)

    for i in range(args.points):
        text = f"doc_{i} seed_{args.seed}"
        emb = hash_embedding(text, dim=args.dim)
        await store.upsert(
            point_id=f"{args.seed}-{i}",
            vector=emb,
            payload={"agent_id": "bench", "key": f"bench-{i}", "kind": "bench"},
        )

    latencies: list[float] = []
    for i in range(args.searches):
        q = hash_embedding(f"query_{rng.randint(0, args.points - 1)}", dim=args.dim)
        t0 = time.perf_counter()
        _ = await store.search(q, limit=args.top_k, agent_id="bench")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    summary = summarize_latency(latencies)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(str(summary), encoding="utf-8")
    print(f"Wrote {out}")
    await store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", type=str, default="http://127.0.0.1:6333")
    ap.add_argument("--collection", type=str, default="memlab_docs")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--searches", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="logs/qdrant_bench.txt")
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
