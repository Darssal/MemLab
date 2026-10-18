from __future__ import annotations

import argparse
import random
from pathlib import Path

from memlab.io import jsonl_append


def parse_ratio(ratio: str) -> tuple[int, int]:
    left, right = ratio.split(":")
    return int(left), int(right)


def bounded_lognormal_bytes(rng: random.Random, mean_bytes: int, p99_bytes: int) -> int:
    # Approximation tuned for reproducibility, not distribution fitting precision.
    sigma = 1.0
    mu = max(1.0, mean_bytes / 3.0)
    x = int(rng.lognormvariate(mu=0.0, sigma=sigma) * mu)
    return max(64, min(p99_bytes, x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--duration-s", type=int, default=300)
    ap.add_argument("--rw-ratio", type=str, default="9:1")
    ap.add_argument("--zipf-s", type=float, default=1.3)
    ap.add_argument("--mean-bytes", type=int, default=2048)
    ap.add_argument("--p99-bytes", type=int, default=131072)
    ap.add_argument("--ttl-s", type=int, default=300)
    ap.add_argument("--out", type=str, default="traces/trace_200a.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    read_w, write_w = parse_ratio(args.rw_ratio)
    op_weights = [("get", read_w), ("put", write_w)]

    # Build key popularity to emulate hot set access.
    n_keys = max(1000, args.agents * 20)
    key_weights = [1.0 / ((i + 1) ** args.zipf_s) for i in range(n_keys)]
    total = sum(key_weights)
    key_probs = [w / total for w in key_weights]
    key_cdf = []
    acc = 0.0
    for p in key_probs:
        acc += p
        key_cdf.append(acc)

    def sample_key() -> str:
        x = rng.random()
        lo, hi = 0, len(key_cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if key_cdf[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return f"k-{lo:06d}"

    # Average 20 ops/sec/agent in simulated timeline.
    total_ops = args.duration_s * args.agents * 20
    ts_ms = 0
    for _ in range(total_ops):
        agent_idx = rng.randrange(args.agents)
        agent_id = f"a{agent_idx:04d}"
        op = rng.choices([x[0] for x in op_weights], weights=[x[1] for x in op_weights], k=1)[0]
        key = sample_key()
        b = bounded_lognormal_bytes(rng, args.mean_bytes, args.p99_bytes)
        rec = {
            "ts_ms": ts_ms,
            "agent_id": agent_id,
            "op": op,
            "key": key,
            "bytes": b,
        }
        if op == "put":
            rec["ttl_s"] = args.ttl_s
            rec["importance"] = round(rng.random(), 4)
        jsonl_append(out_path, rec)
        ts_ms += rng.randint(0, 4)

    print(f"Wrote {total_ops} events to {out_path}")


if __name__ == "__main__":
    main()
