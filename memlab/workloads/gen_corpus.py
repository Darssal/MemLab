from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from memlab.io import jsonl_append


def random_secret(rng: random.Random, n: int = 12) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def random_body(rng: random.Random, min_bytes: int, max_bytes: int, secret: str) -> str:
    target = rng.randint(min_bytes, max_bytes)
    words = []
    while len(" ".join(words).encode("utf-8")) < target:
        token = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 11)))
        words.append(token)
    text = " ".join(words)
    return f"{text}\nSECRET_CODE:{secret}\n{text[: min(400, len(text))]}"


def make_zipf_sampler(n_docs: int, s: float) -> tuple[list[float], list[int]]:
    weights = [1.0 / (i + 1) ** s for i in range(n_docs)]
    total = sum(weights)
    probs = [w / total for w in weights]
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    return cdf, list(range(n_docs))


def sample_zipf(rng: random.Random, cdf: list[float], items: list[int]) -> int:
    x = rng.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return items[lo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--docs", type=int, default=10000)
    ap.add_argument("--tasks", type=int, default=1000)
    ap.add_argument("--doc-bytes-min", type=int, default=1024)
    ap.add_argument("--doc-bytes-max", type=int, default=8192)
    ap.add_argument("--zipf-s", type=float, default=1.2)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs_path = out / "docs.jsonl"
    tasks_path = out / "tasks.jsonl"
    if docs_path.exists():
        docs_path.unlink()
    if tasks_path.exists():
        tasks_path.unlink()

    rng = random.Random(args.seed)
    docs: list[dict] = []
    for i in range(args.docs):
        doc_id = f"d{i:05d}"
        secret = random_secret(rng)
        title = f"Document {i:05d}"
        body = random_body(rng, args.doc_bytes_min, args.doc_bytes_max, secret)
        row = {
            "doc_id": doc_id,
            "title": title,
            "body": body,
            "secret_code": secret,
        }
        docs.append(row)
        jsonl_append(docs_path, row)

    cdf, items = make_zipf_sampler(len(docs), args.zipf_s)
    for t in range(args.tasks):
        idx = sample_zipf(rng, cdf, items)
        doc = docs[idx]
        task = {
            "task_id": f"t{t:05d}",
            "doc_id": doc["doc_id"],
            "doc_title_hint": doc["title"],
            "question": f'Find the SECRET_CODE in the document titled "{doc["title"]}". Return JSON only.',
            "answer_key": doc["secret_code"],
        }
        jsonl_append(tasks_path, task)

    print(f"Wrote {args.docs} docs to {docs_path}")
    print(f"Wrote {args.tasks} tasks to {tasks_path}")


if __name__ == "__main__":
    main()
