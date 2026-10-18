from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_longmemeval_profile(question: str, question_type: str) -> tuple[str, str]:
    q = str(question or "").strip().lower()
    qtype = str(question_type or "").strip().lower()
    if "preference" in qtype or re.search(r"\b(recommend|suggest|prefer|would like)\b", q):
        return "summary", "token_f1"
    if "temporal-reasoning" in qtype and re.search(r"\b(order|first to last|chronological)\b", q):
        return "summary", "token_f1"
    return "short_answer", "relaxed_span"


def main() -> None:
    # Optional dependencies are required only for the explicit download command.
    import pandas as pd
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/benchmarks/longmemeval")
    ap.add_argument("--repo-id", type=str, default="yuyijiong/LongMemEval_24k")
    ap.add_argument("--filename", type=str, default="longmemeval_24k.parquet")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_path = Path(
        hf_hub_download(
            repo_id=str(args.repo_id),
            repo_type="dataset",
            filename=str(args.filename),
        )
    )
    df = pd.read_parquet(raw_path)
    rows = df.to_dict(orient="records")
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    if int(args.max_samples) > 0:
        rows = rows[: max(1, int(args.max_samples))]

    tasks_public: list[dict[str, Any]] = []
    tasks_internal: list[dict[str, Any]] = []
    docs_internal: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        task_id = f"lme_{i:06d}"
        doc_id = f"longmemeval_doc_{i:06d}"
        question = str(row.get("question", "")).strip()
        context = str(row.get("prompt", ""))
        gold = str(row.get("answer", "")).strip()
        qtype = str(row.get("question_type", "unknown")).strip()
        prompt_len = int(row.get("prompt_len", len(context)))
        qa_prompt_profile, answer_metric_hint = _resolve_longmemeval_profile(question, qtype)
        metadata = {
            "question_type": qtype,
            "prompt_len": prompt_len,
            "context_chars": len(context),
            "task_type": "long_context_qa",
            "memory_needed": True,
            "answer_metric_hint": answer_metric_hint,
            "qa_prompt_profile": qa_prompt_profile,
        }
        tasks_public.append(
            {
                "task_id": task_id,
                "source_dataset": "longmemeval",
                "question": question,
                "context": context,
                "dialogue": "",
                "docs": "",
                "gold_answer": gold,
                "metadata": metadata,
            }
        )
        tasks_internal.append(
            {
                "task_id": task_id,
                "doc_id": doc_id,
                "doc_title_hint": f"LongMemEval {qtype}",
                "question": question,
                "answer_key": gold,
                "gold_answer": gold,
                "source_dataset": "longmemeval",
                "task_type": "qa",
                "answer_field": "final_answer",
                "metadata": metadata,
            }
        )
        docs_internal.append(
            {
                "doc_id": doc_id,
                "title": f"LongMemEval Prompt {qtype}",
                "body": context,
            }
        )

    _write_jsonl(out_dir / "tasks_public.jsonl", tasks_public)
    _write_jsonl(out_dir / "tasks_internal.jsonl", tasks_internal)
    _write_jsonl(out_dir / "docs_internal.jsonl", docs_internal)
    manifest = {
        "dataset": "longmemeval",
        "source_repo": str(args.repo_id),
        "source_file": str(args.filename),
        "raw_path": str(raw_path),
        "sample_count": len(tasks_public),
        "seed": int(args.seed),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
