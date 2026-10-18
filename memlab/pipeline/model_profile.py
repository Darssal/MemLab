from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from memlab.io import now_ms


MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def estimate_model_size_b(model_id: str) -> float | None:
    match = MODEL_SIZE_RE.search(str(model_id or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def classify_model_scale(size_b: float | None) -> str:
    if size_b is None:
        return "unknown"
    if size_b <= 8.0:
        return "small"
    if size_b <= 34.0:
        return "medium"
    return "large"


def runtime_recommendation(scale: str) -> dict[str, Any]:
    if scale == "small":
        return {
            "e2e_inflight": 2,
            "worker_processes": 1,
            "sessions_per_process": 50,
            "context_top_k": 12,
            "max_evidence_chars": 3000,
            "answer_retry_count": 1,
        }
    if scale == "medium":
        return {
            "e2e_inflight": 2,
            "worker_processes": 1,
            "sessions_per_process": 36,
            "context_top_k": 10,
            "max_evidence_chars": 2400,
            "answer_retry_count": 1,
        }
    if scale == "large":
        return {
            "e2e_inflight": 1,
            "worker_processes": 1,
            "sessions_per_process": 24,
            "context_top_k": 8,
            "max_evidence_chars": 1800,
            "answer_retry_count": 0,
        }
    # Unknown defaults to conservative medium-like settings.
    return {
        "e2e_inflight": 1,
        "worker_processes": 1,
        "sessions_per_process": 30,
        "context_top_k": 10,
        "max_evidence_chars": 2200,
        "answer_retry_count": 1,
    }


def training_recommendation(scale: str) -> dict[str, Any]:
    if scale == "small":
        return {
            "global_batch_size": 64,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "precision": "bf16",
        }
    if scale == "medium":
        return {
            "global_batch_size": 32,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "precision": "bf16",
        }
    if scale == "large":
        return {
            "global_batch_size": 8,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "precision": "bf16",
        }
    return {
        "global_batch_size": 16,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "precision": "bf16",
    }


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    providers = payload.get("providers", [])
    out: list[dict[str, Any]] = []
    for provider in providers:
        provider_name = str(provider.get("name", "")).strip()
        base_url = str(provider.get("base_url", "")).strip().rstrip("/")
        for model in provider.get("models", []) or []:
            out.append(
                {
                    "provider": provider_name,
                    "base_url": base_url,
                    "model_id": str(model.get("id", "")).strip(),
                    "tags": [str(t) for t in (model.get("tags", []) or [])],
                }
            )
    return out


def _resolve_registry_entry(
    entries: list[dict[str, Any]],
    *,
    model_id: str,
    base_url: str,
) -> dict[str, Any] | None:
    if not entries:
        return None
    model_id_norm = str(model_id or "").strip()
    base_url_norm = str(base_url or "").strip().rstrip("/")
    exact = [
        e
        for e in entries
        if e.get("model_id") == model_id_norm
        and str(e.get("base_url", "")).strip().rstrip("/") == base_url_norm
    ]
    if exact:
        return exact[0]
    by_model = [e for e in entries if e.get("model_id") == model_id_norm]
    if by_model:
        return by_model[0]
    return None


def build_profile(
    *,
    model_id: str,
    base_url: str,
    registry_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    size_b = estimate_model_size_b(model_id)
    scale = classify_model_scale(size_b)
    entry = _resolve_registry_entry(registry_entries, model_id=model_id, base_url=base_url)
    provider = entry.get("provider") if entry else "unknown"
    tags = entry.get("tags") if entry else []
    profile = {
        "generated_at_ms": now_ms(),
        "model": {
            "provider": provider,
            "base_url": str(base_url or "").strip().rstrip("/"),
            "model_id": str(model_id or "").strip(),
            "estimated_params_b": size_b,
            "scale_class": scale,
            "tags": tags,
        },
        "interface_contract": {
            "api_style": "openai_compatible_chat",
            "required_env": ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"],
            "request_schema": {
                "required_fields": ["model", "messages"],
                "optional_fields": ["temperature", "response_format", "max_tokens", "stream"],
            },
            "response_schema": {
                "required_fields": ["choices"],
                "usage_fields": ["prompt_tokens", "completion_tokens"],
            },
        },
        "recommended_runtime": runtime_recommendation(scale),
        "recommended_training": training_recommendation(scale),
        "metric_contract": {
            "performance": ["throughput_tasks_per_min", "processing_p95_ms", "end_to_end_p95_ms"],
            "cost": ["prompt_tokens_per_success", "total_tokens_per_success", "context_injected_tokens_mean"],
            "amplification": ["memory_ops_per_task", "db_queries_per_task", "vector_searches_per_task"],
            "reliability": ["success_rate", "retry_count", "no_answer_rate", "parse_failure_rate"],
        },
    }
    return profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=str, default="configs/model_registry.json")
    ap.add_argument("--model-id", type=str, default="")
    ap.add_argument("--base-url", type=str, default="")
    ap.add_argument("--out", type=str, default="artifacts/model_runtime_profile.json")
    args = ap.parse_args()

    model_id = str(args.model_id or os.getenv("OPENAI_MODEL", "")).strip()
    base_url = str(args.base_url or os.getenv("OPENAI_BASE_URL", "")).strip().rstrip("/")
    if not model_id:
        model_id = "unknown-model"
    if not base_url:
        base_url = "unknown-base-url"

    registry_entries = _load_registry(Path(args.registry))
    profile = build_profile(model_id=model_id, base_url=base_url, registry_entries=registry_entries)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "scale_class": profile["model"]["scale_class"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
