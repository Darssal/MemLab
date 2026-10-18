from __future__ import annotations

import argparse
import os
import platform
import random
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memlab.config import MemlabConfig, load_config
from memlab.io import jsonl_read, now_ms
from memlab.runner_core import normalize_variant
from memlab.types import RunnerConfig


SECRET_RE = re.compile(r"SECRET_CODE[:=]\s*([A-Z0-9]+)")


def parse_args(default_agents: int = 50) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/memlab.default.yaml")
    ap.add_argument("--variant", type=str, default=None, help="B0|B1|V1|F1|F2 (legacy aliases: A1|A2)")
    ap.add_argument("--agents", type=int, default=default_agents)
    ap.add_argument("--repeat-idx", type=int, default=0)
    return ap.parse_args()


def apply_variant(cfg: MemlabConfig, variant: str) -> MemlabConfig:
    canonical, legacy_alias = normalize_variant(variant)
    cfg.experiment.variant_id = canonical
    if canonical == "B0":
        cfg.memory.l1.enabled = False
        cfg.memory.l2.enabled = False
        cfg.memory.l3.enabled = False
        cfg.experiment.summarization_on = False
    elif canonical == "B1":
        cfg.memory.l1.enabled = False
        cfg.memory.l2.enabled = False
        cfg.memory.l3.enabled = False
        cfg.experiment.summarization_on = True
    elif canonical == "V1":
        cfg.memory.l1.enabled = False
        cfg.memory.l2.enabled = True
        cfg.memory.l3.enabled = False
        cfg.experiment.summarization_on = True
    elif canonical == "F1":
        cfg.memory.l1.enabled = True
        cfg.memory.l2.enabled = True
        cfg.memory.l3.enabled = False
        cfg.experiment.summarization_on = True
    elif canonical == "F2":
        cfg.memory.l1.enabled = True
        cfg.memory.l2.enabled = True
        cfg.memory.l3.enabled = True
        cfg.experiment.summarization_on = True
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if legacy_alias == "A2":
        cfg.experiment.redis_policy = "allkeys-lru"
    return cfg


def load_runtime_config(args: argparse.Namespace) -> tuple[MemlabConfig, RunnerConfig]:
    cfg = load_config(args.config)
    if args.variant:
        cfg = apply_variant(cfg, args.variant)
    cfg.experiment.n_agents = args.agents
    rcfg = RunnerConfig(
        variant_id=cfg.experiment.variant_id,
        n_agents=cfg.experiment.n_agents,
        llm_backend=cfg.experiment.llm_backend,
        seed=cfg.experiment.seed,
        duration_s=cfg.experiment.duration_s,
        redis_policy=cfg.experiment.redis_policy,
        summarization_on=cfg.experiment.summarization_on,
        repeat_idx=args.repeat_idx,
    )
    return cfg, rcfg


def load_docs(path: str) -> dict[str, dict[str, Any]]:
    rows = jsonl_read(path)
    return {r["doc_id"]: r for r in rows}


def load_tasks(path: str) -> list[dict[str, Any]]:
    return jsonl_read(path)


def extract_secret_code(text: str) -> str:
    m = SECRET_RE.search(text)
    return m.group(1) if m else ""


def doc_search(task: dict[str, Any], docs: dict[str, dict[str, Any]], rng: random.Random) -> list[str]:
    if task.get("doc_id") in docs:
        primary = task["doc_id"]
    else:
        primary = rng.choice(list(docs.keys()))
    # Return primary + a few distractors for realistic retrieval calls.
    other = [k for k in docs.keys() if k != primary]
    rng.shuffle(other)
    return [primary] + other[:4]


def task_accuracy(pred_secret: str, answer_key: str) -> float:
    return 1.0 if pred_secret == answer_key else 0.0


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out.decode("utf-8").strip()


def build_manifest(
    *,
    run_id: str,
    config_path: str,
    runner_config: RunnerConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ts_ms": now_ms(),
        "config_path": str(Path(config_path).resolve()),
        "runner_config": asdict(runner_config),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "env": {
            "LLM_BACKEND": os.getenv("LLM_BACKEND", ""),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
            "MEMLAB_ENABLE_CHECKPOINTER": os.getenv("MEMLAB_ENABLE_CHECKPOINTER", ""),
        },
        "git_commit": get_git_commit(),
        "extra": extra or {},
    }
