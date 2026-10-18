from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    run_id: str = "F1"
    variant_id: str = "F1"
    seed: int = 42
    repetitions: int = 3
    n_agents: int = 50
    duration_s: int = 300
    llm_backend: Literal["stub", "openai_compatible", "deerflow"] = "stub"
    redis_policy: Literal["allkeys-lfu", "allkeys-lru"] = "allkeys-lfu"
    summarization_on: bool = True
    output_dir: str = "./runs"
    logs_dir: str = "./runs"


class L0Config(BaseModel):
    max_prompt_tokens: int = 8000
    max_memory_injection_tokens: int = 2000
    max_injected_tokens: int = 256
    keep_recent_messages: int = 10
    summarize_trigger_tokens: int = 7000
    summarize_target_tokens: int = 3000
    min_relevance_score: float = 0.2


class L1Config(BaseModel):
    enabled: bool = True
    ttl_s: int = 300
    max_object_bytes: int = 65536
    reuse_threshold: float = 0.2


class L2Config(BaseModel):
    enabled: bool = True
    vector_enabled: bool = True
    vector_backend: Literal["qdrant"] = "qdrant"
    qdrant_shard_number: int = 1
    default_top_k: int = 8
    collection_name: str = "memlab_docs"
    write_batch_size: int = 512
    write_flush_interval_ms: int = 50
    min_write_importance: float = 0.7
    max_batch_per_flush: int = 512
    max_flush_ms: int = 50
    drop_queue_depth_threshold: int = 12000
    backpressure_drop_importance_threshold: float = 0.85
    embedding_cache_ttl_ms: int = 120000
    retrieval_cache_ttl_ms: int = 15000


class L3Config(BaseModel):
    enabled: bool = False
    root_dir: str = "./storage/l3_objects"
    pointer_threshold_bytes: int = 65536
    pointer_preview_chars: int = 240


class PolicyConfig(BaseModel):
    packing_budget: Literal["strict", "balanced"] = "balanced"
    checkpoint_policy: Literal["l2_only", "disabled"] = "l2_only"
    direct_evidence_retrieval_bypass: bool = True


class MemoryConfig(BaseModel):
    enabled_tiers: list[Literal["L0", "L1", "L2", "L3"]] = Field(
        default_factory=lambda: ["L0", "L1", "L2", "L3"]
    )
    l0: L0Config = Field(default_factory=L0Config)
    l1: L1Config = Field(default_factory=L1Config)
    l2: L2Config = Field(default_factory=L2Config)
    l3: L3Config = Field(default_factory=L3Config)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


class ServiceConfig(BaseModel):
    redis_url: str = "redis://127.0.0.1:6379/0"
    pg_uri: str = Field(default_factory=lambda: os.getenv("PG_URI", ""))
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "memlab_docs"


class WorkloadConfig(BaseModel):
    corpus_path: str = "./data/docs.jsonl"
    tasks_path: str = "./data/tasks.jsonl"
    trace_path: str = "./traces/trace_200a.jsonl"


class MemlabConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    services: ServiceConfig = Field(default_factory=ServiceConfig)
    workloads: WorkloadConfig = Field(default_factory=WorkloadConfig)


def _merge_env_overrides(data: dict) -> dict:
    env_map = {
        "MEMLAB_OUTPUT_DIR": ("experiment", "output_dir"),
        "MEMLAB_LOGS_DIR": ("experiment", "logs_dir"),
        "REDIS_URL": ("services", "redis_url"),
        "PG_URI": ("services", "pg_uri"),
        "QDRANT_URL": ("services", "qdrant_url"),
        "QDRANT_COLLECTION": ("services", "qdrant_collection"),
    }
    for env_key, path in env_map.items():
        v = os.getenv(env_key)
        if not v:
            continue
        target = data
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = v
    return data


def load_config(path: str | Path | None = None) -> MemlabConfig:
    candidate = Path(path or os.getenv("MEMLAB_CONFIG_PATH", "configs/memlab.default.yaml"))
    if not candidate.exists():
        return MemlabConfig(**_merge_env_overrides({}))
    with candidate.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    merged = _merge_env_overrides(raw)
    return MemlabConfig(**merged)
