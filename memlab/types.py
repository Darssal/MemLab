from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Tier = Literal["L0", "L1", "L2", "L3"]


@dataclass
class RunnerConfig:
    variant_id: str
    n_agents: int
    llm_backend: Literal["stub", "openai_compatible", "deerflow"]
    seed: int
    duration_s: int
    redis_policy: Literal["allkeys-lfu", "allkeys-lru"]
    summarization_on: bool
    repeat_idx: int


@dataclass
class EventLog:
    run_id: str
    ts_ms: int
    agent_id: str
    task_id: str
    op: str
    tier: str
    lat_ms: float
    hit: bool | None
    bytes: int | None
    tokens_in: int | None
    tokens_out: int | None
    error: str | None
    extra: dict[str, Any] | None = None
