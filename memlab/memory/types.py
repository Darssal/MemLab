from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


MemoryKind = Literal[
    "message",
    "tool_result",
    "fact",
    "summary",
    "checkpoint",
    "blob_ref",
]


@dataclass
class MemoryItem:
    key: str
    agent_id: str
    kind: MemoryKind
    text: str
    bytes: int
    importance: float
    created_at_ms: int
    last_access_ms: int
    access_count: int
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBudget:
    max_prompt_tokens: int
    max_memory_injection_tokens: int
    keep_recent_messages: int


@dataclass
class ContextPack:
    messages: list[dict[str, Any]]
    injected_memories: list[MemoryItem]
    trace: dict[str, Any]


@dataclass
class PlacementDecision:
    key: str
    tiers: list[str]
    ttl_s: int | None
    reasons: list[str]
    admission_reason: str = ""
    eviction_reason: str = ""
    packed_tokens: int | None = None
    pointer_fetch: int = 0
    degrade_mode: str | None = None
