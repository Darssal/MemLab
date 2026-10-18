from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from memlab.memory.types import ContextBudget, MemoryItem


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class ContextPackResult:
    messages: list[dict[str, Any]]
    selected: list[MemoryItem]
    dropped_keys: list[str]
    injected_tokens: int
    injected_item_count: int
    total_tokens: int
    summarized: bool


def _summarize_messages(messages: list[dict[str, Any]], target_tokens: int) -> str:
    chunks: list[str] = []
    remain = max(1, target_tokens)
    for m in messages:
        line = f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        cost = estimate_tokens(line)
        if cost > remain:
            break
        chunks.append(line)
        remain -= cost
        if remain <= 0:
            break
    return "\n".join(chunks[:64]) if chunks else "No prior context."


def _near_duplicate_hash(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    # Limit hash input to keep dedupe cheap while still robust against prompt bloat.
    return hashlib.sha1(normalized[:512].encode("utf-8")).hexdigest()


def assemble_l0_context(
    *,
    recent_messages: list[MemoryItem],
    candidates: list[MemoryItem],
    user_query: str,
    budget: ContextBudget,
    summarization_on: bool,
    summarize_trigger_tokens: int,
    summarize_target_tokens: int,
    min_relevance_score: float,
) -> ContextPackResult:
    eligible_recent = [
        m
        for m in recent_messages
        if not bool((m.metadata or {}).get("exclude_from_recent_context", False))
    ]
    keep_n = min(len(eligible_recent), budget.keep_recent_messages)
    kept_recent = eligible_recent[-keep_n:] if keep_n > 0 else []
    kept_keys = {m.key for m in kept_recent}

    selected: list[MemoryItem] = []
    dropped: list[str] = []
    injected_tokens = 0
    seen_versions: set[tuple[str, str]] = set()
    seen_text_hashes: set[str] = set()
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item.metadata.get("retrieval_score", 0.0) if isinstance(item.metadata.get("retrieval_score"), (int, float)) else 0.0),
            str(item.key),
        ),
    )

    for item in ranked_candidates:
        if item.key in kept_keys:
            continue
        rel = item.metadata.get("retrieval_score")
        score = float(rel) if isinstance(rel, (int, float)) else 1.0
        if score < min_relevance_score:
            dropped.append(item.key)
            continue
        version_raw = item.metadata.get("version", item.created_at_ms)
        version = str(version_raw)
        dedupe_key = (item.key, version)
        if dedupe_key in seen_versions:
            dropped.append(item.key)
            continue
        seen_versions.add(dedupe_key)
        text_hash = _near_duplicate_hash(item.text)
        if text_hash in seen_text_hashes:
            dropped.append(item.key)
            continue
        seen_text_hashes.add(text_hash)
        cost = estimate_tokens(item.text)
        if injected_tokens + cost > budget.max_memory_injection_tokens:
            dropped.append(item.key)
            continue
        selected.append(item)
        injected_tokens += cost

    messages = [
        {
            "role": m.metadata.get("role", "user"),
            "content": m.text,
        }
        for m in kept_recent
    ]
    used = 0
    injected_item_count = 0
    for item in selected:
        cost = estimate_tokens(item.text)
        if used + cost > budget.max_memory_injection_tokens:
            break
        messages.append(
            {
                "role": "system",
                "content": f"[MEMORY:{item.kind}:{item.key}] {item.text}",
            }
        )
        used += cost
        injected_item_count += 1
    injected_tokens = used
    messages.append({"role": "user", "content": user_query})

    total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    summarized = False
    if summarization_on and total_tokens > summarize_trigger_tokens and len(messages) > budget.keep_recent_messages:
        older = messages[:-budget.keep_recent_messages]
        summary_text = _summarize_messages(older, summarize_target_tokens)
        messages = [{"role": "system", "content": f"[SUMMARY] {summary_text}"}] + messages[-budget.keep_recent_messages:]
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        summarized = True

    # Final fail-safe: enforce prompt upper bound by trimming memory injections from the front.
    while messages and total_tokens > budget.max_prompt_tokens and len(messages) > 1:
        # Keep the final user message.
        if messages[0].get("role") == "user" and len(messages) == 1:
            break
        messages.pop(0)
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)

    return ContextPackResult(
        messages=messages,
        selected=selected,
        dropped_keys=dropped,
        injected_tokens=injected_tokens,
        injected_item_count=injected_item_count,
        total_tokens=total_tokens,
        summarized=summarized,
    )
