from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from memlab.runtime_paths import deerflow_root
from memlab.session_logging import current_agent_id, make_operation_logger

from memlab.envcheck import run_env_checks, write_manifest
from memlab.io import json_dump, now_ms
from memlab.memory.types import ContextBudget, MemoryItem
from memlab.memory_manager.manager import MultiLevelMemoryManager
from memlab.metrics.eventlog import EventLogWriter
from memlab.metrics.manifest import build_manifest
from memlab.runner_core import (
    compute_run_summary,
    load_docs,
    load_e2e_tasks,
    load_runtime_config,
    parse_common_args,
    parse_model_answer,
    score_task_answer,
    shuffle_tasks,
)


SYSTEM_PROMPT_SECRET = (
    "You are one of many parallel agent sessions. "
    "Return STRICT JSON only: {\"task_id\":\"...\",\"secret_code\":\"...\"}."
)
SYSTEM_PROMPT_QA = (
    "You are one of many parallel agent sessions. "
    "Return STRICT JSON only: {\"task_id\":\"...\",\"final_answer\":\"...\"}."
)
LOCKED_BASE_URL = ""
LOCKED_MODEL_ID = ""
LOCKED_DEERFLOW_MODEL_NAME = "default-openai-compatible"
DEFAULT_RATE_LIMIT_SAFE_MAX_INFLIGHT = 2
PRECHECK_CHAT_TIMEOUT_S = 45.0
PRECHECK_STREAM_TIMEOUT_S = 120.0
EMBEDDED_WORKER_JSON_PREFIX = "__MEMLAB_JSON__:"

_EMBEDDED_WORKER_PROC: asyncio.subprocess.Process | None = None
_EMBEDDED_WORKER_LOCK: asyncio.Lock | None = None
_EMBEDDED_WORKER_ERR_HANDLE: Any = None


class ProviderBalanceError(RuntimeError):
    pass


def _safe_error_detail(exc: Exception) -> str:
    """Avoid persisting remote bodies, request URLs or exception payloads."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_clip(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars]


def _rate_limit_safe_max_inflight() -> int:
    raw = str(os.getenv("MEMLAB_E2E_RATE_LIMIT_SAFE_MAX_INFLIGHT", "") or "").strip()
    if not raw:
        return DEFAULT_RATE_LIMIT_SAFE_MAX_INFLIGHT
    try:
        value = int(raw)
    except Exception:
        return DEFAULT_RATE_LIMIT_SAFE_MAX_INFLIGHT
    return max(1, value)


def _error_category(msg: str) -> str:
    lower = str(msg or "").lower()
    if not lower:
        return "none"
    if "30001" in lower or ("insufficient" in lower and "balance" in lower):
        return "provider_403_balance"
    if "429" in lower or "ratelimit" in lower or "tpm" in lower:
        return "provider_429_rate_limit"
    if "max_seq_len" in lower or "max_prompt_tokens" in lower or "context_length" in lower:
        return "provider_400_context_length"
    if "input_tokens" in lower and ("exceed" in lower or "larger" in lower):
        return "provider_400_context_length"
    if "unable to use tool calls" in lower or ("tool calls" in lower and "unable" in lower):
        return "tool_call_unsupported"
    if "without ai content" in lower:
        return "deerflow_no_ai_content"
    if "/runs/stream" in lower and "404" in lower:
        return "deerflow_stream_404"
    if "model not found" in lower or "no chat model could be resolved" in lower:
        return "model_not_found"
    if "403" in lower:
        return "provider_403_other"
    if "400" in lower:
        return "provider_400_other"
    return "other"


def _is_rate_limit_error(msg: str) -> bool:
    return _error_category(msg) == "provider_429_rate_limit"


def _is_transient_provider_error(msg: str) -> bool:
    lower = str(msg or "").lower()
    markers = (
        "readtimeout",
        "read timeout",
        "connecttimeout",
        "connection reset",
        "server disconnected without sending a response",
        "temporarily unavailable",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "code\":50500",
        "unknown error",
    )
    return any(marker in lower for marker in markers)


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_error_breakdown(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    error_examples: dict[str, str] = {}
    total_errors = 0
    for row in task_rows:
        if str(row.get("status", "")) != "error":
            continue
        total_errors += 1
        err = str(row.get("error", "") or "")
        cat = _error_category(err)
        by_category[cat] = int(by_category.get(cat, 0)) + 1
        if cat not in error_examples and err:
            error_examples[cat] = err[:240]
    return {
        "total_errors": int(total_errors),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))),
        "examples": error_examples,
    }


def _dedupe_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = " ".join(str(line).strip().lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(line))
    return out


def _question_focus_text(question: str) -> str:
    raw = str(question or "").strip()
    if not raw:
        return ""
    for marker in ("\n\nOptions:", "\nOptions:\n"):
        if marker in raw:
            raw = raw.split(marker, 1)[0].strip()
            break
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("return only "):
            continue
        lines.append(s)
    return " ".join(lines).strip()


def _head_tail_clip(text: str, max_chars: int) -> str:
    body = str(text or "")
    if max_chars <= 0:
        return ""
    if len(body) <= max_chars:
        return body
    if max_chars <= 240:
        return body[:max_chars]
    ellipsis = "\n...\n"
    head = max(80, int(max_chars * 0.38))
    tail = max(80, max_chars - head - len(ellipsis))
    return f"{body[:head].strip()}{ellipsis}{body[-tail:].strip()}"


def _pack_segments(segments: list[str], *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    out: list[str] = []
    used = 0
    for seg in segments:
        piece = str(seg or "").strip()
        if not piece:
            continue
        prefix = "\n---\n" if out else ""
        remain = max_chars - used - len(prefix)
        if remain <= 0:
            break
        if len(piece) > remain:
            if remain < 120:
                break
            piece = _head_tail_clip(piece, remain)
        out.append(prefix + piece)
        used += len(prefix) + len(piece)
    return "".join(out)


def _prepare_context_lines(
    packed_messages: list[dict[str, Any]],
    *,
    top_k: int,
    dedupe: bool,
    item_max_chars: int,
) -> list[str]:
    lines = [f"{m.get('role', 'system')}: {_head_tail_clip(str(m.get('content', '')), int(item_max_chars))}" for m in packed_messages]
    if dedupe:
        lines = _dedupe_lines(lines)
    return lines[: max(1, int(top_k))]


def _build_retrieval_shards(
    text: str,
    *,
    shard_chars: int,
    max_shards: int,
) -> list[dict[str, Any]]:
    body = str(text or "")
    if not body:
        return []
    shard_chars = max(400, int(shard_chars))
    max_shards = max(1, int(max_shards))
    structured_units = [blk.strip() for blk in re.split(r"\n\s*\n+", body) if blk.strip()]
    if len(structured_units) >= 6:
        shards: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        cursor = 0
        current_blocks: list[str] = []
        current_start = 0
        current_chars = 0
        for block in structured_units:
            block_pos = body.find(block, cursor)
            if block_pos < 0:
                block_pos = cursor
            if not current_blocks:
                current_start = block_pos
            block_len = len(block) + 2
            if current_blocks and current_chars + block_len > shard_chars:
                shard_text = "\n\n".join(current_blocks).strip()
                if shard_text:
                    dedupe_key = " ".join(shard_text[:512].split())
                    if dedupe_key not in seen_texts:
                        seen_texts.add(dedupe_key)
                        shards.append(
                            {
                                "text": shard_text,
                                "start": int(current_start),
                                "end": int(current_start + len(shard_text)),
                            }
                        )
                current_blocks = [block]
                current_start = block_pos
                current_chars = block_len
            else:
                current_blocks.append(block)
                current_chars += block_len
            cursor = block_pos + len(block)
        if current_blocks:
            shard_text = "\n\n".join(current_blocks).strip()
            if shard_text:
                dedupe_key = " ".join(shard_text[:512].split())
                if dedupe_key not in seen_texts:
                    seen_texts.add(dedupe_key)
                    shards.append(
                        {
                            "text": shard_text,
                            "start": int(current_start),
                            "end": int(current_start + len(shard_text)),
                        }
                    )
        if shards:
            if len(shards) <= max_shards:
                return shards
            if max_shards == 1:
                return [shards[0]]
            last_idx = len(shards) - 1
            sampled_idx = {
                0,
                last_idx,
                *(int(round(last_idx * i / max(1, max_shards - 1))) for i in range(max_shards)),
            }
            return [shards[i] for i in sorted(sampled_idx)]
    if len(body) <= shard_chars:
        return [{"text": body, "start": 0, "end": len(body)}]
    # Use low-overlap sequential chunks first to avoid shard amplification noise.
    starts = list(range(0, len(body), shard_chars))
    if len(starts) > max_shards:
        if max_shards == 1:
            starts = [0]
        else:
            last_idx = len(starts) - 1
            sampled_idx = {
                0,
                last_idx,
                *(int(round(last_idx * i / max(1, max_shards - 1))) for i in range(max_shards)),
            }
            starts = [starts[i] for i in sorted(sampled_idx)]

    shards: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for start in starts:
        end = min(len(body), start + shard_chars)
        shard_text = body[start:end].strip()
        if not shard_text:
            continue
        dedupe_key = " ".join(shard_text[:512].split())
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        shards.append({"text": shard_text, "start": int(start), "end": int(end)})
    return shards


def _build_question_anchors(
    body: str,
    question: str,
    *,
    max_chunks: int = 4,
    chunk_chars: int = 420,
) -> list[str]:
    text = str(body or "")
    q = str(question or "").lower()
    if not text or not q:
        return []
    recency_sensitive = bool(
        re.search(r"^\s*(where|when)\s+(is|was|are|were)\b", q)
        or re.search(r"\b(current|currently|latest|last|now)\b", q)
    )
    stop = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "which",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "why",
        "how",
        "your",
        "into",
        "about",
        "return",
        "only",
        "json",
        "answer",
    }
    q_tokens = [t for t in re.findall(r"[a-z0-9_]{3,}", q) if t not in stop]
    if not q_tokens:
        q_tokens = re.findall(r"[a-z0-9_]{3,}", q)

    def _clip_around_match(seg: str) -> str:
        low = seg.lower()
        hits: list[int] = []
        for tok in q_tokens:
            start = 0
            while True:
                idx = low.find(tok, start)
                if idx < 0:
                    break
                hits.append(idx)
                start = idx + max(1, len(tok))
        if not hits:
            return _safe_clip(seg, max(120, int(chunk_chars)))
        anchor_pos = max(hits) if recency_sensitive else min(hits)
        width = max(120, int(chunk_chars))
        left = max(0, anchor_pos - int(width * 0.35))
        right = min(len(seg), left + width)
        left = max(0, right - width)
        return seg[left:right].strip()

    structured_blocks = [blk.strip() for blk in re.split(r"\n\s*\n+", text) if blk.strip()]
    segments: list[str] = []
    if len(structured_blocks) >= 6:
        segments = structured_blocks
    else:
        for seg in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
            s = seg.strip()
            if s:
                segments.append(s)
    if not segments:
        segments = [text]

    scored: list[tuple[float, int, str]] = []
    seg_count = max(1, len(segments))
    for seg_idx, seg in enumerate(segments):
        low = seg.lower()
        overlap = sum(1 for tok in q_tokens if tok in low)
        if overlap <= 0:
            continue
        token_count = len(re.findall(r"[a-z0-9_]{3,}", low))
        score = float(overlap) + min(1.0, token_count / 18.0) * 0.2
        if len(low) < 64:
            score -= 0.4
        if recency_sensitive:
            score += 0.35 * ((seg_idx + 1) / seg_count)
        scored.append((score, seg_idx, seg))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _, seg_idx, seg in scored:
        chunk = _clip_around_match(seg)
        key = " ".join(chunk.lower().split())
        if key in seen:
            continue
        seen.add(key)
        selected.append((seg_idx, chunk))
        if len(selected) >= max(1, int(max_chunks)):
            break

    if selected:
        if recency_sensitive:
            selected.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in selected]

    # Fallback: take evenly spaced windows when lexical overlap is weak.
    if len(text) <= chunk_chars:
        return [text]
    starts = list(range(0, len(text), chunk_chars))
    picks: list[int] = []
    if starts:
        picks.append(0)
    if len(starts) > 1:
        picks.append(len(starts) // 2)
    if len(starts) > 2:
        picks.append(len(starts) - 1)
    out: list[str] = []
    for idx in picks:
        chunk = text[starts[idx] : min(len(text), starts[idx] + chunk_chars)].strip()
        if chunk:
            out.append(chunk)
    return out[: max(1, int(max_chunks))]


def _build_qa_evidence_excerpt(
    *,
    body: str,
    question: str,
    max_chars: int,
) -> str:
    text = str(body or "")
    max_chars = max(240, int(max_chars))
    if len(text) <= max_chars:
        return text

    focus = _question_focus_text(question)
    anchors = _build_question_anchors(
        body=text,
        question=focus or question,
        max_chunks=min(6, max(2, max_chars // 450)),
        chunk_chars=max(220, min(900, max_chars // 3)),
    )
    tail_chars = max(200, min(max_chars // 3, 1800))
    head_chars = max(160, min(max_chars // 5, 900))
    segments: list[str] = []
    if focus:
        segments.append(f"[QUESTION] {focus}")
    for idx, anchor in enumerate(anchors[:6], start=1):
        segments.append(f"[EVIDENCE_{idx}] {anchor}")
    if anchors:
        segments.append(f"[TAIL] {text[-tail_chars:].strip()}")
    else:
        segments.append(f"[HEAD] {text[:head_chars].strip()}")
        segments.append(f"[TAIL] {text[-tail_chars:].strip()}")
    packed = _pack_segments(segments, max_chars=max_chars)
    return packed if packed else _head_tail_clip(text, max_chars)


def _qa_reasoning_hint(*, question: str, qa_prompt_profile: str) -> str:
    q = str(question or "").strip().lower()
    hints: list[str] = []
    if re.search(r"^\s*(where|when)\s+(is|was|are|were)\b", q) or re.search(r"\b(current|currently|latest|last|now)\b", q):
        hints.append("If the evidence contains multiple updates for the same entity or state, choose the latest directly supported state or value.")
    if re.search(r"\b(how much more|difference|more expensive|cheaper|compared to|than)\b", q):
        hints.append("Compute the requested difference or comparison from the cited values; do not return an intermediate value.")
    if re.search(r"^\s*how many\b", q) or re.search(r"\bcount\b", q):
        hints.append("Count all directly supported matching items or events before answering.")
    if re.search(r"\b(order|first to last|chronological|before|after)\b", q):
        hints.append("If multiple events are mentioned, order them chronologically from earliest to latest before answering.")
    if re.search(r"\b(recommend|suggest|preference|prefer|would like)\b", q):
        hints.append("Base the answer on the user's documented preferences and prior experiences, not on generic suggestions.")
    profile = str(qa_prompt_profile or "").strip().lower()
    if profile == "entity_span":
        hints.append("Return only the bare entity, location, number, or short value without articles or explanation.")
    elif profile == "multiple_choice_letter":
        hints.append("Reason over the evidence first, then return only one supported option letter.")
    elif profile == "summary":
        hints.append("Return a concise preference-aware or synthesis answer that preserves the key constraints from the evidence.")
    return " ".join(hints).strip()


def _resolve_task_qa_prompt_profile(*, task: dict[str, Any], default_profile: str) -> str:
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        for key in ("qa_prompt_profile", "qa_prompt_profile_override"):
            override = str(metadata.get(key, "")).strip().lower()
            if override in {"default", "yes_no", "entity_span", "short_answer", "summary", "multiple_choice_letter"}:
                return override
    profile = str(default_profile or "default").strip().lower()
    if profile in {"default", "yes_no", "entity_span", "short_answer", "summary", "multiple_choice_letter"}:
        return profile
    return "default"


def _summarize_prompt(
    *,
    session_id: str,
    question: str,
    context_block: str,
    body: str,
    task_type: str,
    answer_field: str,
    answer_format_mode: str,
    prompt_mode: str,
    qa_prompt_profile: str,
    qa_support_mode: str,
    max_prompt_tokens: int,
    max_evidence_chars: int,
) -> tuple[str, int]:
    char_budget = max(2500, int(max_prompt_tokens) * 4)
    if prompt_mode == "memory_neutral":
        max_context_chars = min(1200, max(400, int(char_budget * 0.08)))
        evidence_budget = max(0, min(int(max_evidence_chars), int(char_budget * 0.84)))
    else:
        max_context_chars = max(1000, int(char_budget * 0.45))
        evidence_budget = max(0, min(int(max_evidence_chars), int(char_budget * 0.40)))
    context_clip = _safe_clip(context_block, max_context_chars)
    if task_type == "secret_code":
        evidence_clip = _head_tail_clip(body, evidence_budget)
    else:
        evidence_clip = _build_qa_evidence_excerpt(body=body, question=question, max_chars=evidence_budget)
    if prompt_mode == "memory_beneficial":
        if task_type == "secret_code":
            evidence_block = (
                "[omitted by prompt_mode=memory_beneficial]\n"
                "Use memory retrieval context above to produce the final answer."
            )
            evidence_chars_used = 0
        elif str(qa_support_mode or "standard").strip().lower() == "none":
            evidence_block = (
                "[direct evidence omitted by qa_support_mode=none]\n"
                "Use the memory retrieval context above as the only evidence source. "
                "If that context is insufficient, answer conservatively from retrieved memory only."
            )
            evidence_chars_used = 0
        else:
            support_budget = max(480, min(len(evidence_clip), max(1400, evidence_budget // 2)))
            evidence_block = (
                "[supportive direct evidence excerpt]\n"
                f"{_safe_clip(evidence_clip, support_budget)}\n"
                "Use the memory retrieval context above as the primary recall path, and use this excerpt only as supporting evidence."
            )
            evidence_chars_used = min(len(evidence_clip), support_budget)
    else:
        evidence_block = evidence_clip
        evidence_chars_used = len(evidence_clip)
    if task_type == "secret_code":
        system_prompt = SYSTEM_PROMPT_SECRET
        task_instruction = "Return the secret code for this task."
        expected_json = '{"task_id":"...","secret_code":"..."}'
    else:
        system_prompt = SYSTEM_PROMPT_QA
        task_instruction = _qa_task_instruction(
            qa_prompt_profile=qa_prompt_profile,
            answer_field=answer_field,
        )
        expected_json = '{"task_id":"...","final_answer":"..."}'
    reasoning_hint = "" if task_type == "secret_code" else _qa_reasoning_hint(
        question=question,
        qa_prompt_profile=qa_prompt_profile,
    )
    if answer_format_mode == "final_answer_prefix":
        answer_format_note = 'If JSON fails, output one line: "Final Answer: <value>"'
    else:
        answer_format_note = f"Do not output prose outside JSON. Required schema: {expected_json}"
    prompt = (
        f"{system_prompt}\n\n"
        f"Session: {session_id}\n"
        f"Prompt mode: {prompt_mode}\n"
        f"Task type: {task_type}\n"
        f"QA prompt profile: {qa_prompt_profile}\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_clip}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Instruction: {task_instruction}\n"
        f"{('Reasoning hint: ' + reasoning_hint + chr(10)) if reasoning_hint else ''}"
        f"Answer format: {answer_format_note}"
    )
    if len(prompt) > char_budget:
        prompt = _safe_clip(prompt, char_budget)
    return prompt, evidence_chars_used


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


def _is_ai_message(msg: dict[str, Any]) -> bool:
    mtype = str(msg.get("type", "")).lower()
    role = str(msg.get("role", "")).lower()
    return mtype == "ai" or role in {"assistant", "ai"}


def _extract_usage_from_usage_obj(obj: Any) -> tuple[int, int]:
    if not isinstance(obj, dict):
        return 0, 0
    prompt_tokens = 0
    completion_tokens = 0
    for key, value in obj.items():
        if not isinstance(value, (int, float)):
            continue
        lk = str(key).lower()
        if lk in {"prompt_tokens", "input_tokens"}:
            prompt_tokens = max(prompt_tokens, int(value))
        elif lk in {"completion_tokens", "output_tokens"}:
            completion_tokens = max(completion_tokens, int(value))
    return prompt_tokens, completion_tokens


def _extract_usage_from_ai_message(msg: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    candidates: list[Any] = [msg]
    usage_metadata = msg.get("usage_metadata")
    if usage_metadata is not None:
        candidates.append(usage_metadata)
    response_metadata = msg.get("response_metadata")
    if isinstance(response_metadata, dict):
        candidates.append(response_metadata)
        token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            candidates.append(token_usage)
        nested_usage = response_metadata.get("usage")
        if nested_usage is not None:
            candidates.append(nested_usage)
    usage = msg.get("usage")
    if usage is not None:
        candidates.append(usage)
    for obj in candidates:
        p, c = _extract_usage_from_usage_obj(obj)
        prompt_tokens = max(prompt_tokens, p)
        completion_tokens = max(completion_tokens, c)
    return prompt_tokens, completion_tokens


def _extract_ai_message_from_values(data: dict[str, Any]) -> str:
    msgs = data.get("messages", [])
    if not isinstance(msgs, list):
        return ""
    for msg in reversed(msgs):
        if isinstance(msg, dict) and _is_ai_message(msg):
            return _extract_text(msg.get("content", ""))
    return ""


def _extract_usage_from_values(data: dict[str, Any]) -> tuple[int, int]:
    msgs = data.get("messages", [])
    if not isinstance(msgs, list):
        return 0, 0
    prompt_tokens = 0
    completion_tokens = 0
    for msg in reversed(msgs):
        if isinstance(msg, dict) and _is_ai_message(msg):
            p, c = _extract_usage_from_ai_message(msg)
            prompt_tokens = max(prompt_tokens, p)
            completion_tokens = max(completion_tokens, c)
    return prompt_tokens, completion_tokens


def _is_local_loopback_url(url: str) -> bool:
    host = str(urlparse(str(url or "")).hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


def _should_use_embedded_deerflow(*, langgraph_url: str, experiment_track: str) -> bool:
    return str(experiment_track or "").strip() in {
        "candidate_race",
        "emergency_rebaseline",
        "memory_exposure",
        "focused_validation_v3",
        "alignment_diagnosis",
    } and _is_local_loopback_url(langgraph_url)


def _qa_task_instruction(*, qa_prompt_profile: str, answer_field: str) -> str:
    profile = str(qa_prompt_profile or "default").strip().lower()
    if profile == "multiple_choice_letter":
        return (
            "Return only the single correct option letter (for example `A`, `B`, `C`, or `D`) "
            f"in `{answer_field}`."
        )
    if profile == "yes_no":
        return (
            "Answer with only `yes` or `no` when the question is binary. "
            f"If the question is not binary, return the shortest exact supported answer in `{answer_field}`."
        )
    if profile == "entity_span":
        return (
            "Extract only the minimal entity name, short span, or value that directly answers the question, "
            f"and put only that text in `{answer_field}`."
        )
    if profile == "short_answer":
        return (
            "Return the shortest exact answer span or value supported by the evidence. "
            f"Do not explain or restate the question; put only the answer in `{answer_field}`."
        )
    if profile == "summary":
        return (
            "Return a concise synthesis answer that preserves the user's relevant preferences, constraints, or chronology. "
            f"When the question is simple, still keep the answer compact and put only the answer in `{answer_field}`."
        )
    return (
        "Answer the question with the shortest exact answer span or value supported by the evidence, "
        f"and put only that value in `{answer_field}`."
    )

def _make_embedded_thread_id(seed: str) -> str:
    raw = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(seed or ""))
    collapsed = "-".join(part for part in raw.split("-") if part)
    if not collapsed:
        collapsed = f"memlab-{now_ms()}"
    return collapsed[:96]

def _embedded_worker_lock() -> asyncio.Lock:
    global _EMBEDDED_WORKER_LOCK
    if _EMBEDDED_WORKER_LOCK is None:
        _EMBEDDED_WORKER_LOCK = asyncio.Lock()
    return _EMBEDDED_WORKER_LOCK


def _embedded_worker_paths() -> tuple[Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    backend_root = deerflow_root() / "backend"
    python_exe = backend_root / ".venv" / "Scripts" / "python.exe"
    worker_script = repo_root / "scripts" / "deerflow_embedded_worker.py"
    config_path = deerflow_root() / "config.yaml"
    deerflow_home = backend_root / ".deer-flow"
    return python_exe, worker_script, config_path, deerflow_home


async def _shutdown_embedded_worker(force: bool = False) -> None:
    global _EMBEDDED_WORKER_PROC, _EMBEDDED_WORKER_ERR_HANDLE
    proc = _EMBEDDED_WORKER_PROC
    _EMBEDDED_WORKER_PROC = None
    if proc is not None:
        try:
            if not force and proc.returncode is None and proc.stdin is not None and proc.stdout is not None:
                try:
                    proc.stdin.write(b'{"type":"shutdown"}\n')
                    await proc.stdin.drain()
                    _ = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
                except Exception:
                    pass
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    proc.kill()
                    await proc.wait()
        except (ProcessLookupError, AttributeError):
            pass
    if _EMBEDDED_WORKER_ERR_HANDLE is not None:
        try:
            _EMBEDDED_WORKER_ERR_HANDLE.close()
        except Exception:
            pass
        _EMBEDDED_WORKER_ERR_HANDLE = None


def _shutdown_embedded_worker_sync() -> None:
    global _EMBEDDED_WORKER_PROC, _EMBEDDED_WORKER_ERR_HANDLE
    proc = _EMBEDDED_WORKER_PROC
    _EMBEDDED_WORKER_PROC = None
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if _EMBEDDED_WORKER_ERR_HANDLE is not None:
        try:
            _EMBEDDED_WORKER_ERR_HANDLE.close()
        except Exception:
            pass
        _EMBEDDED_WORKER_ERR_HANDLE = None


async def _ensure_embedded_worker_started() -> asyncio.subprocess.Process:
    global _EMBEDDED_WORKER_PROC, _EMBEDDED_WORKER_ERR_HANDLE
    proc = _EMBEDDED_WORKER_PROC
    if proc is not None and proc.returncode is None and proc.stdin is not None and proc.stdout is not None:
        return proc

    python_exe, worker_script, config_path, deerflow_home = _embedded_worker_paths()
    if not python_exe.exists():
        raise FileNotFoundError(f"Embedded DeerFlow Python not found at {python_exe}")
    if not worker_script.exists():
        raise FileNotFoundError(f"Embedded DeerFlow worker script not found at {worker_script}")
    env = os.environ.copy()
    env["DEER_FLOW_CONFIG_PATH"] = str(config_path)
    env["DEER_FLOW_HOME"] = str(deerflow_home)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log_dir = Path("runs/runtime")
    log_dir.mkdir(parents=True, exist_ok=True)
    if _EMBEDDED_WORKER_ERR_HANDLE is not None:
        try:
            _EMBEDDED_WORKER_ERR_HANDLE.close()
        except Exception:
            pass
    _EMBEDDED_WORKER_ERR_HANDLE = open(log_dir / "embedded_worker.err.log", "a", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        str(python_exe),
        "-u",
        str(worker_script),
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=_EMBEDDED_WORKER_ERR_HANDLE,
    )
    if proc.stdout is None:
        await _shutdown_embedded_worker(force=True)
        raise RuntimeError("Embedded DeerFlow worker stdout pipe is unavailable.")
    try:
        deadline = time.perf_counter() + 20.0
        ready_payload: dict[str, Any] | None = None
        while time.perf_counter() < deadline:
            remaining = max(0.5, deadline - time.perf_counter())
            ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            if not ready_line:
                break
            ready_text = ready_line.decode("utf-8", errors="ignore").strip()
            if not ready_text.startswith(EMBEDDED_WORKER_JSON_PREFIX):
                continue
            ready_payload = json.loads(ready_text[len(EMBEDDED_WORKER_JSON_PREFIX) :])
            break
    except Exception:
        await _shutdown_embedded_worker(force=True)
        raise
    if ready_payload is None:
        await _shutdown_embedded_worker(force=True)
        raise RuntimeError("Embedded DeerFlow worker exited before signaling readiness.")
    if not bool(ready_payload.get("ready", False)):
        await _shutdown_embedded_worker(force=True)
        raise RuntimeError(f"Embedded DeerFlow worker failed to start: {ready_payload}")
    _EMBEDDED_WORKER_PROC = proc
    return proc


async def _run_deerflow_prompt_embedded_worker(
    *,
    thread_id: str,
    prompt: str,
    model_override: str | None,
    agent_name: str | None,
    request_timeout_s: float,
) -> dict[str, Any]:
    lock = _embedded_worker_lock()
    for attempt in range(1, 3):
        proc = await _ensure_embedded_worker_started()
        if proc.stdin is None or proc.stdout is None:
            await _shutdown_embedded_worker(force=True)
            raise RuntimeError("Embedded DeerFlow worker pipes are unavailable.")
        async with lock:
            try:
                payload = {
                    "type": "run",
                    "thread_id": thread_id,
                    "prompt": prompt,
                    "model_override": model_override or "",
                    "agent_name": agent_name or "",
                }
                proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                await proc.stdin.drain()
                deadline = time.perf_counter() + float(max(5.0, request_timeout_s + 5.0))
                data: dict[str, Any] | None = None
                while time.perf_counter() < deadline:
                    remaining = max(0.5, deadline - time.perf_counter())
                    raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                    if not raw_line:
                        raise RuntimeError("Embedded DeerFlow worker closed stdout unexpectedly.")
                    text = raw_line.decode("utf-8", errors="ignore").strip()
                    if not text.startswith(EMBEDDED_WORKER_JSON_PREFIX):
                        continue
                    data = json.loads(text[len(EMBEDDED_WORKER_JSON_PREFIX) :])
                    break
                if data is None:
                    raise RuntimeError("Embedded DeerFlow worker produced no structured response before timeout.")
                if not bool(data.get("ok", False)):
                    raise RuntimeError(str(data.get("error", "embedded worker request failed")))
                return {
                    "content": str(data.get("content", "") or "{}"),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "usage_available": False,
                    "request_latency_ms": float(data.get("request_latency_ms", 0.0)),
                    "llm_queue_wait_ms": 0.0,
                    "model_name": model_override or "",
                }
            except Exception:
                await _shutdown_embedded_worker(force=True)
                if attempt >= 2:
                    raise
    raise RuntimeError("Embedded DeerFlow worker exhausted retries.")


async def _create_thread(client: httpx.AsyncClient, langgraph_url: str, *, session_id: str, agent_id: str) -> str:
    payload = {"metadata": {"session_id": session_id, "agent_id": agent_id}}
    r = await client.post(f"{langgraph_url.rstrip('/')}/threads", json=payload)
    r.raise_for_status()
    body = r.json()
    tid = body.get("thread_id")
    if not isinstance(tid, str) or not tid:
        raise RuntimeError("DeerFlow thread creation returned invalid thread_id")
    return tid


async def _create_deerflow_thread(
    *,
    client: httpx.AsyncClient | None,
    langgraph_url: str,
    session_id: str,
    agent_id: str,
    use_embedded_local: bool,
) -> str:
    if use_embedded_local:
        return _make_embedded_thread_id(f"{session_id}-{agent_id}")
    if client is None:
        raise RuntimeError("HTTP client is required for LangGraph DeerFlow thread creation.")
    return await _create_thread(client, langgraph_url, session_id=session_id, agent_id=agent_id)


async def _run_deerflow_prompt(
    *,
    client: httpx.AsyncClient | None,
    langgraph_url: str,
    thread_id: str,
    prompt: str,
    llm_sema: asyncio.Semaphore,
    model_override: str | None = None,
    assistant_id: str = "lead_agent",
    agent_name: str | None = None,
    use_embedded_local: bool = False,
    request_timeout_s: float = 180.0,
) -> dict[str, Any]:
    req_t0 = 0.0
    sem_wait_t0 = time.perf_counter()
    await llm_sema.acquire()
    llm_queue_wait_ms = (time.perf_counter() - sem_wait_t0) * 1000.0
    try:
        req_t0 = time.perf_counter()
        if use_embedded_local:
            result = await _run_deerflow_prompt_embedded_worker(
                thread_id=thread_id,
                prompt=prompt,
                model_override=model_override,
                agent_name=agent_name,
                request_timeout_s=float(max(1.0, request_timeout_s)),
            )
            result["llm_queue_wait_ms"] = llm_queue_wait_ms
            return result

        if client is None:
            raise RuntimeError("HTTP client is required for LangGraph DeerFlow prompt execution.")

        configurable: dict[str, Any] = {
            "thread_id": thread_id,
            # DeepSeek-V3.2 with DeerFlow may stream long empty reasoning chunks when
            # thinking mode is left on. Force deterministic benchmark mode.
            "thinking_enabled": False,
            "is_plan_mode": False,
            "subagent_enabled": False,
            "max_concurrent_subagents": 1,
        }
        if model_override:
            configurable["model_name"] = model_override
            configurable["model"] = model_override
        if agent_name:
            configurable["agent_name"] = agent_name
        payload = {
            "assistant_id": assistant_id,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            "config": {
                "configurable": configurable
            },
            "stream_mode": ["values", "messages-tuple"],
        }
        ai_chunks: list[str] = []
        final_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with client.stream(
                "POST",
                f"{langgraph_url.rstrip('/')}/threads/{thread_id}/runs/stream",
                json=payload,
                timeout=float(max(1.0, request_timeout_s)),
            ) as resp:
                resp.raise_for_status()
                event_name = ""
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        continue
                    if event_name == "values" and isinstance(data, dict):
                        p, c = _extract_usage_from_values(data)
                        prompt_tokens = max(prompt_tokens, p)
                        completion_tokens = max(completion_tokens, c)
                        maybe = _extract_ai_message_from_values(data)
                        if maybe:
                            final_text = maybe
                    elif event_name in {"messages", "messages-tuple"}:
                        node = data
                        if isinstance(node, list):
                            if node and isinstance(node[0], dict):
                                node = node[0]
                            else:
                                continue
                        if isinstance(node, dict):
                            if "message" in node and isinstance(node["message"], dict):
                                node = node["message"]
                            if _is_ai_message(node):
                                p, c = _extract_usage_from_ai_message(node)
                                prompt_tokens = max(prompt_tokens, p)
                                completion_tokens = max(completion_tokens, c)
                                txt = _extract_text(node.get("content", ""))
                                if txt:
                                    ai_chunks.append(txt)
                    elif event_name == "error":
                        msg = json.dumps(data, ensure_ascii=False)
                        lower = msg.lower()
                        if ("insufficient" in lower and "balance" in lower) or "30001" in lower:
                            raise ProviderBalanceError("provider_balance_rejected")
                        raise RuntimeError("deerflow_stream_rejected")
                    elif event_name == "end":
                        break
        except httpx.HTTPStatusError as exc:
            msg = str(exc)
            lower = msg.lower()
            if ("insufficient" in lower and "balance" in lower) or "30001" in lower or "403" in lower:
                raise ProviderBalanceError("provider_balance_rejected") from None
            raise RuntimeError(f"HTTP {exc.response.status_code}") from None
    finally:
        llm_sema.release()
    req_latency_ms = (time.perf_counter() - req_t0) * 1000.0

    if not final_text and ai_chunks:
        final_text = ai_chunks[-1]
    if not final_text:
        final_text = "\n".join(ai_chunks).strip()
    if not final_text:
        final_text = "{}"

    prompt_tokens_out = int(prompt_tokens) if prompt_tokens > 0 else None
    completion_tokens_out = int(completion_tokens) if completion_tokens > 0 else None
    return {
        "content": final_text,
        "prompt_tokens": prompt_tokens_out,
        "completion_tokens": completion_tokens_out,
        "usage_available": (prompt_tokens_out is not None and completion_tokens_out is not None),
        "request_latency_ms": req_latency_ms,
        "llm_queue_wait_ms": llm_queue_wait_ms,
        "model_name": model_override or "",
    }


async def _run_provider_prompt_direct(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    api_key: str,
    prompt: str,
    llm_sema: asyncio.Semaphore,
    request_timeout_s: float = 180.0,
) -> dict[str, Any]:
    req_t0 = 0.0
    sem_wait_t0 = time.perf_counter()
    await llm_sema.acquire()
    llm_queue_wait_ms = (time.perf_counter() - sem_wait_t0) * 1000.0
    try:
        req_t0 = time.perf_counter()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96,
            "temperature": 0.0,
            "stream": False,
        }
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=float(max(1.0, request_timeout_s)),
        )
        if resp.status_code != 200:
            text = resp.text
            lower = text.lower()
            if ("insufficient" in lower and "balance" in lower) or "30001" in lower or resp.status_code == 403:
                raise ProviderBalanceError(f"HTTP {resp.status_code}: provider_balance_rejected")
            raise RuntimeError(f"HTTP {resp.status_code}: provider_request_rejected")
        data = resp.json()
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Provider chat response contains no choices.")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        final_text = _extract_text(message.get("content", "")) if isinstance(message, dict) else ""
        if not final_text:
            final_text = "{}"
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    finally:
        llm_sema.release()
    req_latency_ms = (time.perf_counter() - req_t0) * 1000.0
    return {
        "content": final_text,
        "prompt_tokens": (prompt_tokens if prompt_tokens > 0 else None),
        "completion_tokens": (completion_tokens if completion_tokens > 0 else None),
        "usage_available": (prompt_tokens > 0 or completion_tokens > 0),
        "request_latency_ms": req_latency_ms,
        "llm_queue_wait_ms": llm_queue_wait_ms,
        "model_name": model_id,
    }


async def _provider_chat_preflight(
    *,
    base_url: str,
    model_id: str,
    api_key: str,
) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "preflight ping"}],
        "max_tokens": 16,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=PRECHECK_CHAT_TIMEOUT_S, trust_env=False) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        data = resp.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return False, "Response contains no choices."
        return True, "ok"
    except Exception as exc:
        detail = _safe_error_detail(exc)
        return False, detail


async def _provider_tool_call_preflight(
    *,
    base_url: str,
    model_id: str,
    api_key: str,
) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Call the ping tool now."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Returns pong.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "ping"}},
        "max_tokens": 32,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=PRECHECK_CHAT_TIMEOUT_S, trust_env=False) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        data = resp.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return False, "Response contains no choices."
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        if not isinstance(msg, dict):
            return False, "Invalid message format."
        tool_calls = msg.get("tool_calls")
        function_call = msg.get("function_call")
        if isinstance(tool_calls, list) and tool_calls:
            return True, "ok"
        if isinstance(function_call, dict) and function_call.get("name"):
            return True, "ok"
        return False, "No tool_calls/function_call found in response."
    except Exception as exc:
        detail = _safe_error_detail(exc)
        return False, detail


async def _deerflow_stream_preflight(
    *,
    langgraph_url: str,
    model_override: str,
    assistant_id: str,
    agent_name: str,
    use_embedded_local: bool,
) -> tuple[bool, str]:
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            if use_embedded_local:
                client = None
                thread_id = await _create_deerflow_thread(
                    client=None,
                    langgraph_url=langgraph_url,
                    session_id=f"preflight-{now_ms()}-{attempt}",
                    agent_id="agent_preflight",
                    use_embedded_local=True,
                )
                sema = asyncio.Semaphore(1)
                _ = await _run_deerflow_prompt(
                    client=client,
                    langgraph_url=langgraph_url,
                    thread_id=thread_id,
                    prompt='Return strict JSON only: {"ok": true}.',
                    llm_sema=sema,
                    model_override=model_override,
                    assistant_id=assistant_id,
                    agent_name=agent_name,
                    use_embedded_local=True,
                    request_timeout_s=PRECHECK_STREAM_TIMEOUT_S,
                )
            else:
                async with httpx.AsyncClient(timeout=PRECHECK_STREAM_TIMEOUT_S, trust_env=False) as client:
                    thread_id = await _create_deerflow_thread(
                        client=client,
                        langgraph_url=langgraph_url,
                        session_id=f"preflight-{now_ms()}-{attempt}",
                        agent_id="agent_preflight",
                        use_embedded_local=False,
                    )
                    sema = asyncio.Semaphore(1)
                    _ = await _run_deerflow_prompt(
                        client=client,
                        langgraph_url=langgraph_url,
                        thread_id=thread_id,
                        prompt='Return strict JSON only: {"ok": true}.',
                        llm_sema=sema,
                        model_override=model_override,
                        assistant_id=assistant_id,
                        agent_name=agent_name,
                        use_embedded_local=False,
                        request_timeout_s=PRECHECK_STREAM_TIMEOUT_S,
                    )
            return True, f"ok(attempt={attempt})"
        except Exception as exc:
            detail = _safe_error_detail(exc)
            errors.append(f"attempt={attempt}: {detail}")
            await asyncio.sleep(0.6 * attempt)
    return False, " | ".join(errors)


async def _run_preflight_checks(
    *,
    model_base_url: str,
    model_id: str,
    api_key: str,
    langgraph_url: str,
    locked_base_url: str,
    locked_model_id: str,
    locked_model_name: str,
    llm_max_inflight: int,
    prompt_mode: str,
    max_evidence_chars: int,
    comparable_mode: bool,
    assistant_id: str,
    agent_name: str,
    use_embedded_local: bool,
    use_provider_direct: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "started_at_utc": _utc_now_iso(),
        "checks": {},
        "warnings": [],
        "deerflow_transport": (
            "provider_direct" if use_provider_direct else ("embedded_local" if use_embedded_local else "langgraph_http")
        ),
    }
    checks: dict[str, Any] = {}
    report["checks"] = checks
    if use_embedded_local:
        report["warnings"].append("Using embedded local DeerFlow transport for this track.")
    if use_provider_direct:
        report["warnings"].append("Using provider_direct transport; DeerFlow stream preflight is skipped.")

    def _is_hard_provider_failure(detail: str) -> bool:
        txt = str(detail or "").lower()
        hard_markers = (
            "401",
            "unauthorized",
            "invalid api key",
            "model not found",
            "no chat model could be resolved",
            "unable to use tool calls",
        )
        return any(marker in txt for marker in hard_markers)

    def _is_timeout_like(detail: str) -> bool:
        txt = str(detail or "").lower()
        markers = (
            "timed out",
            "timeout",
            "readtimeout",
            "read timeout",
            "task was canceled",
            "task was cancelled",
            "a task was canceled",
            "a task was cancelled",
            "cancelled",
            "canceled",
        )
        return any(marker in txt for marker in markers)

    checks["env_vars"] = {
        "ok": bool(model_base_url and model_id and api_key),
        "base_url_present": bool(model_base_url),
        "model_present": bool(model_id),
        "api_key_present": bool(api_key),
    }
    checks["model_lock"] = {
        "ok": (not comparable_mode) or (model_base_url == locked_base_url and model_id == locked_model_id),
        "expected_base_url": locked_base_url,
        "expected_model_id": locked_model_id,
        "actual_base_url": model_base_url,
        "actual_model_id": model_id,
        "comparable_mode": bool(comparable_mode),
    }
    safe_max_inflight = _rate_limit_safe_max_inflight()
    checks["rate_limit_params"] = {
        "ok": int(llm_max_inflight) <= safe_max_inflight,
        "llm_max_inflight": int(llm_max_inflight),
        "recommended_max_inflight": safe_max_inflight,
    }
    checks["prompt_budget"] = {
        "ok": int(max_evidence_chars) >= 256,
        "prompt_mode": str(prompt_mode),
        "max_evidence_chars": int(max_evidence_chars),
    }

    if checks["env_vars"]["ok"]:
        (ok_chat, msg_chat), (ok_tool, msg_tool) = await asyncio.gather(
            _provider_chat_preflight(
                base_url=model_base_url,
                model_id=model_id,
                api_key=api_key,
            ),
            _provider_tool_call_preflight(
                base_url=model_base_url,
                model_id=model_id,
                api_key=api_key,
            ),
        )
        checks["provider_chat"] = {"ok": bool(ok_chat), "detail": msg_chat}
        checks["provider_tool_call"] = {"ok": bool(ok_tool), "detail": msg_tool}
    else:
        checks["provider_chat"] = {"ok": False, "detail": "Skipped due to missing env vars."}
        checks["provider_tool_call"] = {"ok": False, "detail": "Skipped due to missing env vars."}

    chat_detail = str(checks.get("provider_chat", {}).get("detail", ""))
    tool_detail = str(checks.get("provider_tool_call", {}).get("detail", ""))
    provider_checks_ok = bool(checks.get("provider_chat", {}).get("ok", False)) or bool(
        checks.get("provider_tool_call", {}).get("ok", False)
    )
    skip_deerflow_stream = False
    skip_reason = ""
    if checks["env_vars"]["ok"] and not provider_checks_ok:
        if _is_hard_provider_failure(chat_detail) or _is_hard_provider_failure(tool_detail):
            skip_deerflow_stream = True
            skip_reason = "Skipped because provider preflight hard-failed for the locked model."
        elif _is_timeout_like(chat_detail) and _is_timeout_like(tool_detail):
            skip_deerflow_stream = True
            skip_reason = "Skipped because provider preflight timed out for the locked model."

    if use_provider_direct:
        checks["deerflow_stream"] = {
            "ok": True,
            "detail": "Skipped because provider_direct transport is enabled.",
            "skipped": True,
        }
    elif skip_deerflow_stream:
        checks["deerflow_stream"] = {"ok": False, "detail": skip_reason, "skipped": True}
    else:
        ok_df, msg_df = await _deerflow_stream_preflight(
            langgraph_url=langgraph_url,
            model_override=locked_model_name,
            assistant_id=assistant_id,
            agent_name=agent_name,
            use_embedded_local=use_embedded_local,
        )
        checks["deerflow_stream"] = {"ok": bool(ok_df), "detail": msg_df}

    if checks["deerflow_stream"]["ok"]:
        for check_name in ("provider_chat", "provider_tool_call"):
            check = checks.get(check_name, {})
            if bool(check.get("ok", False)):
                continue
            detail = str(check.get("detail", ""))
            if _is_hard_provider_failure(detail):
                continue
            report["warnings"].append(f"{check_name} degraded to warning: {detail}")
            check["ok"] = True
            check["degraded"] = True
            checks[check_name] = check

    failed: list[str] = []
    for name, detail in checks.items():
        if not bool(detail.get("ok", False)):
            failed.append(f"{name}: {detail.get('detail', 'failed')}")
    report["failed_checks"] = failed
    report["passed"] = len(failed) == 0
    report["finished_at_utc"] = _utc_now_iso()
    return report


async def _session_worker(**kwargs) -> None:
    token = current_agent_id.set(f"agent_{kwargs['session_index'] + 1:04d}")
    try:
        await _session_worker_body(**kwargs)
    finally:
        current_agent_id.reset(token)


async def _session_worker_body(
    *,
    queue: asyncio.Queue[dict[str, Any]],
    session_index: int,
    run_id: str,
    docs: dict[str, dict[str, Any]],
    memory: MultiLevelMemoryManager,
    event_writer: EventLogWriter,
    task_rows: list[dict[str, Any]],
    mem_rows: list[dict[str, Any]],
    langgraph_url: str,
    client: httpx.AsyncClient,
    llm_sema: asyncio.Semaphore,
    cfg,
    debug_print: bool,
    debug_max_output_chars: int,
    stop_event: asyncio.Event,
    stop_reasons: list[str],
    allow_fallback_retry: bool,
    fallback_model: str,
    fallback_reasons: list[str],
    locked_model_name: str,
    prompt_mode: str,
    max_evidence_chars: int,
    answer_retry_count: int,
    store_raw_output_chars: int,
    answer_format_mode: str,
    structured_evidence: bool,
    context_top_k: int,
    context_dedupe: bool,
    context_item_max_chars: int,
    memory_stress_turns: int,
    session_scope: str,
    assistant_id: str,
    agent_name: str,
    use_embedded_local: bool,
    use_provider_direct: bool,
    provider_base_url: str,
    provider_model_id: str,
    provider_api_key: str,
    qa_prompt_profile: str,
    qa_support_mode: str,
) -> None:
    base_agent_id = f"agent_{session_index + 1:04d}"
    base_session_id = f"{run_id}-{base_agent_id}"
    agent_id = base_agent_id
    session_id = base_session_id
    # Hard guardrail to prevent a single task from hanging the entire E2E run.
    task_timeout_s = float(max(60.0, float(os.getenv("MEMLAB_E2E_TASK_TIMEOUT_S", "420"))))
    thread_timeout_s = float(max(30.0, float(os.getenv("MEMLAB_E2E_THREAD_TIMEOUT_S", "90"))))
    stress_turns = max(0, int(memory_stress_turns))

    while True:
        if stop_event.is_set():
            return
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        task_id = str(task["task_id"])
        if str(session_scope or "worker") == "task":
            token = re.sub(r"[^A-Za-z0-9_-]+", "-", task_id).strip("-") or f"task-{now_ms()}"
            token = token[:48]
            agent_id = f"agent_task_{token}"
            session_id = f"{run_id}-{agent_id}"
        else:
            agent_id = base_agent_id
            session_id = base_session_id
        current_agent_id.set(agent_id)
        task_enqueued_ts = int(task.get("task_enqueued_ts", now_ms()))
        task_start_ts = now_ms()
        status = "success"
        error_message = ""
        accuracy = 0.0
        strict_score = 0.0
        relaxed_score = 0.0
        partial_score = 0.0
        benchmark_score = 0.0
        benchmark_metric = "strict_exact"
        quality_label = "FACTUALLY_WRONG"
        parse_failure = False
        no_answer = False
        parse_source = ""
        answer_field = "secret_code"
        task_type = str(task.get("task_type", "secret_code") or "secret_code")
        answer_metric_hint = str(((task.get("metadata") or {}) if isinstance(task.get("metadata"), dict) else {}).get("answer_metric_hint", ""))
        effective_qa_prompt_profile = _resolve_task_qa_prompt_profile(task=task, default_profile=qa_prompt_profile)
        tokens_in: int | None = None
        tokens_out: int | None = None
        injected_tokens = 0
        injected_items = 0
        llm_request_latency_ms: float | None = None
        llm_queue_wait_ms = 0.0
        parsed_answer = ""
        parsed_answer_raw = ""
        raw_output = ""
        raw_output_preview = ""
        raw_output_stored = ""
        retrieve_skipped_reason: str | None = None
        embedding_cache_hit: bool | None = None
        retrieval_cache_hit: bool | None = None
        fallback_retry_used = False
        format_retry_count = 0
        evidence_chars_used = 0
        try:
            if use_provider_direct:
                thread_id = f"provider-direct-{session_id}-{task_id}"
            else:
                thread_id = await asyncio.wait_for(
                    _create_deerflow_thread(
                        client=client,
                        langgraph_url=langgraph_url,
                        session_id=f"{session_id}-{task_id}",
                        agent_id=agent_id,
                        use_embedded_local=use_embedded_local,
                    ),
                    timeout=thread_timeout_s,
                )
            doc = docs.get(str(task.get("doc_id", "")), {})
            body = str(doc.get("body", ""))
            question = str(task.get("question", ""))
            question_focus = _question_focus_text(question) or question
            user_text = question_focus
            answer_field = str(task.get("answer_field", "secret_code" if task_type == "secret_code" else "final_answer"))

            user_item = MemoryItem(
                key=f"user-{task_id}",
                agent_id=agent_id,
                kind="message",
                text=user_text,
                bytes=len(user_text.encode("utf-8")),
                importance=0.7,
                created_at_ms=now_ms(),
                last_access_ms=now_ms(),
                access_count=1,
                metadata={"role": "user"},
            )
            await memory.put(user_item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)

            if task_type == "secret_code":
                evidence_text = _head_tail_clip(body, int(max_evidence_chars))
            else:
                evidence_text = _build_qa_evidence_excerpt(body=body, question=question_focus, max_chars=int(max_evidence_chars))
            evidence_metadata: dict[str, Any] = {"role": "tool", "doc_id": task.get("doc_id", "")}
            if str(prompt_mode) == "memory_neutral":
                # Public standard QA should not duplicate the direct evidence block inside recent-context packing.
                evidence_metadata["exclude_from_recent_context"] = True
            if stress_turns > 0 and str(prompt_mode) == "memory_beneficial":
                # In stress mode, force evidence to be recalled via retrieval rather than direct recent-window carryover.
                evidence_metadata["exclude_from_recent_context"] = True
                evidence_metadata["memory_stress"] = True
            evidence_item = MemoryItem(
                key=f"doc-{task.get('doc_id', 'unknown')}-{task_id}",
                agent_id=agent_id,
                kind="tool_result",
                text=evidence_text,
                bytes=len(evidence_text.encode("utf-8")),
                importance=0.9,
                created_at_ms=now_ms(),
                last_access_ms=now_ms(),
                access_count=1,
                metadata=evidence_metadata,
            )
            await memory.put(evidence_item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)
            if cfg.memory.l2.enabled and str(prompt_mode) == "memory_beneficial":
                anchors = _build_question_anchors(
                    body=body,
                    question=question,
                    max_chunks=4,
                    chunk_chars=max(240, int(context_item_max_chars)),
                )
                for anchor_idx, anchor_text in enumerate(anchors):
                    if not anchor_text.strip():
                        continue
                    anchor_item = MemoryItem(
                        key=f"doc-anchor-{task.get('doc_id', 'unknown')}-{task_id}-{anchor_idx:02d}",
                        agent_id=agent_id,
                        kind="tool_result",
                        text=anchor_text,
                        bytes=len(anchor_text.encode("utf-8")),
                        importance=0.95,
                        created_at_ms=now_ms(),
                        last_access_ms=now_ms(),
                        access_count=1,
                        metadata={
                            "role": "tool",
                            "doc_id": task.get("doc_id", ""),
                            "retrieval_anchor": True,
                            "exclude_from_recent_context": True,
                            "anchor_for_question": question[:160],
                        },
                    )
                    await memory.put(anchor_item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)
            if cfg.memory.l2.enabled and str(prompt_mode) == "memory_beneficial" and len(body) > int(max_evidence_chars):
                shard_chars = max(1600, int(context_item_max_chars) * 3)
                for shard_idx, shard in enumerate(_build_retrieval_shards(body, shard_chars=shard_chars, max_shards=8)):
                    shard_text = str(shard.get("text", "")).strip()
                    if not shard_text:
                        continue
                    shard_item = MemoryItem(
                        key=f"doc-shard-{task.get('doc_id', 'unknown')}-{task_id}-{shard_idx:02d}",
                        agent_id=agent_id,
                        kind="tool_result",
                        text=shard_text,
                        bytes=len(shard_text.encode("utf-8")),
                        importance=0.85,
                        created_at_ms=now_ms(),
                        last_access_ms=now_ms(),
                        access_count=1,
                        metadata={
                            "role": "tool",
                            "doc_id": task.get("doc_id", ""),
                            "retrieval_shard": True,
                            "exclude_from_recent_context": True,
                            "span_start": int(shard.get("start", 0)),
                            "span_end": int(shard.get("end", 0)),
                        },
                    )
                    await memory.put(shard_item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)
            if bool(structured_evidence) and task_type == "secret_code":
                secret_hint = str(doc.get("secret_code", "")).strip()
                if not secret_hint:
                    m_secret = parse_model_answer(body, answer_field="secret_code", task_type="secret_code")
                    secret_hint = str(m_secret.get("parsed_answer_raw", "")).strip()
                if secret_hint:
                    hint_text = f"SECRET_CODE:{secret_hint}"
                    hint_item = MemoryItem(
                        key=f"doc-secret-{task.get('doc_id', 'unknown')}-{task_id}",
                        agent_id=agent_id,
                        kind="tool_result",
                        text=hint_text,
                        bytes=len(hint_text.encode("utf-8")),
                        importance=1.0,
                        created_at_ms=now_ms(),
                        last_access_ms=now_ms(),
                        access_count=1,
                        metadata={"role": "tool", "doc_id": task.get("doc_id", ""), "structured": True},
                    )
                    await memory.put(hint_item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)

            packed = await memory.assemble_context(
                agent_id=agent_id,
                user_query=question_focus,
                budget=ContextBudget(
                    max_prompt_tokens=cfg.memory.l0.max_prompt_tokens,
                    max_memory_injection_tokens=cfg.memory.l0.max_memory_injection_tokens,
                    keep_recent_messages=cfg.memory.l0.keep_recent_messages,
                ),
            )
            if str(prompt_mode) == "memory_neutral":
                packed.messages = [{"role": "user", "content": question_focus}]
                packed.trace = {
                    "selected_keys": [],
                    "dropped_keys": [],
                    "summarized": False,
                    "total_tokens": 0,
                    "packed_tokens": 0,
                    "injection_tokens": 0,
                    "injection_item_count": 0,
                    "admission_reason": "direct_evidence_prompt",
                    "eviction_reason": "",
                    "pointer_fetch": 0,
                    "degrade_mode": "none",
                    "retrieve_skipped_reason": "prompt_mode_memory_neutral",
                    "embedding_cache_hit": None,
                    "retrieval_cache_hit": None,
                }
            injected_tokens = int(packed.trace.get("injection_tokens", 0))
            injected_items = int(packed.trace.get("injection_item_count", 0))
            retrieve_skipped_reason = packed.trace.get("retrieve_skipped_reason")
            embedding_cache_hit = packed.trace.get("embedding_cache_hit")
            retrieval_cache_hit = packed.trace.get("retrieval_cache_hit")
            context_lines = _prepare_context_lines(
                packed.messages,
                top_k=int(context_top_k),
                dedupe=bool(context_dedupe),
                item_max_chars=int(context_item_max_chars),
            )
            context_block = "\n".join(context_lines)
            prompt, evidence_chars_used = _summarize_prompt(
                session_id=session_id,
                question=question,
                context_block=context_block,
                body=body,
                task_type=task_type,
                answer_field=answer_field,
                answer_format_mode=answer_format_mode,
                prompt_mode=str(prompt_mode),
                qa_prompt_profile=str(effective_qa_prompt_profile),
                qa_support_mode=str(qa_support_mode),
                max_prompt_tokens=int(cfg.memory.l0.max_prompt_tokens),
                max_evidence_chars=int(max_evidence_chars),
            )

            async def _invoke_with_rate_limit_retry(model_override: str, prompt_override: str | None = None) -> dict[str, Any]:
                last_exc: Exception | None = None
                max_attempts = 6
                for attempt in range(1, max_attempts + 1):
                    try:
                        if use_provider_direct:
                            model_for_direct = str(model_override or provider_model_id).strip() or provider_model_id
                            return await _run_provider_prompt_direct(
                                client=client,
                                base_url=provider_base_url,
                                model_id=model_for_direct,
                                api_key=provider_api_key,
                                prompt=(prompt_override or prompt),
                                llm_sema=llm_sema,
                            )
                        return await _run_deerflow_prompt(
                            client=client,
                            langgraph_url=langgraph_url,
                            thread_id=thread_id,
                            prompt=(prompt_override or prompt),
                            llm_sema=llm_sema,
                            model_override=model_override,
                            assistant_id=assistant_id,
                            agent_name=agent_name,
                            use_embedded_local=use_embedded_local,
                        )
                    except Exception as exc:
                        last_exc = exc
                        if attempt < max_attempts and (
                            _is_rate_limit_error(str(exc)) or _is_transient_provider_error(str(exc))
                        ):
                            await asyncio.sleep(float(min(12, 1.5 * attempt)))
                            continue
                        break
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("unexpected empty retry state")

            try:
                resp = await asyncio.wait_for(
                    _invoke_with_rate_limit_retry(locked_model_name),
                    timeout=task_timeout_s,
                )
            except ProviderBalanceError as bal_exc:
                if allow_fallback_retry and fallback_model:
                    try:
                        resp = await asyncio.wait_for(
                            _invoke_with_rate_limit_retry(fallback_model),
                            timeout=task_timeout_s,
                        )
                    except ProviderBalanceError as bal_exc2:
                        reason = f"Provider balance error after fallback retry (task={task_id}): {_safe_error_detail(bal_exc2)}"
                        if reason not in stop_reasons:
                            stop_reasons.append(reason)
                        stop_event.set()
                        raise
                    fallback_retry_used = True
                    reason = f"Fallback retry used model '{fallback_model}' due to provider balance error."
                    if reason not in fallback_reasons:
                        fallback_reasons.append(reason)
                else:
                    reason = f"Provider balance error (task={task_id}): {_safe_error_detail(bal_exc)}"
                    if reason not in stop_reasons:
                        stop_reasons.append(reason)
                    stop_event.set()
                    raise
            tokens_in = resp.get("prompt_tokens")
            tokens_out = resp.get("completion_tokens")
            llm_request_latency_ms = float(resp.get("request_latency_ms", 0.0))
            llm_queue_wait_ms = float(resp.get("llm_queue_wait_ms", 0.0))
            raw_output = str(resp.get("content", ""))

            max_format_retry = int(max(0, answer_retry_count))
            while True:
                parsed = parse_model_answer(raw_output, answer_field=answer_field, task_type=task_type)
                parsed_answer = str(parsed.get("parsed_answer", ""))
                parsed_answer_raw = str(parsed.get("parsed_answer_raw", ""))
                score = score_task_answer(
                    raw_output=raw_output,
                    parsed_answer=parsed_answer,
                    parsed_answer_raw=parsed_answer_raw,
                    answer_key=str(task.get("answer_key", task.get("gold_answer", ""))),
                    task_type=task_type,
                    answer_metric_hint=answer_metric_hint,
                    parse_source=str(parsed.get("parse_source", "")),
                    wrong_field_candidate=str(parsed.get("wrong_field_candidate", "")),
                    status=status,
                )
                strict_score = float(score.get("strict_score", 0.0))
                relaxed_score = float(score.get("relaxed_score", 0.0))
                partial_score = float(score.get("partial_score", 0.0))
                benchmark_score = float(score.get("benchmark_score", 0.0))
                benchmark_metric = str(score.get("benchmark_metric", "strict_exact"))
                quality_label = str(score.get("label", "FACTUALLY_WRONG"))
                parse_failure = bool(score.get("parse_failure", False))
                no_answer = bool(score.get("no_answer", False))
                parse_source = str(score.get("parse_source", ""))
                accuracy = strict_score
                retry_needed = quality_label in {"PARSE_EMPTY", "PARSE_WRONG_FIELD", "REFUSAL_OR_NOANSWER"}
                if retry_needed and format_retry_count < max_format_retry:
                    format_retry_count += 1
                    repair_prompt = (
                        f"{prompt}\n\n"
                        "Previous output was invalid.\n"
                        f"Return STRICT JSON only with field `{answer_field}` and no extra prose."
                    )
                    retry_resp = await asyncio.wait_for(
                        _invoke_with_rate_limit_retry(locked_model_name, prompt_override=repair_prompt),
                        timeout=task_timeout_s,
                    )
                    raw_output = str(retry_resp.get("content", ""))
                    tokens_in = retry_resp.get("prompt_tokens")
                    tokens_out = retry_resp.get("completion_tokens")
                    llm_request_latency_ms = float(retry_resp.get("request_latency_ms", 0.0))
                    llm_queue_wait_ms = float(retry_resp.get("llm_queue_wait_ms", 0.0))
                    continue
                break

            result_text = (
                f"task_id={task_id} answer={parsed_answer} strict={strict_score} "
                f"relaxed={relaxed_score} benchmark={benchmark_score:.3f} "
                f"metric={benchmark_metric} label={quality_label}"
            )
            raw_output_preview = raw_output[:debug_max_output_chars]
            if int(store_raw_output_chars) > 0:
                raw_output_stored = raw_output[: int(store_raw_output_chars)]
            if debug_print:
                preview = raw_output_preview.replace("\r", " ").replace("\n", " ")
                print(
                    f"[DEBUG][{task_id}] raw='{preview}' parsed='{parsed_answer}' answer='{task.get('answer_key','')}' "
                    f"tokens_in={tokens_in} tokens_out={tokens_out}",
                    flush=True,
                )
            result_item = MemoryItem(
                key=f"result-{task_id}",
                agent_id=agent_id,
                kind="fact",
                text=result_text,
                bytes=len(result_text.encode("utf-8")),
                importance=1.0 if relaxed_score == 1.0 else 0.3,
                created_at_ms=now_ms(),
                last_access_ms=now_ms(),
                access_count=1,
                metadata={
                    "role": "assistant",
                    "strict_score": strict_score,
                    "relaxed_score": relaxed_score,
                    "benchmark_score": benchmark_score,
                    "benchmark_metric": benchmark_metric,
                    "quality_label": quality_label,
                },
            )
            await memory.put(result_item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)
            await memory.checkpoint(
                agent_id=agent_id,
                thread_id=session_id,
                state={
                    "task_id": task_id,
                    "accuracy": accuracy,
                    "strict_score": strict_score,
                    "relaxed_score": relaxed_score,
                    "benchmark_score": benchmark_score,
                    "benchmark_metric": benchmark_metric,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
            )
        except Exception as exc:
            status = "error"
            detail = _safe_error_detail(exc)
            error_message = detail[:320]
        task_end_ts = now_ms()
        queue_wait_ms = max(0.0, float(task_start_ts - task_enqueued_ts))
        processing_ms = max(0.0, float(task_end_ts - task_start_ts))
        proc_llm_ms = max(0.0, float(llm_request_latency_ms or 0.0))
        proc_non_llm_ms = max(0.0, processing_ms - proc_llm_ms)
        end_to_end_ms = max(0.0, float(task_end_ts - task_enqueued_ts))
        row = {
            "run_id": run_id,
            "op_type": "task",
            "op_name": "task_result",
            "task_id": task_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "start_ts": task_start_ts,
            "end_ts": task_end_ts,
            "task_enqueued_ts": task_enqueued_ts,
            "task_start_ts": task_start_ts,
            "task_end_ts": task_end_ts,
            "queue_wait_ms": float(queue_wait_ms),
            "processing_ms": float(processing_ms),
            "proc_whole_ms": float(processing_ms),
            "proc_llm_ms": float(proc_llm_ms),
            "proc_non_llm_ms": float(proc_non_llm_ms),
            "end_to_end_ms": float(end_to_end_ms),
            "status": status,
            "accuracy_applicable": True,
            "accuracy": float(accuracy),
            "strict_score": float(strict_score),
            "relaxed_score": float(relaxed_score),
            "partial_score": float(partial_score),
            "benchmark_score": float(benchmark_score),
            "benchmark_metric": benchmark_metric,
            "quality_label": quality_label,
            "parse_failure": bool(parse_failure),
            "no_answer": bool(no_answer),
            "parse_source": parse_source,
            "answer_metric_hint": answer_metric_hint,
            "task_type": task_type,
            "answer_field": answer_field,
            "qa_prompt_profile": str(effective_qa_prompt_profile),
            "qa_support_mode": str(qa_support_mode),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": float(end_to_end_ms),
            "context_injected_tokens": int(injected_tokens),
            "context_injected_items": int(injected_items),
            "retrieve_skipped_reason": retrieve_skipped_reason,
            "embedding_cache_hit": embedding_cache_hit,
            "retrieval_cache_hit": retrieval_cache_hit,
            "prompt_mode": str(prompt_mode),
            "memory_stress_turns": int(stress_turns),
            "evidence_chars_used": int(evidence_chars_used),
            "llm_queue_wait_ms": float(llm_queue_wait_ms),
            "llm_request_latency_ms": llm_request_latency_ms,
            "llm_latency_ms": llm_request_latency_ms,
            "parsed_answer": parsed_answer,
            "parsed_answer_raw": parsed_answer_raw,
            "raw_output": raw_output_stored,
            "format_retry_count": int(format_retry_count),
            "fallback_retry_used": bool(fallback_retry_used),
            "error": error_message,
        }
        if debug_print:
            row["raw_output_preview"] = raw_output_preview
        task_rows.append(row)
        event_writer.append(row)
        queue.task_done()
        if stop_event.is_set():
            return


async def _run() -> int:
    args = parse_common_args(default_sessions=50, default_workload_id="E2E-50")
    cfg = load_runtime_config(args)
    model_base_url = (os.getenv("OPENAI_BASE_URL", "") or "").strip().rstrip("/")
    model_id = (os.getenv("OPENAI_MODEL", "") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    locked_base_url = str(args.locked_base_url or LOCKED_BASE_URL).strip().rstrip("/")
    locked_model_id = str(args.locked_model_id or LOCKED_MODEL_ID).strip()
    if not model_base_url or not model_id or not api_key:
        raise RuntimeError("Set OPENAI_BASE_URL, OPENAI_MODEL and OPENAI_API_KEY before an E2E run")
    comparable_mode = not bool(args.allow_fallback_retry)
    use_embedded_local = _should_use_embedded_deerflow(
        langgraph_url=str(args.deerflow_langgraph_url),
        experiment_track=str(args.experiment_track),
    )
    use_provider_direct = _is_truthy(os.getenv("MEMLAB_E2E_PROVIDER_DIRECT"))
    if (not use_provider_direct) and str(locked_model_id).lower() == "pro/deepseek-ai/deepseek-v3.2":
        # Default DeepSeek runs to direct provider transport for stability and throughput.
        use_provider_direct = True
    runtime_model_name = locked_model_id if use_provider_direct else LOCKED_DEERFLOW_MODEL_NAME

    run_id = f"{args.variant.lower()}-e2e-r{args.repeat_idx}-{now_ms()}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = run_env_checks(
        variant_id=args.variant,
        require_deerflow=True,
        deerflow_langgraph_url=args.deerflow_langgraph_url,
        deerflow_gateway_url=args.deerflow_gateway_url,
        locked_base_url=locked_base_url,
        locked_model_id=locked_model_id,
    )
    manifest = build_manifest(
        run_id=run_id,
        variant_id=args.variant,
        workload_id=args.workload_id,
        seed=args.seed,
        repeat_idx=args.repeat_idx,
        env_valid=env.env_valid,
        env_invalid_reasons=env.env_invalid_reasons,
        repro_valid=bool(env.memlab_git_commit),
        repro_invalid_reasons=[] if env.memlab_git_commit else ["memlab git commit hash is unavailable."],
        llm_real=None,
        llm_invalid_reasons=[],
        usage_unavailable=False,
        usage_unavailable_reasons=[],
        model_base_url=model_base_url or None,
        model_id=model_id or None,
        model_fallback_used=False,
        comparable_mode=comparable_mode,
        comparability_invalid_reasons=[],
        metrics_schema_version=5,
        memlab_git_commit=env.memlab_git_commit,
        deerflow_git_commit=env.deerflow_git_commit,
        docker_image_digests=env.docker_image_digests,
        infra_versions=env.infra_versions,
        track_id=(str(args.track_id).strip() or f"{str(args.experiment_track).strip()}:{str(args.prompt_mode).strip()}"),
        extra={
            "runner": "runner_e2e",
            "llm_max_inflight": args.llm_max_inflight,
            "rate_limit_safe_max_inflight": _rate_limit_safe_max_inflight(),
            "experiment_track": args.experiment_track,
            "track_id": (str(args.track_id).strip() or f"{str(args.experiment_track).strip()}:{str(args.prompt_mode).strip()}"),
            "deerflow_transport": (
                "provider_direct" if use_provider_direct else ("embedded_local" if use_embedded_local else "langgraph_http")
            ),
            "provider_direct": bool(use_provider_direct),
            "dataset_id": str(args.dataset_id or ""),
            "prepared_full_or_slice": str(args.prepared_full_or_slice or ""),
            "prompt_mode": args.prompt_mode,
            "qa_prompt_profile": args.qa_prompt_profile,
            "qa_support_mode": args.qa_support_mode,
            "session_scope": args.session_scope,
            "max_evidence_chars": args.max_evidence_chars,
            "assistant_id": args.assistant_id,
            "agent_name": args.agent_name,
            "worker_processes": args.worker_processes,
            "sessions_per_process": args.sessions_per_process,
            "legacy_variant_alias": args.legacy_variant_alias,
            "deerflow_langgraph_url": args.deerflow_langgraph_url,
            "deerflow_gateway_url": args.deerflow_gateway_url,
            "tasks_path": args.tasks_path,
            "docs_path": args.docs_path,
            "debug_print": args.debug_print,
            "answer_retry_count": args.answer_retry_count,
            "store_raw_output_chars": args.store_raw_output_chars,
            "answer_format_mode": args.answer_format_mode,
            "structured_evidence": args.structured_evidence,
            "context_top_k": args.context_top_k,
            "context_dedupe": args.context_dedupe,
            "context_item_max_chars": args.context_item_max_chars,
            "memory_stress_turns": args.memory_stress_turns,
            "allow_fallback_retry": args.allow_fallback_retry,
            "fallback_model": (args.fallback_model.strip() or os.getenv("OPENAI_FALLBACK_MODEL", "").strip()),
            "model_lock": {
                "expected_base_url": locked_base_url,
                "expected_model_id": locked_model_id,
                "expected_deerflow_model_name": runtime_model_name,
                "actual_base_url": model_base_url,
                "actual_model_id": model_id,
            },
            "preflight_passed": None,
            "preflight_report_path": "preflight.json",
            "error_breakdown": {},
        },
    )
    write_manifest(run_dir / "manifest.json", manifest)

    if comparable_mode:
        if model_base_url != locked_base_url or model_id != locked_model_id:
            reason = f"Model lock violation: expected {locked_model_id} on {locked_base_url}"
            manifest["env_valid"] = False
            manifest.setdefault("env_invalid_reasons", [])
            if reason not in manifest["env_invalid_reasons"]:
                manifest["env_invalid_reasons"].append(reason)
            manifest["repro_valid"] = False
            manifest.setdefault("repro_invalid_reasons", [])
            if reason not in manifest["repro_invalid_reasons"]:
                manifest["repro_invalid_reasons"].append(reason)
            manifest.setdefault("comparability_invalid_reasons", [])
            if reason not in manifest["comparability_invalid_reasons"]:
                manifest["comparability_invalid_reasons"].append(reason)
            write_manifest(run_dir / "manifest.json", manifest)
            print("STOP: run invalid; refusing fallback.")
            print(f"- {reason}")
            return 1
        safe_max_inflight = _rate_limit_safe_max_inflight()
        if int(args.llm_max_inflight) > safe_max_inflight:
            reason = (
                "Rate-limit safety violation: "
                f"llm_max_inflight={int(args.llm_max_inflight)} > recommended {safe_max_inflight}."
            )
            manifest["env_valid"] = False
            manifest.setdefault("env_invalid_reasons", [])
            if reason not in manifest["env_invalid_reasons"]:
                manifest["env_invalid_reasons"].append(reason)
            manifest["repro_valid"] = False
            manifest.setdefault("repro_invalid_reasons", [])
            if reason not in manifest["repro_invalid_reasons"]:
                manifest["repro_invalid_reasons"].append(reason)
            manifest.setdefault("comparability_invalid_reasons", [])
            if reason not in manifest["comparability_invalid_reasons"]:
                manifest["comparability_invalid_reasons"].append(reason)
            write_manifest(run_dir / "manifest.json", manifest)
            print("STOP: run invalid; refusing fallback.")
            print(f"- {reason}")
            return 1

    if not bool(args.skip_preflight):
        preflight_report = await _run_preflight_checks(
            model_base_url=model_base_url,
            model_id=model_id,
            api_key=api_key,
            langgraph_url=args.deerflow_langgraph_url,
            locked_base_url=locked_base_url,
            locked_model_id=locked_model_id,
            locked_model_name=runtime_model_name,
            llm_max_inflight=int(args.llm_max_inflight),
            prompt_mode=str(args.prompt_mode),
            max_evidence_chars=int(args.max_evidence_chars),
            comparable_mode=bool(comparable_mode),
            assistant_id=str(args.assistant_id),
            agent_name=str(args.agent_name),
            use_embedded_local=use_embedded_local,
            use_provider_direct=bool(use_provider_direct),
        )
        json_dump(run_dir / "preflight.json", preflight_report)
        manifest.setdefault("extra", {})
        manifest["extra"]["preflight_passed"] = bool(preflight_report.get("passed", False))
        manifest["extra"]["preflight_report"] = preflight_report
        if not bool(preflight_report.get("passed", False)):
            manifest["env_valid"] = False
            manifest.setdefault("env_invalid_reasons", [])
            manifest["repro_valid"] = False
            manifest.setdefault("repro_invalid_reasons", [])
            for reason in preflight_report.get("failed_checks", []):
                reason_text = f"Preflight failed: {reason}"
                if reason_text not in manifest["env_invalid_reasons"]:
                    manifest["env_invalid_reasons"].append(reason_text)
                if reason_text not in manifest["repro_invalid_reasons"]:
                    manifest["repro_invalid_reasons"].append(reason_text)
                manifest.setdefault("comparability_invalid_reasons", [])
                if reason_text not in manifest["comparability_invalid_reasons"]:
                    manifest["comparability_invalid_reasons"].append(reason_text)
            write_manifest(run_dir / "manifest.json", manifest)
            print("STOP: run invalid; refusing fallback.")
            for reason in preflight_report.get("failed_checks", []):
                print(f"- Preflight failed: {reason}")
            return 1
        write_manifest(run_dir / "manifest.json", manifest)
    else:
        manifest.setdefault("extra", {})
        manifest["extra"]["preflight_passed"] = None
        manifest["extra"]["preflight_report"] = None
        write_manifest(run_dir / "manifest.json", manifest)

    if bool(args.preflight_only):
        print(f"Preflight complete: {run_id}")
        print(f"Manifest: {run_dir / 'manifest.json'}")
        print(f"Preflight: {run_dir / 'preflight.json'}")
        return 0

    fatal_reasons = list(env.env_invalid_reasons)
    if fatal_reasons:
        print("STOP: required services missing; refusing fallback.")
        for r in fatal_reasons:
            print(f"- {r}")
        return 1

    tasks = shuffle_tasks(load_e2e_tasks(args.tasks_path), seed=args.seed + args.repeat_idx)
    docs = load_docs(args.docs_path)

    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for task in tasks:
        enriched = dict(task)
        enriched["task_enqueued_ts"] = now_ms()
        await q.put(enriched)

    event_writer = EventLogWriter(run_dir / "eventlog.jsonl")
    task_rows: list[dict[str, Any]] = []
    mem_rows: list[dict[str, Any]] = []
    memory = MultiLevelMemoryManager(
        cfg, op_logger=make_operation_logger(run_id=run_id, mem_rows=mem_rows, event_writer=event_writer)
    )
    await memory.init()
    llm_sema = asyncio.Semaphore(max(1, args.llm_max_inflight))
    stop_event = asyncio.Event()
    stop_reasons: list[str] = []
    fallback_reasons: list[str] = []

    capacity = max(1, args.worker_processes * args.sessions_per_process)
    if capacity < args.n_sessions:
        reason = (
            "Invalid session topology: "
            f"worker_processes*sessions_per_process={capacity} is smaller than n_sessions={args.n_sessions}."
        )
        manifest["env_valid"] = False
        manifest.setdefault("env_invalid_reasons", [])
        manifest["env_invalid_reasons"].append(reason)
        write_manifest(run_dir / "manifest.json", manifest)
        print("STOP: required services missing; refusing fallback.")
        print(f"- {reason}")
        await memory.close()
        return 1
    worker_count = args.n_sessions
    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
        workers = [
            _session_worker(
                queue=q,
                session_index=i,
                run_id=run_id,
                docs=docs,
                memory=memory,
                event_writer=event_writer,
                task_rows=task_rows,
                mem_rows=mem_rows,
                langgraph_url=args.deerflow_langgraph_url,
                client=client,
                llm_sema=llm_sema,
                cfg=cfg,
                debug_print=args.debug_print,
                debug_max_output_chars=args.debug_max_output_chars,
                stop_event=stop_event,
                stop_reasons=stop_reasons,
                allow_fallback_retry=args.allow_fallback_retry,
                fallback_model=(args.fallback_model.strip() or os.getenv("OPENAI_FALLBACK_MODEL", "").strip()),
                fallback_reasons=fallback_reasons,
                locked_model_name=runtime_model_name,
                prompt_mode=str(args.prompt_mode),
                max_evidence_chars=int(args.max_evidence_chars),
                answer_retry_count=int(args.answer_retry_count),
                store_raw_output_chars=int(args.store_raw_output_chars),
                answer_format_mode=str(args.answer_format_mode),
                structured_evidence=bool(args.structured_evidence),
                context_top_k=int(args.context_top_k),
                context_dedupe=bool(args.context_dedupe),
                context_item_max_chars=int(args.context_item_max_chars),
                memory_stress_turns=int(args.memory_stress_turns),
                session_scope=str(args.session_scope),
                assistant_id=str(args.assistant_id),
                agent_name=str(args.agent_name),
                use_embedded_local=use_embedded_local,
                use_provider_direct=bool(use_provider_direct),
                provider_base_url=locked_base_url,
                provider_model_id=locked_model_id,
                provider_api_key=api_key,
                qa_prompt_profile=str(args.qa_prompt_profile),
                qa_support_mode=str(args.qa_support_mode),
            )
            for i in range(worker_count)
        ]
        await asyncio.gather(*workers)
    try:
        await memory.close()
    except Exception as exc:
        stop_reasons.append(f"memory_close_failed: {_safe_error_detail(exc)}")

    summary = compute_run_summary(
        run_id=run_id,
        variant_id=args.variant,
        workload_id=args.workload_id,
        n_sessions=args.n_sessions,
        tasks=task_rows,
        mem_ops=mem_rows,
    )
    retry_count_total = int(sum(int(r.get("format_retry_count", 0)) for r in task_rows))
    summary["retry_count_total"] = retry_count_total
    error_breakdown = _build_error_breakdown(task_rows)
    summary["error_breakdown"] = error_breakdown
    json_dump(run_dir / "summary.json", summary)

    comparability_reasons: list[str] = []
    llm_post_reasons: list[str] = []
    p50 = float(summary.get("llm_latency", summary.get("llm_request_latency", {})).get("p50", 0.0))
    default_min_success_rate = 0.60 if use_provider_direct else 0.80
    try:
        min_success_rate = float(os.getenv("MEMLAB_E2E_MIN_SUCCESS_RATE", str(default_min_success_rate)))
    except Exception:
        min_success_rate = default_min_success_rate
    min_success_rate = max(0.0, min(1.0, float(min_success_rate)))
    manifest.setdefault("extra", {})
    manifest["extra"]["comparability_min_success_rate"] = min_success_rate
    min_realistic_p50_ms = 120.0 if use_provider_direct else 300.0
    if int(summary.get("success_tasks", 0)) <= 0:
        llm_post_reasons.append("E2E run has no successful tasks; refusing to mark comparable.")
    if summary.get("success_tasks", 0) > 0 and p50 < min_realistic_p50_ms:
        llm_post_reasons.append(
            f"E2E LLM request latency suspiciously low (p50={p50:.2f}ms < {min_realistic_p50_ms:.0f}ms)."
        )
    success_rate = float(summary.get("success_rate", 0.0))
    if success_rate < min_success_rate:
        comparability_reasons.append(
            f"E2E success_rate too low for comparability ({success_rate:.3f} < {min_success_rate:.3f})."
        )
    usage_unavailable = bool(summary.get("usage_unavailable", False))
    usage_reasons: list[str] = []
    if usage_unavailable:
        usage_reasons.append("Provider usage fields are missing for one or more successful tasks.")

    if stop_reasons:
        manifest["repro_valid"] = False
        manifest["llm_real"] = False
        manifest["llm_invalid_reasons"] = stop_reasons
        manifest.setdefault("extra", {})
        manifest["extra"]["error_breakdown"] = error_breakdown
        manifest["extra"]["retry_count"] = retry_count_total
        manifest.setdefault("repro_invalid_reasons", [])
        for reason in stop_reasons:
            if reason not in manifest["repro_invalid_reasons"]:
                manifest["repro_invalid_reasons"].append(reason)
        manifest.setdefault("comparability_invalid_reasons", [])
        for reason in stop_reasons:
            if reason not in manifest["comparability_invalid_reasons"]:
                manifest["comparability_invalid_reasons"].append(reason)
        manifest["usage_unavailable"] = usage_unavailable
        manifest["usage_unavailable_reasons"] = usage_reasons
        write_manifest(run_dir / "manifest.json", manifest)
        print("STOP: run invalid; refusing fallback.")
        for reason in stop_reasons:
            print(f"- {reason}")
        return 1

    manifest["llm_real"] = len(llm_post_reasons) == 0
    manifest["llm_invalid_reasons"] = llm_post_reasons
    manifest["usage_unavailable"] = usage_unavailable
    manifest["usage_unavailable_reasons"] = usage_reasons
    manifest.setdefault("extra", {})
    manifest["extra"]["error_breakdown"] = error_breakdown
    manifest["extra"]["retry_count"] = retry_count_total
    if fallback_reasons:
        manifest["repro_valid"] = False
        manifest["llm_real"] = False
        manifest["model_fallback_used"] = True
        manifest.setdefault("repro_invalid_reasons", [])
        for reason in fallback_reasons:
            if reason not in manifest["repro_invalid_reasons"]:
                manifest["repro_invalid_reasons"].append(reason)
        manifest.setdefault("comparability_invalid_reasons", [])
        for reason in fallback_reasons:
            if reason not in manifest["comparability_invalid_reasons"]:
                manifest["comparability_invalid_reasons"].append(reason)
        manifest["llm_invalid_reasons"] = list(manifest.get("llm_invalid_reasons", [])) + list(fallback_reasons)
    if llm_post_reasons or comparability_reasons:
        manifest["repro_valid"] = False
        manifest.setdefault("repro_invalid_reasons", [])
        for reason in llm_post_reasons:
            if reason not in manifest["repro_invalid_reasons"]:
                manifest["repro_invalid_reasons"].append(reason)
        for reason in comparability_reasons:
            if reason not in manifest["repro_invalid_reasons"]:
                manifest["repro_invalid_reasons"].append(reason)
        manifest.setdefault("comparability_invalid_reasons", [])
        for reason in llm_post_reasons + comparability_reasons:
            if reason not in manifest["comparability_invalid_reasons"]:
                manifest["comparability_invalid_reasons"].append(reason)
        write_manifest(run_dir / "manifest.json", manifest)
        print("STOP: run invalid; refusing fallback.")
        for reason in llm_post_reasons + comparability_reasons:
            print(f"- {reason}")
        return 1

    manifest["summary_chain_valid"] = bool((run_dir / "eventlog.jsonl").exists() and (run_dir / "summary.json").exists())
    write_manifest(run_dir / "manifest.json", manifest)
    print(f"Run complete: {run_id}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Event log: {run_dir / 'eventlog.jsonl'}")
    print(f"Summary: {run_dir / 'summary.json'}")
    return 0


def main() -> None:
    rc = 1
    try:
        rc = asyncio.run(_run())
    finally:
        _shutdown_embedded_worker_sync()
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
