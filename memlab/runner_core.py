from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memlab.config import MemlabConfig, load_config
from memlab.io import jsonl_read, now_ms
from memlab.metrics.stats import summarize_latency


SECRET_RE = re.compile(r"SECRET[_\s-]*CODE\s*[:=]\s*([A-Z0-9_-]+)", re.IGNORECASE)
JSON_SECRET_RE = re.compile(r'"secret_code"\s*:\s*"([A-Z0-9_-]+)"', re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)
LIKELY_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{6,}$")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
OPTION_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
REFUSAL_HINT_RE = re.compile(
    r"(?:\b(can(?:not|'t)|unable|sorry|cannot comply|don't know|not enough info|no answer|refuse|decline)\b|无法|不能|抱歉|不知道|无法回答)",
    re.IGNORECASE,
)


@dataclass
class CommonArgs:
    config: str
    variant: str
    legacy_variant_alias: str | None
    n_sessions: int
    llm_max_inflight: int
    worker_processes: int
    sessions_per_process: int
    repeat_idx: int
    seed: int
    workload_id: str
    deerflow_langgraph_url: str
    deerflow_gateway_url: str
    locked_base_url: str
    locked_model_id: str
    experiment_track: str
    track_id: str
    dataset_id: str
    prepared_full_or_slice: str
    qa_prompt_profile: str
    tasks_path: str
    docs_path: str
    debug_print: bool
    debug_max_output_chars: int
    allow_fallback_retry: bool
    fallback_model: str
    prompt_mode: str
    qa_support_mode: str
    max_evidence_chars: int
    answer_retry_count: int
    store_raw_output_chars: int
    answer_format_mode: str
    structured_evidence: bool
    context_top_k: int
    context_dedupe: bool
    context_item_max_chars: int
    memory_stress_turns: int
    session_scope: str
    preflight_only: bool
    skip_preflight: bool
    assistant_id: str
    agent_name: str


CANONICAL_VARIANTS = ("B0", "B1", "V1", "F1", "F2")
LEGACY_VARIANT_ALIASES = {
    "A1": "V1",
    "A2": "F1",
}


def normalize_variant(variant: str) -> tuple[str, str | None]:
    requested = str(variant or "").upper().strip()
    if requested in CANONICAL_VARIANTS:
        return requested, None
    if requested in LEGACY_VARIANT_ALIASES:
        return LEGACY_VARIANT_ALIASES[requested], requested
    valid = ", ".join([*CANONICAL_VARIANTS, *LEGACY_VARIANT_ALIASES.keys()])
    raise ValueError(f"Unknown variant: {variant}. Expected one of: {valid}")


def apply_variant(cfg: MemlabConfig, variant: str, *, legacy_alias: str | None = None) -> MemlabConfig:
    canonical, inferred_alias = normalize_variant(variant)
    alias = legacy_alias or inferred_alias
    cfg.experiment.variant_id = canonical
    if canonical == "B0":
        cfg.memory.l1.enabled = False
        cfg.memory.l2.enabled = False
        cfg.memory.l3.enabled = False
        cfg.experiment.summarization_on = False
        cfg.memory.l0.max_memory_injection_tokens = 0
    elif canonical == "B1":
        cfg.memory.l1.enabled = False
        cfg.memory.l2.enabled = False
        cfg.memory.l3.enabled = False
        cfg.experiment.summarization_on = True
        cfg.memory.l0.max_memory_injection_tokens = 0
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
    if alias == "A2":
        # Preserve historical behavior for legacy ablation alias.
        cfg.experiment.redis_policy = "allkeys-lru"
    return cfg


def parse_common_args(default_sessions: int, default_workload_id: str) -> CommonArgs:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/memlab.default.yaml")
    ap.add_argument("--variant", type=str, required=True, help="B0|B1|V1|F1|F2 (legacy aliases: A1|A2)")
    ap.add_argument("--n_sessions", type=int, default=default_sessions)
    ap.add_argument("--llm_max_inflight", type=int, default=2)
    ap.add_argument("--worker_processes", type=int, default=1)
    ap.add_argument("--sessions_per_process", type=int, default=50)
    ap.add_argument("--repeat_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workload_id", type=str, default=default_workload_id)
    ap.add_argument("--deerflow_langgraph_url", type=str, default="http://127.0.0.1:2024")
    ap.add_argument("--deerflow_gateway_url", type=str, default="http://127.0.0.1:8001")
    ap.add_argument("--locked_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    ap.add_argument("--locked_model_id", type=str, default=os.getenv("OPENAI_MODEL", ""))
    ap.add_argument(
        "--experiment_track",
        type=str,
        default="mainline",
        choices=(
            "mainline",
            "candidate_race",
            "emergency_rebaseline",
            "memory_exposure",
            "focused_validation_v3",
            "alignment_diagnosis",
        ),
    )
    ap.add_argument("--track_id", type=str, default="")
    ap.add_argument("--dataset_id", type=str, default="")
    ap.add_argument("--prepared_full_or_slice", type=str, default="")
    ap.add_argument(
        "--qa_prompt_profile",
        type=str,
        default="default",
        choices=("default", "yes_no", "entity_span", "short_answer", "summary", "multiple_choice_letter"),
    )
    ap.add_argument("--tasks_path", type=str, default="workloads/e2e/tasks.jsonl")
    ap.add_argument("--docs_path", type=str, default="data/docs.jsonl")
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--debug_max_output_chars", type=int, default=200)
    ap.add_argument("--allow_fallback_retry", action="store_true")
    ap.add_argument("--fallback_model", type=str, default="")
    ap.add_argument(
        "--prompt_mode",
        type=str,
        default="memory_neutral",
        choices=("memory_neutral", "memory_beneficial"),
        help="memory_neutral injects evidence directly; memory_beneficial relies on memory retrieval.",
    )
    ap.add_argument(
        "--qa_support_mode",
        type=str,
        default="standard",
        choices=("standard", "none"),
        help="Controls whether QA prompts include direct evidence excerpts in addition to memory retrieval context.",
    )
    ap.add_argument("--max_evidence_chars", type=int, default=3000)
    ap.add_argument("--answer_retry_count", type=int, default=1)
    ap.add_argument("--store_raw_output_chars", type=int, default=0)
    ap.add_argument(
        "--answer_format_mode",
        type=str,
        default="strict_json",
        choices=("strict_json", "final_answer_prefix"),
    )
    ap.add_argument("--context_top_k", type=int, default=12)
    ap.add_argument("--structured_evidence", action="store_true")
    ap.add_argument("--context_dedupe", action="store_true")
    ap.add_argument("--context_item_max_chars", type=int, default=700)
    ap.add_argument(
        "--session_scope",
        type=str,
        default="worker",
        choices=("worker", "task"),
        help="worker reuses one memory namespace per worker; task isolates memory namespace per task.",
    )
    ap.add_argument(
        "--memory_stress_turns",
        type=int,
        default=0,
        help="Inject N distractor turns after evidence write to stress long-history memory retrieval.",
    )
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--assistant_id", type=str, default="lead_agent")
    ap.add_argument("--agent_name", type=str, default="memlab-e2e")
    ns = ap.parse_args()
    canonical_variant, legacy_alias = normalize_variant(ns.variant)
    return CommonArgs(
        config=ns.config,
        variant=canonical_variant,
        legacy_variant_alias=legacy_alias,
        n_sessions=int(ns.n_sessions),
        llm_max_inflight=int(ns.llm_max_inflight),
        worker_processes=int(ns.worker_processes),
        sessions_per_process=int(ns.sessions_per_process),
        repeat_idx=int(ns.repeat_idx),
        seed=int(ns.seed),
        workload_id=str(ns.workload_id),
        deerflow_langgraph_url=str(ns.deerflow_langgraph_url),
        deerflow_gateway_url=str(ns.deerflow_gateway_url),
        locked_base_url=str(ns.locked_base_url or "").strip().rstrip("/"),
        locked_model_id=str(ns.locked_model_id or "").strip(),
        experiment_track=str(ns.experiment_track or "mainline").strip(),
        track_id=str(ns.track_id or "").strip(),
        dataset_id=str(ns.dataset_id or "").strip(),
        prepared_full_or_slice=str(ns.prepared_full_or_slice or "").strip(),
        qa_prompt_profile=str(ns.qa_prompt_profile or "default").strip(),
        tasks_path=str(ns.tasks_path),
        docs_path=str(ns.docs_path),
        debug_print=bool(ns.debug_print),
        debug_max_output_chars=max(1, int(ns.debug_max_output_chars)),
        allow_fallback_retry=bool(ns.allow_fallback_retry),
        fallback_model=str(ns.fallback_model or ""),
        prompt_mode=str(ns.prompt_mode),
        qa_support_mode=str(ns.qa_support_mode or "standard").strip(),
        max_evidence_chars=max(256, int(ns.max_evidence_chars)),
        answer_retry_count=max(0, int(ns.answer_retry_count)),
        store_raw_output_chars=max(0, int(ns.store_raw_output_chars)),
        answer_format_mode=str(ns.answer_format_mode),
        structured_evidence=bool(ns.structured_evidence),
        context_top_k=max(1, int(ns.context_top_k)),
        context_dedupe=bool(ns.context_dedupe),
        context_item_max_chars=max(120, int(ns.context_item_max_chars)),
        memory_stress_turns=max(0, int(ns.memory_stress_turns)),
        session_scope=str(ns.session_scope or "worker"),
        preflight_only=bool(ns.preflight_only),
        skip_preflight=bool(ns.skip_preflight),
        assistant_id=str(ns.assistant_id or "lead_agent"),
        agent_name=str(ns.agent_name or "memlab-e2e"),
    )


def load_runtime_config(common_args: CommonArgs) -> MemlabConfig:
    cfg = load_config(common_args.config)
    cfg = apply_variant(cfg, common_args.variant, legacy_alias=common_args.legacy_variant_alias)
    cfg.experiment.n_agents = common_args.n_sessions
    cfg.experiment.seed = common_args.seed
    return cfg


def load_e2e_tasks(path: str | Path = "workloads/e2e/tasks.jsonl") -> list[dict[str, Any]]:
    return jsonl_read(path)


def load_docs(path: str | Path = "data/docs.jsonl") -> dict[str, dict[str, Any]]:
    docs = jsonl_read(path)
    return {str(d["doc_id"]): d for d in docs}


def normalize_secret(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def normalize_text_relaxed(text: str) -> str:
    text = str(text or "").lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_option_letter(text: str) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return ""
    m = OPTION_LETTER_RE.search(raw)
    if m:
        return str(m.group(1)).upper()
    first = raw[:1]
    return first if first in {"A", "B", "C", "D"} else ""


def normalize_exact_span(text: str) -> str:
    norm = normalize_text_relaxed(text)
    norm = LEADING_ARTICLE_RE.sub("", norm).strip()
    return norm


def resolve_answer_metric_hint(*, task_type: str, answer_metric_hint: str = "") -> str:
    if str(task_type or "").strip().lower() == "secret_code":
        return "secret_code"
    hint = str(answer_metric_hint or "").strip().lower()
    return hint or "strict_exact"


def compute_benchmark_score(
    *,
    prediction: str,
    answer_key: str,
    task_type: str,
    answer_metric_hint: str = "",
    strict_score: float = 0.0,
    relaxed_score: float = 0.0,
    partial_score: float = 0.0,
) -> tuple[str, float]:
    metric = resolve_answer_metric_hint(task_type=task_type, answer_metric_hint=answer_metric_hint)
    pred = str(prediction or "").strip()
    gold = str(answer_key or "").strip()
    if not pred or not gold:
        return metric, 0.0

    if metric == "secret_code":
        score = 1.0 if normalize_secret(pred) == normalize_secret(gold) else 0.0
    elif metric == "exact_option":
        score = 1.0 if normalize_option_letter(pred) == normalize_option_letter(gold) else 0.0
    elif metric == "exact_span":
        score = 1.0 if normalize_exact_span(pred) == normalize_exact_span(gold) else 0.0
    elif metric == "relaxed_span":
        pred_norm = normalize_exact_span(pred)
        gold_norm = normalize_exact_span(gold)
        score = 1.0 if pred_norm and pred_norm == gold_norm else _token_overlap_f1(pred_norm, gold_norm)
    elif metric == "token_f1":
        score = _token_overlap_f1(normalize_text_relaxed(pred), normalize_text_relaxed(gold))
    else:
        score = strict_score if strict_score > 0.0 else max(relaxed_score, partial_score)
    return metric, max(0.0, min(1.0, float(score)))


def extract_secret_code(text: str) -> str:
    m = SECRET_RE.search(str(text or ""))
    if not m:
        return ""
    return normalize_secret(m.group(1))


def _norm_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _looks_like_code(text: str) -> bool:
    return bool(LIKELY_CODE_RE.match(str(text or "").strip()))


def _find_first_field(obj: Any, target_fields: set[str], path: str = "") -> tuple[str, str] | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = _norm_key(key)
            next_path = f"{path}.{key_norm}" if path else key_norm
            if key_norm in target_fields and isinstance(value, str):
                return str(value), next_path
            nested = _find_first_field(value, target_fields, next_path)
            if nested:
                return nested
        return None
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            next_path = f"{path}[{idx}]"
            nested = _find_first_field(item, target_fields, next_path)
            if nested:
                return nested
        return None
    return None


def _find_wrong_field_candidate(obj: Any, target_fields: set[str], task_type: str) -> str:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = _norm_key(key)
            if key_norm in target_fields:
                continue
            if isinstance(value, str):
                val = str(value).strip()
                if task_type == "secret_code":
                    if ("answer" in key_norm or "code" in key_norm or "secret" in key_norm) and _looks_like_code(val):
                        return key_norm
                else:
                    if "answer" in key_norm and val:
                        return key_norm
            nested = _find_wrong_field_candidate(value, target_fields, task_type)
            if nested:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _find_wrong_field_candidate(item, target_fields, task_type)
            if nested:
                return nested
    return ""


def _json_candidates(text: str) -> list[Any]:
    out: list[Any] = []
    try:
        out.append(json.loads(text))
    except Exception:
        pass
    for match in FENCED_JSON_RE.finditer(text):
        block = match.group(1)
        try:
            out.append(json.loads(block))
        except Exception:
            continue
    for candidate in re.findall(r"\{[^{}]*\}", text):
        try:
            out.append(json.loads(candidate))
        except Exception:
            continue
    return out


def parse_model_answer(
    text: str,
    *,
    answer_field: str = "secret_code",
    task_type: str = "secret_code",
) -> dict[str, Any]:
    body = str(text or "")
    field = _norm_key(answer_field or ("secret_code" if task_type == "secret_code" else "final_answer"))
    target_fields = {field}
    if task_type != "secret_code":
        target_fields |= {"final_answer", "answer", "result", "output"}
    wrong_field_candidate = ""
    for obj in _json_candidates(body):
        found = _find_first_field(obj, target_fields)
        if found:
            raw, source_path = found
            parsed = normalize_secret(raw) if task_type == "secret_code" else str(raw).strip()
            return {
                "parsed_answer": parsed,
                "parsed_answer_raw": str(raw),
                "parse_source": f"json:{source_path}",
                "wrong_field_candidate": "",
            }
        wrong_field_candidate = wrong_field_candidate or _find_wrong_field_candidate(obj, target_fields, task_type)

    if task_type == "secret_code":
        m = JSON_SECRET_RE.search(body)
        if m:
            return {
                "parsed_answer": normalize_secret(m.group(1)),
                "parsed_answer_raw": str(m.group(1)),
                "parse_source": "regex:json_secret",
                "wrong_field_candidate": wrong_field_candidate,
            }
        secret = extract_secret_code(body)
        if secret:
            return {
                "parsed_answer": secret,
                "parsed_answer_raw": secret,
                "parse_source": "regex:secret_code",
                "wrong_field_candidate": wrong_field_candidate,
            }
        return {
            "parsed_answer": "",
            "parsed_answer_raw": "",
            "parse_source": "none",
            "wrong_field_candidate": wrong_field_candidate,
        }

    field_pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*"([^"]+)"', re.IGNORECASE)
    m_field = field_pattern.search(body)
    if m_field:
        ans = str(m_field.group(1))
        return {
            "parsed_answer": ans.strip(),
            "parsed_answer_raw": ans,
            "parse_source": "regex:field",
            "wrong_field_candidate": wrong_field_candidate,
        }
    m_final = FINAL_ANSWER_RE.search(body)
    if m_final:
        ans = str(m_final.group(1)).strip().splitlines()[0].strip()
        return {
            "parsed_answer": ans,
            "parsed_answer_raw": ans,
            "parse_source": "regex:final_answer_prefix",
            "wrong_field_candidate": wrong_field_candidate,
        }
    plain = str(body).strip()
    if plain:
        return {
            "parsed_answer": plain.splitlines()[0].strip(),
            "parsed_answer_raw": plain.splitlines()[0].strip(),
            "parse_source": "fallback:first_line",
            "wrong_field_candidate": wrong_field_candidate,
        }
    return {
        "parsed_answer": "",
        "parsed_answer_raw": "",
        "parse_source": "none",
        "wrong_field_candidate": wrong_field_candidate,
    }


def parse_model_secret(text: str) -> str:
    return str(parse_model_answer(text, answer_field="secret_code", task_type="secret_code").get("parsed_answer", ""))


def _token_overlap_f1(a: str, b: str) -> float:
    aa = [x for x in a.split(" ") if x]
    bb = [x for x in b.split(" ") if x]
    if not aa or not bb:
        return 0.0
    from collections import Counter

    ca = Counter(aa)
    cb = Counter(bb)
    hit = sum(min(ca[k], cb[k]) for k in set(ca) | set(cb))
    if hit <= 0:
        return 0.0
    p = hit / len(aa)
    r = hit / len(bb)
    return (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0


def _secret_partial_ratio(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    pred_n = normalize_secret(pred)
    gold_n = normalize_secret(gold)
    if not pred_n or not gold_n:
        return 0.0
    prefix = 0
    for a, b in zip(pred_n, gold_n):
        if a != b:
            break
        prefix += 1
    return prefix / max(1, len(gold_n))


def score_task_answer(
    *,
    raw_output: str,
    parsed_answer: str,
    parsed_answer_raw: str,
    answer_key: str,
    task_type: str,
    answer_metric_hint: str = "",
    parse_source: str = "",
    wrong_field_candidate: str = "",
    status: str = "success",
) -> dict[str, Any]:
    raw = str(raw_output or "").strip()
    parsed = str(parsed_answer or "").strip()
    parsed_raw = str(parsed_answer_raw or "").strip()
    gold = str(answer_key or "").strip()
    strict_score = 0.0
    relaxed_score = 0.0
    partial_score = 0.0
    label = "FACTUALLY_WRONG"
    no_answer = False
    benchmark_metric = resolve_answer_metric_hint(task_type=task_type, answer_metric_hint=answer_metric_hint)
    benchmark_score = 0.0

    if status != "success":
        if not raw or REFUSAL_HINT_RE.search(raw):
            no_answer = True
            label = "REFUSAL_OR_NOANSWER"
        return {
            "strict_score": strict_score,
            "relaxed_score": relaxed_score,
            "partial_score": partial_score,
            "label": label,
            "parse_failure": False,
            "no_answer": no_answer,
            "parse_source": parse_source,
            "benchmark_metric": benchmark_metric,
            "benchmark_score": benchmark_score,
        }

    if not raw or raw.lower() in {"{}", "[]", "null"}:
        no_answer = True
    if not parsed:
        if no_answer or REFUSAL_HINT_RE.search(raw):
            label = "REFUSAL_OR_NOANSWER"
            no_answer = True
        elif wrong_field_candidate:
            label = "PARSE_WRONG_FIELD"
        else:
            label = "PARSE_EMPTY"
        return {
            "strict_score": 0.0,
            "relaxed_score": 0.0,
            "partial_score": 0.0,
            "label": label,
            "parse_failure": label in {"PARSE_EMPTY", "PARSE_WRONG_FIELD"},
            "no_answer": no_answer,
            "parse_source": parse_source,
            "benchmark_metric": benchmark_metric,
            "benchmark_score": benchmark_score,
        }

    strict_score = 1.0 if parsed_raw == gold else 0.0
    if task_type == "secret_code":
        pred_relaxed = normalize_secret(parsed_raw or parsed)
        gold_relaxed = normalize_secret(gold)
        partial_score = _secret_partial_ratio(parsed_raw or parsed, gold)
    else:
        pred_relaxed = normalize_text_relaxed(parsed_raw or parsed)
        gold_relaxed = normalize_text_relaxed(gold)
        partial_score = _token_overlap_f1(pred_relaxed, gold_relaxed)
    relaxed_score = 1.0 if pred_relaxed and pred_relaxed == gold_relaxed else 0.0
    benchmark_metric, benchmark_score = compute_benchmark_score(
        prediction=(parsed_raw or parsed),
        answer_key=gold,
        task_type=task_type,
        answer_metric_hint=answer_metric_hint,
        strict_score=strict_score,
        relaxed_score=relaxed_score,
        partial_score=partial_score,
    )

    if strict_score >= 1.0:
        label = "OK"
    elif relaxed_score >= 1.0:
        label = "FORMAT_MISMATCH"
    elif partial_score >= 0.5:
        label = "PARTIAL_MATCH"
    else:
        label = "FACTUALLY_WRONG"

    return {
        "strict_score": strict_score,
        "relaxed_score": relaxed_score,
        "partial_score": partial_score,
        "label": label,
        "parse_failure": False,
        "no_answer": no_answer,
        "parse_source": parse_source,
        "benchmark_metric": benchmark_metric,
        "benchmark_score": benchmark_score,
    }


def task_accuracy(predicted_secret: str, answer_key: str) -> float:
    return 1.0 if normalize_secret(predicted_secret) == normalize_secret(answer_key) else 0.0


def compute_run_summary(
    *,
    run_id: str,
    variant_id: str,
    workload_id: str,
    n_sessions: int,
    tasks: list[dict[str, Any]],
    mem_ops: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [t for t in tasks if t.get("status") == "success"]
    failed = [t for t in tasks if t.get("status") != "success"]
    prompt_modes = sorted({str(t.get("prompt_mode", "")).strip() for t in tasks if str(t.get("prompt_mode", "")).strip()})
    queue_wait_latencies = [float(t.get("queue_wait_ms", 0.0)) for t in successful if t.get("queue_wait_ms") is not None]
    processing_latencies = [float(t.get("proc_whole_ms", t.get("processing_ms", 0.0))) for t in successful if t.get("processing_ms") is not None or t.get("proc_whole_ms") is not None]
    proc_llm_latencies = [float(t.get("proc_llm_ms", 0.0)) for t in successful if t.get("proc_llm_ms") is not None]
    proc_non_llm_latencies = [float(t.get("proc_non_llm_ms", 0.0)) for t in successful if t.get("proc_non_llm_ms") is not None]
    end_to_end_latencies = [float(t.get("end_to_end_ms", t.get("latency_ms", 0.0))) for t in successful]
    llm_latencies = [
        float(t.get("llm_latency_ms", t.get("llm_request_latency_ms", 0.0)))
        for t in successful
        if (t.get("llm_latency_ms") is not None or t.get("llm_request_latency_ms") is not None)
    ]
    usage_missing = any(t.get("tokens_in") is None or t.get("tokens_out") is None for t in successful)
    token_in_total: int | None = None
    token_out_total: int | None = None
    if successful and not usage_missing:
        token_in_total = int(sum(int(t.get("tokens_in", 0)) for t in successful))
        token_out_total = int(sum(int(t.get("tokens_out", 0)) for t in successful))
    injected = [int(t.get("context_injected_tokens", 0)) for t in successful]
    injected_items = [int(t.get("context_injected_items", 0)) for t in successful]
    is_syn = str(workload_id).upper().startswith("SYN")
    accuracy_applicable = not is_syn
    total_tasks = len(tasks)
    accuracy_values = [float(t.get("accuracy")) for t in successful if t.get("accuracy") is not None]
    strict_values = [float(t.get("strict_score")) for t in successful if t.get("strict_score") is not None]
    relaxed_values = [float(t.get("relaxed_score")) for t in successful if t.get("relaxed_score") is not None]
    partial_values = [float(t.get("partial_score")) for t in successful if t.get("partial_score") is not None]
    benchmark_values = [float(t.get("benchmark_score")) for t in successful if t.get("benchmark_score") is not None]
    strict_correct_total = int(
        sum(1 for t in tasks if float(t.get("strict_score", 0.0) or 0.0) >= 1.0)
    )
    relaxed_correct_total = int(
        sum(1 for t in tasks if float(t.get("relaxed_score", 0.0) or 0.0) >= 1.0)
    )
    partial_score_total = float(sum(float(t.get("partial_score", 0.0) or 0.0) for t in tasks))
    benchmark_score_total = float(sum(float(t.get("benchmark_score", 0.0) or 0.0) for t in tasks))
    benchmark_metrics = sorted(
        {
            str(t.get("benchmark_metric", "")).strip()
            for t in tasks
            if str(t.get("benchmark_metric", "")).strip()
        }
    )
    benchmark_metric = benchmark_metrics[0] if len(benchmark_metrics) == 1 else ("mixed" if benchmark_metrics else "strict_exact")
    parse_fail_count = sum(1 for t in successful if bool(t.get("parse_failure", False)))
    no_answer_count = sum(1 for t in successful if bool(t.get("no_answer", False)))
    parse_fail_count_total = sum(1 for t in tasks if bool(t.get("parse_failure", False)))
    no_answer_count_total = sum(1 for t in tasks if bool(t.get("no_answer", False)))
    accuracy: float | None = None if not accuracy_applicable else (sum(accuracy_values) / max(1, len(accuracy_values)))
    strict_accuracy: float | None = None if not accuracy_applicable else (sum(strict_values) / max(1, len(strict_values)))
    relaxed_accuracy: float | None = None if not accuracy_applicable else (sum(relaxed_values) / max(1, len(relaxed_values)))
    partial_accuracy: float | None = None if not accuracy_applicable else (sum(partial_values) / max(1, len(partial_values)))
    benchmark_score: float | None = None if not accuracy_applicable else (sum(benchmark_values) / max(1, len(benchmark_values)))
    strict_accuracy_on_total: float | None = None if not accuracy_applicable else (strict_correct_total / max(1, total_tasks))
    relaxed_accuracy_on_total: float | None = None if not accuracy_applicable else (relaxed_correct_total / max(1, total_tasks))
    partial_accuracy_on_total: float | None = None if not accuracy_applicable else (partial_score_total / max(1, total_tasks))
    benchmark_score_on_total: float | None = None if not accuracy_applicable else (benchmark_score_total / max(1, total_tasks))
    parse_failure_rate = parse_fail_count / max(1, len(successful))
    no_answer_rate = no_answer_count / max(1, len(successful))
    parse_failure_rate_on_total = parse_fail_count_total / max(1, total_tasks)
    no_answer_rate_on_total = no_answer_count_total / max(1, total_tasks)
    success_count = len(successful)
    success_rate = (success_count / total_tasks) if total_tasks > 0 else 0.0

    l1_hits = [m for m in mem_ops if m.get("tier") == "L1" and m.get("hit") is not None]
    l2_hits = [m for m in mem_ops if m.get("tier") == "L2" and m.get("op_name") in {"vector_search", "get_by_keys", "search_text"} and m.get("hit") is not None]
    l3_hits = [m for m in mem_ops if m.get("tier") == "L3" and m.get("hit") is not None]
    l1_hit_rate = sum(1 for x in l1_hits if x.get("hit")) / max(1, len(l1_hits))
    l2_hit_rate = sum(1 for x in l2_hits if x.get("hit")) / max(1, len(l2_hits))
    l3_hit_rate = sum(1 for x in l3_hits if x.get("hit")) / max(1, len(l3_hits))
    memory_ops_per_task = float(len(mem_ops)) / max(1, len(tasks))
    db_query_ops = [m for m in mem_ops if m.get("tier") == "L2" and m.get("op_name") in {"get_by_keys", "search_text", "get_recent"}]
    vector_search_ops = [m for m in mem_ops if m.get("tier") == "L2" and m.get("op_name") == "vector_search"]
    db_queries_per_task = float(len(db_query_ops)) / max(1, len(tasks))
    vector_searches_per_task = float(len(vector_search_ops)) / max(1, len(tasks))

    op_bucket: dict[str, list[float]] = {}
    for op in mem_ops:
        key = f"{op.get('tier','?')}:{op.get('op_name','?')}"
        op_bucket.setdefault(key, []).append(float(op.get("latency_ms", 0.0)))
    op_p95 = {k: summarize_latency(v)["p95"] for k, v in sorted(op_bucket.items())}

    prompt_per_success: float | None = None
    total_per_success: float | None = None
    if token_in_total is not None and token_out_total is not None and success_count > 0:
        prompt_per_success = token_in_total / success_count
        total_per_success = (token_in_total + token_out_total) / success_count

    retrieval_skipped_count = sum(1 for t in successful if t.get("retrieve_skipped_reason"))
    retrieval_skip_rate = retrieval_skipped_count / max(1, len(successful))
    memory_injection_nonzero_tasks = sum(1 for t in successful if int(t.get("context_injected_tokens", 0) or 0) > 0)
    memory_injection_nonzero_rate = memory_injection_nonzero_tasks / max(1, len(successful))
    retrieval_engaged_tasks = sum(1 for t in successful if not t.get("retrieve_skipped_reason"))
    retrieval_engaged_rate = retrieval_engaged_tasks / max(1, len(successful))
    retrieval_engaged_strict_values = [
        float(t.get("strict_score", 0.0))
        for t in successful
        if not t.get("retrieve_skipped_reason")
    ]
    strict_accuracy_retrieval_engaged: float | None = (
        (sum(retrieval_engaged_strict_values) / len(retrieval_engaged_strict_values))
        if retrieval_engaged_strict_values
        else None
    )
    embedding_cache_values = [t.get("embedding_cache_hit") for t in successful if t.get("embedding_cache_hit") is not None]
    retrieval_cache_values = [t.get("retrieval_cache_hit") for t in successful if t.get("retrieval_cache_hit") is not None]
    embedding_cache_hit_rate = (
        sum(1 for v in embedding_cache_values if bool(v)) / len(embedding_cache_values)
        if embedding_cache_values
        else None
    )
    retrieval_cache_hit_rate = (
        sum(1 for v in retrieval_cache_values if bool(v)) / len(retrieval_cache_values)
        if retrieval_cache_values
        else None
    )
    evidence_chars_values = [int(t.get("evidence_chars_used", 0)) for t in successful if t.get("evidence_chars_used") is not None]

    enqueue_ts = [int(t.get("task_enqueued_ts")) for t in tasks if t.get("task_enqueued_ts") is not None]
    end_ts = [int(t.get("task_end_ts", t.get("end_ts"))) for t in tasks if t.get("task_end_ts", t.get("end_ts")) is not None]
    throughput_tasks_per_min = 0.0
    if enqueue_ts and end_ts:
        wall_ms = max(0, max(end_ts) - min(enqueue_ts))
        if wall_ms > 0:
            throughput_tasks_per_min = float(success_count) / (wall_ms / 60000.0)

    injected_summary = summarize_latency(injected) if injected else summarize_latency([])
    injected_items_summary = summarize_latency(injected_items) if injected_items else summarize_latency([])
    queue_wait_summary = summarize_latency(queue_wait_latencies)
    processing_summary = summarize_latency(processing_latencies)
    proc_llm_summary = summarize_latency(proc_llm_latencies)
    proc_non_llm_summary = summarize_latency(proc_non_llm_latencies)
    end_to_end_summary = summarize_latency(end_to_end_latencies)

    def classify_error(msg: str) -> str:
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
        return "other"

    errors_by_category: dict[str, int] = {}
    for row in failed:
        cat = classify_error(str(row.get("error", "")))
        errors_by_category[cat] = int(errors_by_category.get(cat, 0)) + 1
    errors_by_category = dict(sorted(errors_by_category.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "run_id": run_id,
        "variant_id": variant_id,
        "workload_id": workload_id,
        "n_sessions": int(n_sessions),
        "accuracy_applicable": bool(accuracy_applicable),
        "success_tasks": int(success_count),
        "failed_tasks": int(len(failed)),
        "success_rate": float(success_rate),
        "parse_failure_tasks": int(parse_fail_count),
        "no_answer_tasks": int(no_answer_count),
        "throughput_tasks_per_min": float(throughput_tasks_per_min),
        "task_latency": end_to_end_summary,
        "latency": {
            "queue_wait": queue_wait_summary,
            "processing": processing_summary,
            "proc_llm": proc_llm_summary,
            "proc_non_llm": proc_non_llm_summary,
            "end_to_end": end_to_end_summary,
        },
        "llm_request_latency": summarize_latency(llm_latencies),
        "llm_latency": summarize_latency(llm_latencies),
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "relaxed_accuracy": relaxed_accuracy,
        "partial_accuracy": partial_accuracy,
        "benchmark_metric": benchmark_metric,
        "benchmark_score": benchmark_score,
        "strict_accuracy_on_total": strict_accuracy_on_total,
        "relaxed_accuracy_on_total": relaxed_accuracy_on_total,
        "partial_accuracy_on_total": partial_accuracy_on_total,
        "benchmark_score_on_total": benchmark_score_on_total,
        "strict_correct_total": int(strict_correct_total),
        "relaxed_correct_total": int(relaxed_correct_total),
        "parse_failure_rate": float(parse_failure_rate),
        "no_answer_rate": float(no_answer_rate),
        "parse_failure_rate_on_total": float(parse_failure_rate_on_total),
        "no_answer_rate_on_total": float(no_answer_rate_on_total),
        "usage_unavailable": bool(usage_missing),
        "prompt_tokens_total": token_in_total,
        "completion_tokens_total": token_out_total,
        "prompt_tokens_per_success": prompt_per_success,
        "total_tokens_per_success": total_per_success,
        "l1_hit_rate": float(l1_hit_rate),
        "l2_hit_rate": float(l2_hit_rate),
        "l3_hit_rate": float(l3_hit_rate),
        "memory_ops_per_task": float(memory_ops_per_task),
        "db_queries_per_task": float(db_queries_per_task),
        "vector_searches_per_task": float(vector_searches_per_task),
        "retrieval_skip_rate": float(retrieval_skip_rate),
        "retrieval_engaged_rate": float(retrieval_engaged_rate),
        "memory_injection_nonzero_rate": float(memory_injection_nonzero_rate),
        "strict_accuracy_retrieval_engaged": strict_accuracy_retrieval_engaged,
        "embedding_cache_hit_rate": embedding_cache_hit_rate,
        "retrieval_cache_hit_rate": retrieval_cache_hit_rate,
        "context_injected_tokens_mean": float(sum(injected) / max(1, len(injected))),
        "context_injected_items_mean": float(sum(injected_items) / max(1, len(injected_items))),
        "evidence_chars_used_mean": float(sum(evidence_chars_values) / max(1, len(evidence_chars_values))),
        "prompt_modes": prompt_modes,
        "injected_tokens": {
            "mean": float(sum(injected) / max(1, len(injected))),
            "p95": float(injected_summary.get("p95", 0.0)),
        },
        "injected_items": {
            "mean": float(sum(injected_items) / max(1, len(injected_items))),
            "p95": float(injected_items_summary.get("p95", 0.0)),
        },
        "op_p95_ms": op_p95,
        "error_breakdown": {
            "total_errors": int(len(failed)),
            "by_category": errors_by_category,
        },
        "generated_at_ms": now_ms(),
    }


def shuffle_tasks(tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = list(tasks)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
