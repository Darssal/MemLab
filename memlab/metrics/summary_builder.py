from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from memlab.pipeline.control_plane import accepted_for_official_reporting, infer_track_id_from_manifest

from .stats import mean_std_ci95


SUMMARY_FIELDS = [
    "run_id",
    "variant_id",
    "legacy_variant_alias",
    "workload_id",
    "track_id",
    "prompt_mode",
    "tasks_path",
    "preflight_passed",
    "model_base_url",
    "model_id",
    "n_sessions",
    "repeat_idx",
    "seed",
    "processing_p50_ms",
    "processing_p95_ms",
    "processing_p99_ms",
    "queue_wait_p50_ms",
    "queue_wait_p95_ms",
    "queue_wait_p99_ms",
    "end_to_end_p50_ms",
    "end_to_end_p95_ms",
    "end_to_end_p99_ms",
    "proc_llm_p50_ms",
    "proc_llm_p95_ms",
    "proc_llm_p99_ms",
    "proc_non_llm_p50_ms",
    "proc_non_llm_p95_ms",
    "proc_non_llm_p99_ms",
    "task_p50_ms",
    "task_p95_ms",
    "task_p99_ms",
    "throughput_tasks_per_min",
    "success_rate",
    "accuracy",
    "strict_accuracy",
    "relaxed_accuracy",
    "partial_accuracy",
    "benchmark_metric",
    "benchmark_score",
    "strict_accuracy_on_total",
    "relaxed_accuracy_on_total",
    "partial_accuracy_on_total",
    "benchmark_score_on_total",
    "strict_correct_total",
    "relaxed_correct_total",
    "parse_failure_rate",
    "no_answer_rate",
    "parse_failure_rate_on_total",
    "no_answer_rate_on_total",
    "parse_failure_tasks",
    "no_answer_tasks",
    "accuracy_applicable",
    "usage_unavailable",
    "prompt_tokens_total",
    "completion_tokens_total",
    "prompt_tokens_per_success",
    "total_tokens_per_success",
    "retrieval_skip_rate",
    "retrieval_engaged_rate",
    "memory_injection_nonzero_rate",
    "strict_accuracy_retrieval_engaged",
    "embedding_cache_hit_rate",
    "retrieval_cache_hit_rate",
    "l1_hit_rate",
    "l2_hit_rate",
    "l3_hit_rate",
    "memory_ops_per_task",
    "db_queries_per_task",
    "vector_searches_per_task",
    "context_injected_tokens_mean",
    "context_injected_items_mean",
    "success_tasks",
    "failed_tasks",
    "error_total",
    "error_403_balance",
    "error_429_rate_limit",
    "error_400_context_length",
    "error_tool_call_unsupported",
    "error_no_ai_content",
    "error_stream_404",
    "error_model_not_found",
    "retry_count",
    "acceptance_status",
    "diagnostic_only",
    "summary_chain_valid",
    "invalid_reason_codes",
]
MIN_METRICS_SCHEMA_VERSION = 3


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        if txt.lower() == "n/a":
            return None
        return float(txt)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _is_e2e(workload: str) -> bool:
    return workload.upper().startswith("E2E")


def _min_e2e_success_rate() -> float:
    try:
        value = float(os.getenv("MEMLAB_SUMMARY_MIN_E2E_SUCCESS_RATE", "0.60"))
    except Exception:
        value = 0.60
    return max(0.0, min(1.0, value))


def _iter_valid_run_rows(runs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], list[dict[str, Any]]]:
    rows_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    op_p95_by_run: dict[str, dict[str, float]] = {}
    updated_ts_by_key: dict[tuple[str, str, int, int], float] = {}
    excluded_rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        manifest = _load_json(manifest_path)
        summary = _load_json(summary_path)
        run_id = str(manifest.get("run_id", run_dir.name))
        if not accepted_for_official_reporting(manifest):
            excluded_rows.append(
                {
                    "run_id": run_id,
                    "workload_id": str(manifest.get("workload_id", summary.get("workload_id", ""))),
                    "variant_id": str(manifest.get("variant_id", summary.get("variant_id", ""))),
                    "reason": "not_accepted_for_official_reporting",
                    "acceptance_status": str(manifest.get("acceptance_status", "")),
                    "diagnostic_only": bool(manifest.get("diagnostic_only", False)),
                    "summary_chain_valid": manifest.get("summary_chain_valid", None),
                }
            )
            continue
        schema_ver = int(manifest.get("metrics_schema_version", 0))
        if schema_ver < MIN_METRICS_SCHEMA_VERSION:
            excluded_rows.append({"run_id": run_id, "reason": "metrics_schema_too_old"})
            continue
        if not bool(manifest.get("env_valid", False)):
            excluded_rows.append({"run_id": run_id, "reason": "env_invalid"})
            continue
        if not bool(manifest.get("repro_valid", False)):
            excluded_rows.append({"run_id": run_id, "reason": "repro_invalid"})
            continue
        workload = str(manifest.get("workload_id", summary.get("workload_id", "")))
        extra = manifest.get("extra", {}) if isinstance(manifest.get("extra", {}), dict) else {}
        if _is_e2e(workload):
            if bool(extra.get("debug_print", False)):
                excluded_rows.append({"run_id": run_id, "reason": "debug_print_run"})
                continue
            tasks_path = str(extra.get("tasks_path", "")).replace("\\", "/")
            if tasks_path and not tasks_path.startswith("workloads/e2e/"):
                excluded_rows.append({"run_id": run_id, "reason": "tasks_path_not_runtime"})
                continue
        if _is_e2e(workload) and not bool(manifest.get("llm_real", False)):
            excluded_rows.append({"run_id": run_id, "reason": "llm_not_real"})
            continue
        success_rate = float(summary.get("success_rate", 0.0))
        if _is_e2e(workload) and success_rate < _min_e2e_success_rate():
            excluded_rows.append({"run_id": run_id, "reason": "success_rate_below_threshold"})
            continue
        latency = summary.get("latency", {}) if isinstance(summary.get("latency"), dict) else {}
        queue_wait = latency.get("queue_wait", {}) if isinstance(latency.get("queue_wait"), dict) else {}
        processing = latency.get("processing", {}) if isinstance(latency.get("processing"), dict) else {}
        proc_llm = latency.get("proc_llm", {}) if isinstance(latency.get("proc_llm"), dict) else {}
        proc_non_llm = latency.get("proc_non_llm", {}) if isinstance(latency.get("proc_non_llm"), dict) else {}
        end_to_end = latency.get("end_to_end", {}) if isinstance(latency.get("end_to_end"), dict) else {}
        if not end_to_end and isinstance(summary.get("task_latency"), dict):
            end_to_end = summary.get("task_latency", {})
        accuracy_value = _as_float_or_none(summary.get("accuracy"))
        strict_accuracy = _as_float_or_none(summary.get("strict_accuracy"))
        relaxed_accuracy = _as_float_or_none(summary.get("relaxed_accuracy"))
        benchmark_score = _as_float_or_none(summary.get("benchmark_score"))
        strict_accuracy_on_total = _as_float_or_none(summary.get("strict_accuracy_on_total"))
        relaxed_accuracy_on_total = _as_float_or_none(summary.get("relaxed_accuracy_on_total"))
        benchmark_score_on_total = _as_float_or_none(summary.get("benchmark_score_on_total"))
        if strict_accuracy_on_total is None:
            strict_accuracy_on_total = strict_accuracy
        if relaxed_accuracy_on_total is None:
            relaxed_accuracy_on_total = relaxed_accuracy
        accuracy_applicable = bool(summary.get("accuracy_applicable", not workload.upper().startswith("SYN")))
        if not accuracy_applicable:
            accuracy_value = None
            strict_accuracy = None
            relaxed_accuracy = None
            benchmark_score = None
            strict_accuracy_on_total = None
            relaxed_accuracy_on_total = None
            benchmark_score_on_total = None
        tasks_path = str(extra.get("tasks_path", "")).replace("\\", "/")
        prompt_mode = str(extra.get("prompt_mode", "memory_neutral") or "memory_neutral")
        raw_variant_id = str(manifest.get("variant_id", summary.get("variant_id", "")))
        variant_id = {"A1": "V1", "A2": "F1"}.get(raw_variant_id, raw_variant_id)
        legacy_variant_alias = str(extra.get("legacy_variant_alias") or "")
        if not legacy_variant_alias and raw_variant_id in {"A1", "A2"}:
            legacy_variant_alias = raw_variant_id
        track_id = infer_track_id_from_manifest(manifest)
        error_breakdown = extra.get("error_breakdown", summary.get("error_breakdown", {}))
        by_category = error_breakdown.get("by_category", {}) if isinstance(error_breakdown, dict) else {}
        if not isinstance(by_category, dict):
            by_category = {}
        row = {
            "run_id": manifest.get("run_id", run_dir.name),
            "variant_id": variant_id,
            "legacy_variant_alias": legacy_variant_alias,
            "workload_id": workload,
            "track_id": track_id,
            "prompt_mode": prompt_mode,
            "tasks_path": tasks_path,
            "preflight_passed": bool(extra.get("preflight_passed", False)),
            "model_base_url": manifest.get("model_base_url") or "",
            "model_id": manifest.get("model_id") or "",
            "n_sessions": int(summary.get("n_sessions", 0)),
            "repeat_idx": int(manifest.get("repeat_idx", 0)),
            "seed": int(manifest.get("seed", 0)),
            "processing_p50_ms": float(processing.get("p50", 0.0)),
            "processing_p95_ms": float(processing.get("p95", 0.0)),
            "processing_p99_ms": float(processing.get("p99", 0.0)),
            "queue_wait_p50_ms": float(queue_wait.get("p50", 0.0)),
            "queue_wait_p95_ms": float(queue_wait.get("p95", 0.0)),
            "queue_wait_p99_ms": float(queue_wait.get("p99", 0.0)),
            "end_to_end_p50_ms": float(end_to_end.get("p50", 0.0)),
            "end_to_end_p95_ms": float(end_to_end.get("p95", 0.0)),
            "end_to_end_p99_ms": float(end_to_end.get("p99", 0.0)),
            "proc_llm_p50_ms": float(proc_llm.get("p50", 0.0)),
            "proc_llm_p95_ms": float(proc_llm.get("p95", 0.0)),
            "proc_llm_p99_ms": float(proc_llm.get("p99", 0.0)),
            "proc_non_llm_p50_ms": float(proc_non_llm.get("p50", 0.0)),
            "proc_non_llm_p95_ms": float(proc_non_llm.get("p95", 0.0)),
            "proc_non_llm_p99_ms": float(proc_non_llm.get("p99", 0.0)),
            "task_p50_ms": float(end_to_end.get("p50", 0.0)),
            "task_p95_ms": float(end_to_end.get("p95", 0.0)),
            "task_p99_ms": float(end_to_end.get("p99", 0.0)),
            "throughput_tasks_per_min": float(summary.get("throughput_tasks_per_min", 0.0)),
            "success_rate": success_rate,
            "accuracy": accuracy_value,
            "strict_accuracy": strict_accuracy,
            "relaxed_accuracy": relaxed_accuracy,
            "partial_accuracy": _as_float_or_none(summary.get("partial_accuracy")),
            "benchmark_metric": str(summary.get("benchmark_metric", "")),
            "benchmark_score": benchmark_score,
            "strict_accuracy_on_total": strict_accuracy_on_total,
            "relaxed_accuracy_on_total": relaxed_accuracy_on_total,
            "partial_accuracy_on_total": _as_float_or_none(summary.get("partial_accuracy_on_total")),
            "benchmark_score_on_total": benchmark_score_on_total,
            "strict_correct_total": int(summary.get("strict_correct_total", 0)),
            "relaxed_correct_total": int(summary.get("relaxed_correct_total", 0)),
            "parse_failure_rate": float(summary.get("parse_failure_rate", 0.0)),
            "no_answer_rate": float(summary.get("no_answer_rate", 0.0)),
            "parse_failure_rate_on_total": float(summary.get("parse_failure_rate_on_total", 0.0)),
            "no_answer_rate_on_total": float(summary.get("no_answer_rate_on_total", 0.0)),
            "parse_failure_tasks": int(summary.get("parse_failure_tasks", 0)),
            "no_answer_tasks": int(summary.get("no_answer_tasks", 0)),
            "accuracy_applicable": accuracy_applicable,
            "usage_unavailable": bool(manifest.get("usage_unavailable", summary.get("usage_unavailable", False))),
            "prompt_tokens_total": _as_float_or_none(summary.get("prompt_tokens_total")),
            "completion_tokens_total": _as_float_or_none(summary.get("completion_tokens_total")),
            "prompt_tokens_per_success": _as_float_or_none(summary.get("prompt_tokens_per_success")),
            "total_tokens_per_success": _as_float_or_none(summary.get("total_tokens_per_success")),
            "retrieval_skip_rate": float(summary.get("retrieval_skip_rate", 0.0)),
            "retrieval_engaged_rate": float(summary.get("retrieval_engaged_rate", 0.0)),
            "memory_injection_nonzero_rate": float(summary.get("memory_injection_nonzero_rate", 0.0)),
            "strict_accuracy_retrieval_engaged": _as_float_or_none(summary.get("strict_accuracy_retrieval_engaged")),
            "embedding_cache_hit_rate": _as_float_or_none(summary.get("embedding_cache_hit_rate")),
            "retrieval_cache_hit_rate": _as_float_or_none(summary.get("retrieval_cache_hit_rate")),
            "l1_hit_rate": float(summary.get("l1_hit_rate", 0.0)),
            "l2_hit_rate": float(summary.get("l2_hit_rate", 0.0)),
            "l3_hit_rate": float(summary.get("l3_hit_rate", 0.0)),
            "memory_ops_per_task": float(summary.get("memory_ops_per_task", 0.0)),
            "db_queries_per_task": float(summary.get("db_queries_per_task", 0.0)),
            "vector_searches_per_task": float(summary.get("vector_searches_per_task", 0.0)),
            "context_injected_tokens_mean": float(summary.get("context_injected_tokens_mean", 0.0)),
            "context_injected_items_mean": float(summary.get("context_injected_items_mean", 0.0)),
            "success_tasks": int(summary.get("success_tasks", 0)),
            "failed_tasks": int(summary.get("failed_tasks", 0)),
            "error_total": int(error_breakdown.get("total_errors", summary.get("failed_tasks", 0)) if isinstance(error_breakdown, dict) else int(summary.get("failed_tasks", 0))),
            "error_403_balance": int(by_category.get("provider_403_balance", 0)),
            "error_429_rate_limit": int(by_category.get("provider_429_rate_limit", 0)),
            "error_400_context_length": int(by_category.get("provider_400_context_length", 0)),
            "error_tool_call_unsupported": int(by_category.get("tool_call_unsupported", 0)),
            "error_no_ai_content": int(by_category.get("deerflow_no_ai_content", 0)),
            "error_stream_404": int(by_category.get("deerflow_stream_404", 0)),
            "error_model_not_found": int(by_category.get("model_not_found", 0)),
            "retry_count": int(extra.get("retry_count", summary.get("retry_count_total", 0))),
            "acceptance_status": str(manifest.get("acceptance_status", "")),
            "diagnostic_only": bool(manifest.get("diagnostic_only", False)),
            "summary_chain_valid": bool(manifest.get("summary_chain_valid", False)),
            "invalid_reason_codes": ",".join([str(x) for x in manifest.get("invalid_reason_codes", [])]),
        }
        key = (
            str(row["variant_id"]),
            str(row["workload_id"]),
            int(row["repeat_idx"]),
            int(row["seed"]),
        )
        updated_ts = float(run_dir.stat().st_mtime)
        prev_ts = updated_ts_by_key.get(key, -1.0)
        if updated_ts >= prev_ts:
            updated_ts_by_key[key] = updated_ts
            rows_by_key[key] = row
        op_p95 = summary.get("op_p95_ms", {}) or {}
        if isinstance(op_p95, dict):
            op_p95_by_run[row["run_id"]] = {str(k): float(v) for k, v in op_p95.items()}
    rows = sorted(
        rows_by_key.values(),
        key=lambda r: (str(r["workload_id"]), str(r["variant_id"]), int(r["repeat_idx"]), int(r["seed"])),
    )
    return rows, op_p95_by_run, excluded_rows


def build_acceptance_result(
    rows: list[dict[str, Any]],
    *,
    min_p95_improve: float = 0.03,
    min_prompt_drop: float = 0.03,
    max_accuracy_drop: float = 0.03,
) -> dict[str, Any]:
    e2e_rows = [
        r
        for r in rows
        if _is_e2e(str(r.get("workload_id", "")))
        and str(r.get("prompt_mode", "memory_neutral")) == "memory_neutral"
        and str(r.get("tasks_path", "")).replace("\\", "/").endswith("tasks.jsonl")
    ]
    if not e2e_rows:
        e2e_rows = [r for r in rows if _is_e2e(str(r.get("workload_id", "")))]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in e2e_rows:
        by_variant[str(row.get("variant_id", ""))].append(row)

    f1 = by_variant.get("F1", [])
    b1 = by_variant.get("B1", [])
    if not f1 or not b1:
        return {
            "status": "MISSING",
            "reason": "Missing F1 or B1 valid E2E rows",
        }

    f1_p95 = _mean([float(r["processing_p95_ms"]) for r in f1])
    b1_p95 = _mean([float(r["processing_p95_ms"]) for r in b1])
    f1_token_subset = [r for r in f1 if not bool(r.get("usage_unavailable"))]
    b1_token_subset = [r for r in b1 if not bool(r.get("usage_unavailable"))]
    f1_prompt = _mean_optional([_as_float_or_none(r.get("prompt_tokens_per_success")) for r in f1_token_subset])
    b1_prompt = _mean_optional([_as_float_or_none(r.get("prompt_tokens_per_success")) for r in b1_token_subset])

    def _pick_accuracy(row: dict[str, Any]) -> float | None:
        val = _as_float_or_none(row.get("strict_accuracy_on_total"))
        if val is not None:
            return val
        val = _as_float_or_none(row.get("strict_accuracy"))
        if val is not None:
            return val
        return _as_float_or_none(row.get("accuracy"))

    # Use total-task denominator as the primary accuracy lens for memory-effect evaluation.
    f1_acc = _mean_optional([_pick_accuracy(r) for r in f1])
    b1_acc = _mean_optional([_pick_accuracy(r) for r in b1])
    if f1_acc is None or b1_acc is None:
        return {
            "status": "MISSING",
            "reason": "Missing E2E accuracy rows for F1 or B1",
        }

    p95_improve = ((b1_p95 - f1_p95) / b1_p95) if b1_p95 > 0 else 0.0
    prompt_drop = None
    if b1_prompt is not None and f1_prompt is not None and b1_prompt > 0:
        prompt_drop = (b1_prompt - f1_prompt) / b1_prompt
    acc_drop = b1_acc - f1_acc

    pass_latency = p95_improve >= float(min_p95_improve)
    pass_tokens = bool(
        prompt_drop is not None
        and prompt_drop >= float(min_prompt_drop)
        and acc_drop <= float(max_accuracy_drop)
    )
    accepted = pass_latency or pass_tokens
    return {
        "status": "PASS" if accepted else "FAIL",
        "latency_metric": "processing_p95_ms",
        "accuracy_metric": "strict_accuracy_on_total",
        "p95_improve": p95_improve,
        "prompt_drop": prompt_drop,
        "token_subset_runs_f1": len(f1_token_subset),
        "token_subset_runs_b1": len(b1_token_subset),
        "accuracy_drop": acc_drop,
        "thresholds": {
            "min_p95_improve": float(min_p95_improve),
            "min_prompt_drop": float(min_prompt_drop),
            "max_accuracy_drop": float(max_accuracy_drop),
        },
        "pass_latency": pass_latency,
        "pass_tokens": pass_tokens,
    }


def write_experiment_report(
    rows: list[dict[str, Any]],
    *,
    acceptance: dict[str, Any],
    op_p95_by_run: dict[str, dict[str, float]],
    excluded_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r["workload_id"]), str(r["variant_id"]))].append(r)

    workloads = sorted({str(r["workload_id"]) for r in rows})
    variants = ["B0", "B1", "V1", "F1", "F2"]
    lines: list[str] = []
    lines.append("# Experiment Report")
    lines.append("")
    usage_unknown_count = sum(1 for r in rows if str(r.get("workload_id", "")).upper().startswith("E2E") and bool(r.get("usage_unavailable")))
    lines.append("## Metric Validity")
    lines.append("")
    lines.append(f"- Comparable rows in summary.csv: {len(rows)}")
    lines.append(
        "- E2E comparability requires: "
        f"env_valid=true, repro_valid=true, llm_real=true, success_rate>={_min_e2e_success_rate():.2f}."
    )
    lines.append(f"- E2E rows with usage unavailable (tokens shown as N/A): {usage_unknown_count}")
    lines.append("")
    lines.append("## Excluded Runs")
    lines.append("")
    lines.append(f"- Excluded rows from official summary: {len(excluded_rows)}")
    for row in excluded_rows[:20]:
        lines.append(
            "- `{run_id}`: {reason} (acceptance_status={acceptance_status}, diagnostic_only={diagnostic_only}, summary_chain_valid={summary_chain_valid})".format(
                run_id=str(row.get("run_id", "")),
                reason=str(row.get("reason", "")),
                acceptance_status=str(row.get("acceptance_status", "")),
                diagnostic_only=str(row.get("diagnostic_only", "")),
                summary_chain_valid=str(row.get("summary_chain_valid", "")),
            )
        )
    lines.append("")

    if not rows:
        lines.append(
            "No comparable runs found "
            f"(`env_valid=true`, `repro_valid=true`, `llm_real=true`, and E2E `success_rate>={_min_e2e_success_rate():.2f}`)."
        )
        lines.append("")
    else:
        for workload in workloads:
            lines.append(f"## {workload}")
            lines.append("")
            lines.append("| Variant | proc_whole p95 (ms) | proc_llm p95 (ms) | proc_non_llm p95 (ms) | end-to-end p95 (ms) | throughput (/min) | tokens/success | strict_acc(success) | strict_acc(total) | relaxed_acc(total) | parse_fail_rate(total) | no_answer_rate(total) | retrieval_engaged | injected_nonzero | L1 hit | L2 hit | L3 hit | mem_ops/task |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for v in variants:
                g = grouped.get((workload, v), [])
                if not g:
                    continue
                token_values = [_as_float_or_none(x.get("total_tokens_per_success")) for x in g]
                strict_acc_values = [_as_float_or_none(x.get("strict_accuracy", x.get("accuracy"))) for x in g]
                strict_acc_total_values = [_as_float_or_none(x.get("strict_accuracy_on_total")) for x in g]
                relaxed_acc_total_values = [_as_float_or_none(x.get("relaxed_accuracy_on_total")) for x in g]
                token_mean = _mean_optional(token_values)
                strict_acc_mean = _mean_optional(strict_acc_values)
                strict_acc_total_mean = _mean_optional(strict_acc_total_values)
                relaxed_acc_total_mean = _mean_optional(relaxed_acc_total_values)
                token_txt = "N/A" if token_mean is None else f"{token_mean:.1f}"
                strict_acc_txt = "N/A" if strict_acc_mean is None else f"{strict_acc_mean:.4f}"
                strict_acc_total_txt = "N/A" if strict_acc_total_mean is None else f"{strict_acc_total_mean:.4f}"
                relaxed_acc_total_txt = "N/A" if relaxed_acc_total_mean is None else f"{relaxed_acc_total_mean:.4f}"
                lines.append(
                    "| {0} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.2f} | {6} | {7} | {8} | {9} | {10:.2%} | {11:.2%} | {12:.2%} | {13:.2%} | {14:.4f} | {15:.4f} | {16:.4f} | {17:.3f} |".format(
                        v,
                        _mean([float(x["processing_p95_ms"]) for x in g]),
                        _mean([float(x.get("proc_llm_p95_ms", 0.0)) for x in g]),
                        _mean([float(x.get("proc_non_llm_p95_ms", 0.0)) for x in g]),
                        _mean([float(x["end_to_end_p95_ms"]) for x in g]),
                        _mean([float(x.get("throughput_tasks_per_min", 0.0)) for x in g]),
                        token_txt,
                        strict_acc_txt,
                        strict_acc_total_txt,
                        relaxed_acc_total_txt,
                        _mean([float(x.get("parse_failure_rate_on_total", 0.0)) for x in g]),
                        _mean([float(x.get("no_answer_rate_on_total", 0.0)) for x in g]),
                        _mean([float(x.get("retrieval_engaged_rate", 0.0)) for x in g]),
                        _mean([float(x.get("memory_injection_nonzero_rate", 0.0)) for x in g]),
                        _mean([float(x.get("l1_hit_rate", 0.0)) for x in g]),
                        _mean([float(x.get("l2_hit_rate", 0.0)) for x in g]),
                        _mean([float(x.get("l3_hit_rate", 0.0)) for x in g]),
                        _mean([float(x.get("memory_ops_per_task", 0.0)) for x in g]),
                    )
                )
            lines.append("")

    lines.append("## Acceptance Check (F1 vs B1, E2E)")
    lines.append("")
    if acceptance.get("status") == "MISSING":
        lines.append(f"- Status: MISSING ({acceptance.get('reason', '')})")
    else:
        lines.append(f"- Status: {acceptance.get('status')}")
        thresholds = acceptance.get("thresholds", {}) if isinstance(acceptance.get("thresholds"), dict) else {}
        try:
            min_p95_improve = float(thresholds.get("min_p95_improve", 0.03))
        except Exception:
            min_p95_improve = 0.03
        try:
            min_prompt_drop = float(thresholds.get("min_prompt_drop", 0.03))
        except Exception:
            min_prompt_drop = 0.03
        try:
            max_accuracy_drop = float(thresholds.get("max_accuracy_drop", 0.03))
        except Exception:
            max_accuracy_drop = 0.03
        lines.append(
            "- acceptance thresholds: "
            f"p95_improve>={min_p95_improve:.2%}, "
            f"prompt_drop>={min_prompt_drop:.2%}, "
            f"accuracy_drop<={max_accuracy_drop:.2%}"
        )
        lines.append(f"- primary latency metric: {acceptance.get('latency_metric', 'processing_p95_ms')}")
        lines.append(f"- primary accuracy metric: {acceptance.get('accuracy_metric', 'strict_accuracy_on_total')}")
        lines.append(f"- processing p95 reduction: {float(acceptance.get('p95_improve', 0.0)):.2%}")
        prompt_drop = acceptance.get("prompt_drop")
        if prompt_drop is None:
            lines.append("- prompt token reduction: N/A (usage unavailable)")
        else:
            lines.append(f"- prompt token reduction: {float(prompt_drop):.2%}")
        lines.append(f"- accuracy drop (B1 - F1): {float(acceptance.get('accuracy_drop', 0.0)):.2%}")
    lines.append("")

    if acceptance.get("status") == "FAIL":
        lines.append("## Diagnosis")
        lines.append("")
        f1_runs = [r["run_id"] for r in rows if r.get("variant_id") == "F1" and _is_e2e(str(r.get("workload_id", "")))]
        op_buckets: dict[str, list[float]] = defaultdict(list)
        for rid in f1_runs:
            for op_name, p95 in op_p95_by_run.get(rid, {}).items():
                op_buckets[op_name].append(float(p95))
        if not op_buckets:
            lines.append("- No op-level p95 data available for diagnosis.")
        else:
            ranked = sorted(((k, _mean(v)) for k, v in op_buckets.items()), key=lambda x: x[1], reverse=True)
            lines.append("- Highest p95 components in F1:")
            for name, value in ranked[:10]:
                lines.append(f"  - {name}: {value:.3f} ms")
        lines.append("")
        lines.append("## Ablation Notes")
        lines.append("")
        f1_rows = [r for r in rows if r.get("variant_id") == "F1" and _is_e2e(str(r.get("workload_id", "")))]
        if f1_rows:
            lines.append(f"- Mean retrieval skip rate (F1 E2E): {_mean([float(x.get('retrieval_skip_rate', 0.0)) for x in f1_rows]):.2%}.")
            emb_hits = _mean_optional([_as_float_or_none(x.get("embedding_cache_hit_rate")) for x in f1_rows])
            ret_hits = _mean_optional([_as_float_or_none(x.get("retrieval_cache_hit_rate")) for x in f1_rows])
            if emb_hits is not None:
                lines.append(f"- Mean embedding cache hit rate (F1 E2E): {emb_hits:.2%}.")
            if ret_hits is not None:
                lines.append(f"- Mean retrieval cache hit rate (F1 E2E): {ret_hits:.2%}.")
        lines.append("- Injection controls should reduce context_injected_tokens_mean in F1.")
        lines.append("- Retrieval gating/caching should reduce p95 of L2:vector_search and L2:get_by_keys.")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_runs(
    *,
    runs_dir: str | Path = "runs",
    summary_csv: str | Path = "summary.csv",
    stats_json: str | Path = "summary.stats.json",
    report_md: str | Path = "experiment_report.md",
    report_alias_md: str | Path = "report.md",
    min_p95_improve: float = 0.03,
    min_prompt_drop: float = 0.03,
    max_accuracy_drop: float = 0.03,
) -> dict[str, Any]:
    runs_path = Path(runs_dir)
    rows, op_p95_by_run, excluded_rows = _iter_valid_run_rows(runs_path)

    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload_id"]), str(row["variant_id"]))].append(float(row["processing_p95_ms"]))
    stats = {
        f"{w}:{v}": mean_std_ci95(vals)
        for (w, v), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1]))
    }
    stats_path = Path(stats_json)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    acceptance = build_acceptance_result(
        rows,
        min_p95_improve=float(min_p95_improve),
        min_prompt_drop=float(min_prompt_drop),
        max_accuracy_drop=float(max_accuracy_drop),
    )
    report_path = Path(report_md)
    write_experiment_report(
        rows,
        acceptance=acceptance,
        op_p95_by_run=op_p95_by_run,
        excluded_rows=excluded_rows,
        output_path=report_path,
    )
    Path(report_alias_md).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "rows": len(rows),
        "excluded_rows": len(excluded_rows),
        "summary_csv": str(summary_path),
        "summary_stats_json": str(stats_path),
        "experiment_report": str(report_path),
        "report_md": str(Path(report_alias_md)),
        "acceptance": acceptance,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--summary-csv", type=str, default="summary.csv")
    ap.add_argument("--stats-json", type=str, default="summary.stats.json")
    ap.add_argument("--report-md", type=str, default="experiment_report.md")
    ap.add_argument("--report-alias-md", type=str, default="report.md")
    ap.add_argument("--min-p95-improve", type=float, default=0.03)
    ap.add_argument("--min-prompt-drop", type=float, default=0.03)
    ap.add_argument("--max-accuracy-drop", type=float, default=0.03)
    args = ap.parse_args()
    result = aggregate_runs(
        runs_dir=args.runs_dir,
        summary_csv=args.summary_csv,
        stats_json=args.stats_json,
        report_md=args.report_md,
        report_alias_md=args.report_alias_md,
        min_p95_improve=float(args.min_p95_improve),
        min_prompt_drop=float(args.min_prompt_drop),
        max_accuracy_drop=float(args.max_accuracy_drop),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
