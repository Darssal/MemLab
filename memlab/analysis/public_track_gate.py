from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from memlab.pipeline.control_plane import (
    accepted_for_official_reporting,
    infer_dataset_id_from_manifest,
    infer_track_id_from_manifest,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))


def _parse_csv_list(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _resolve_baseline_variant(explicit_variant: str, by_variant: dict[str, list[dict[str, Any]]]) -> str:
    variant = str(explicit_variant or "").strip()
    if variant:
        return variant
    if "B1" in by_variant:
        return "B1"
    if "B0" in by_variant:
        return "B0"
    return ""


def _collect_rows(
    runs_dir: Path,
    *,
    workload_id: str,
    dataset_id: str,
    track_id: str,
    model_id: str,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    updated_ts_by_key: dict[tuple[str, int], float] = {}
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        if str(manifest.get("workload_id", "")) != workload_id:
            continue
        if dataset_id and infer_dataset_id_from_manifest(manifest) != dataset_id:
            continue
        if track_id and infer_track_id_from_manifest(manifest) != track_id:
            continue
        if model_id and str(manifest.get("model_id", "")) != model_id:
            continue
        if not accepted_for_official_reporting(manifest):
            continue
        extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
        row = {
            "run_id": str(manifest.get("run_id", run_dir.name)),
            "variant": str(manifest.get("variant_id", "")),
            "repeat_idx": int(manifest.get("repeat_idx", 0)),
            "prompt_mode": str(extra.get("prompt_mode", "")),
            "qa_prompt_profile": str(extra.get("qa_prompt_profile", "")),
            "qa_support_mode": str(extra.get("qa_support_mode", "")),
            "max_evidence_chars": extra.get("max_evidence_chars"),
            "context_top_k": extra.get("context_top_k"),
            "context_item_max_chars": extra.get("context_item_max_chars"),
            "answer_format_mode": str(extra.get("answer_format_mode", "")),
            "benchmark_metric": str(summary.get("benchmark_metric", "")),
            "benchmark_score_on_total": _safe_float(summary.get("benchmark_score_on_total", 0.0)),
            "strict_accuracy_on_total": _safe_float(summary.get("strict_accuracy_on_total", 0.0)),
            "partial_accuracy_on_total": _safe_float(summary.get("partial_accuracy_on_total", 0.0)),
            "parse_failure_rate_on_total": _safe_float(summary.get("parse_failure_rate_on_total", 0.0)),
            "no_answer_rate_on_total": _safe_float(summary.get("no_answer_rate_on_total", 0.0)),
            "success_rate": _safe_float(summary.get("success_rate", 0.0)),
        }
        key = (row["variant"], row["repeat_idx"])
        updated_ts = float(run_dir.stat().st_mtime)
        if updated_ts >= updated_ts_by_key.get(key, -1.0):
            updated_ts_by_key[key] = updated_ts
            rows_by_key[key] = row
    rows = list(rows_by_key.values())
    rows.sort(key=lambda item: (str(item["variant"]), int(item["repeat_idx"])))
    return rows


def _aggregate_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    sub = [row for row in rows if str(row.get("variant", "")) == variant]
    metrics = {}
    for key in (
        "benchmark_score_on_total",
        "strict_accuracy_on_total",
        "partial_accuracy_on_total",
        "parse_failure_rate_on_total",
        "no_answer_rate_on_total",
        "success_rate",
    ):
        values = [_safe_float(row.get(key, 0.0)) for row in sub]
        metrics[key] = {"mean": _mean(values), "std": _std(values), "runs": len(values)}
    return {"variant": variant, "runs": len(sub), "metrics": metrics}


def _build_result(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    track_id: str,
    workload_id: str,
    stage: str,
    expected_variants: list[str],
    expected_repeats: list[int],
    invalid_rate_threshold: float,
    std_threshold: float,
    baseline_variant: str = "",
    must_beat_baseline_variants: list[str] | None = None,
    min_mean_delta: float = 0.0,
    min_repeat_delta: float = 0.0,
    baseline_lower_bound: float | None = None,
    baseline_upper_bound: float | None = None,
) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_modes = sorted({str(row.get("prompt_mode", "")) for row in rows if str(row.get("prompt_mode", "")).strip()})
    protocol_signatures = sorted(
        {
            json.dumps(
                {
                    "prompt_mode": row.get("prompt_mode"),
                    "qa_prompt_profile": row.get("qa_prompt_profile"),
                    "qa_support_mode": row.get("qa_support_mode"),
                    "max_evidence_chars": row.get("max_evidence_chars"),
                    "context_top_k": row.get("context_top_k"),
                    "context_item_max_chars": row.get("context_item_max_chars"),
                    "answer_format_mode": row.get("answer_format_mode"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            for row in rows
        }
    )
    for row in rows:
        by_variant[str(row.get("variant", ""))].append(row)

    missing_variants = [variant for variant in expected_variants if variant not in by_variant]
    missing_repeats: dict[str, list[int]] = {}
    invalid_rate_spikes: dict[str, dict[str, float]] = {}
    variance_warnings: dict[str, float] = {}
    direction_conflicts: dict[str, list[float]] = {}
    ordering_reversals: dict[str, dict[str, Any]] = {}
    baseline_band_violation: dict[str, Any] = {}
    variant_stats = [_aggregate_variant(rows, variant) for variant in expected_variants if variant in by_variant]

    for variant in expected_variants:
        sub = by_variant.get(variant, [])
        if not sub:
            continue
        observed = sorted({int(row.get("repeat_idx", 0)) for row in sub})
        expected_missing = [repeat for repeat in expected_repeats if repeat not in observed]
        if expected_missing:
            missing_repeats[variant] = expected_missing
        parse_rate = _mean([_safe_float(row.get("parse_failure_rate_on_total", 0.0)) for row in sub])
        no_answer_rate = _mean([_safe_float(row.get("no_answer_rate_on_total", 0.0)) for row in sub])
        if parse_rate > invalid_rate_threshold or no_answer_rate > invalid_rate_threshold:
            invalid_rate_spikes[variant] = {
                "parse_failure_rate_on_total_mean": parse_rate,
                "no_answer_rate_on_total_mean": no_answer_rate,
            }
        score_std = _std([_safe_float(row.get("benchmark_score_on_total", 0.0)) for row in sub])
        if len(sub) >= 2 and score_std > std_threshold:
            variance_warnings[variant] = score_std

    baseline_variant = _resolve_baseline_variant(baseline_variant, by_variant)
    if baseline_variant:
        base_by_repeat = {
            int(row.get("repeat_idx", 0)): _safe_float(row.get("benchmark_score_on_total", 0.0))
            for row in by_variant.get(baseline_variant, [])
        }
        base_scores = [_safe_float(row.get("benchmark_score_on_total", 0.0)) for row in by_variant.get(baseline_variant, [])]
        baseline_mean = _mean(base_scores)
        if baseline_lower_bound is not None and baseline_mean < float(baseline_lower_bound):
            baseline_band_violation = {
                "variant": baseline_variant,
                "mean_score": baseline_mean,
                "lower_bound": float(baseline_lower_bound),
                "upper_bound": (None if baseline_upper_bound is None else float(baseline_upper_bound)),
            }
        if baseline_upper_bound is not None and baseline_mean > float(baseline_upper_bound):
            baseline_band_violation = {
                "variant": baseline_variant,
                "mean_score": baseline_mean,
                "lower_bound": (None if baseline_lower_bound is None else float(baseline_lower_bound)),
                "upper_bound": float(baseline_upper_bound),
            }
        for variant in ("V1", "F1", "F2"):
            deltas: list[float] = []
            for row in by_variant.get(variant, []):
                repeat_idx = int(row.get("repeat_idx", 0))
                if repeat_idx not in base_by_repeat:
                    continue
                deltas.append(_safe_float(row.get("benchmark_score_on_total", 0.0)) - base_by_repeat[repeat_idx])
            if len(deltas) >= 2 and any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas):
                direction_conflicts[variant] = deltas
        monitored_variants = [
            variant
            for variant in (must_beat_baseline_variants or [])
            if variant in by_variant and variant != baseline_variant
        ]
        base_mean = _mean([_safe_float(row.get("benchmark_score_on_total", 0.0)) for row in by_variant.get(baseline_variant, [])])
        for variant in monitored_variants:
            variant_scores = [_safe_float(row.get("benchmark_score_on_total", 0.0)) for row in by_variant.get(variant, [])]
            variant_mean = _mean(variant_scores)
            paired_deltas: list[float] = []
            for row in by_variant.get(variant, []):
                repeat_idx = int(row.get("repeat_idx", 0))
                if repeat_idx not in base_by_repeat:
                    continue
                paired_deltas.append(_safe_float(row.get("benchmark_score_on_total", 0.0)) - base_by_repeat[repeat_idx])
            reasons: list[str] = []
            if variant_mean < (base_mean + float(min_mean_delta)):
                reasons.append("mean_delta_below_threshold")
            if paired_deltas and any(delta < float(min_repeat_delta) for delta in paired_deltas):
                reasons.append("paired_repeat_delta_below_threshold")
            if reasons:
                ordering_reversals[variant] = {
                    "baseline_variant": baseline_variant,
                    "baseline_mean": base_mean,
                    "variant_mean": variant_mean,
                    "mean_delta": variant_mean - base_mean,
                    "paired_repeat_deltas": paired_deltas,
                    "min_mean_delta": float(min_mean_delta),
                    "min_repeat_delta": float(min_repeat_delta),
                    "reasons": reasons,
                }

    anomaly_codes: list[str] = []
    if len(prompt_modes) > 1:
        anomaly_codes.append("MIXED_TRACK_COMPARISON")
    if len(protocol_signatures) > 1:
        anomaly_codes.append("PROVIDER_PARAM_DRIFT")
    if missing_variants or missing_repeats:
        anomaly_codes.append("RAW_RUN_TABLE_MISMATCH")
    if invalid_rate_spikes:
        anomaly_codes.append("INVALID_RATE_SPIKE")
    if direction_conflicts:
        anomaly_codes.append("REPEAT_DIRECTION_CONFLICT")
    if variance_warnings:
        anomaly_codes.append("REPEAT_VARIANCE_ABNORMAL")
    if baseline_band_violation:
        anomaly_codes.append("BASELINE_OUT_OF_BAND")
    if ordering_reversals:
        anomaly_codes.append("UNEXPLAINED_ORDERING_REVERSAL")

    hard_block_codes = {
        "RAW_RUN_TABLE_MISMATCH",
        "BASELINE_OUT_OF_BAND",
        "UNEXPLAINED_ORDERING_REVERSAL",
    }
    if missing_variants or missing_repeats or any(code in hard_block_codes for code in anomaly_codes):
        verdict = "invalid"
    elif anomaly_codes:
        verdict = "diagnostic_only"
    else:
        verdict = "accepted"

    return {
        "dataset": dataset,
        "track_id": track_id,
        "workload_id": workload_id,
        "stage": stage,
        "expected_variants": expected_variants,
        "expected_repeats": expected_repeats,
        "rows": len(rows),
        "prompt_modes": prompt_modes,
        "protocol_signatures": protocol_signatures,
        "baseline_variant": baseline_variant,
        "variant_stats": variant_stats,
        "missing_variants": missing_variants,
        "missing_repeats": missing_repeats,
        "invalid_rate_spikes": invalid_rate_spikes,
        "variance_warnings": variance_warnings,
        "direction_conflicts": direction_conflicts,
        "baseline_band_violation": baseline_band_violation,
        "ordering_reversals": ordering_reversals,
        "anomaly_codes": sorted(set(anomaly_codes)),
        "verdict": verdict,
    }


def _write_md(path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# {result['dataset']} {str(result['stage']).upper()} Track Gate")
    lines.append("")
    lines.append(f"- Dataset: `{result['dataset']}`")
    lines.append(f"- Track: `{result['track_id']}`")
    lines.append(f"- Workload: `{result['workload_id']}`")
    lines.append(f"- Verdict: `{result['verdict']}`")
    lines.append(f"- Prompt modes observed: `{', '.join(result.get('prompt_modes', []))}`")
    lines.append(f"- Protocol signatures observed: `{len(result.get('protocol_signatures', []))}`")
    lines.append(f"- Baseline variant: `{result.get('baseline_variant', '')}`")
    lines.append("")
    lines.append("## Variant Aggregate")
    lines.append("")
    lines.append("| variant | runs | benchmark_score_on_total mean+/-std | strict_acc_total mean+/-std | partial_acc_total mean+/-std | parse_fail_total mean | no_answer_total mean | success_rate mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for entry in result.get("variant_stats", []):
        metrics = entry.get("metrics", {})
        lines.append(
            "| {0} | {1} | {2:.4f}+/-{3:.4f} | {4:.4f}+/-{5:.4f} | {6:.4f}+/-{7:.4f} | {8:.4f} | {9:.4f} | {10:.4f} |".format(
                str(entry.get("variant", "")),
                int(entry.get("runs", 0)),
                _safe_float(((metrics.get("benchmark_score_on_total") or {}).get("mean", 0.0))),
                _safe_float(((metrics.get("benchmark_score_on_total") or {}).get("std", 0.0))),
                _safe_float(((metrics.get("strict_accuracy_on_total") or {}).get("mean", 0.0))),
                _safe_float(((metrics.get("strict_accuracy_on_total") or {}).get("std", 0.0))),
                _safe_float(((metrics.get("partial_accuracy_on_total") or {}).get("mean", 0.0))),
                _safe_float(((metrics.get("partial_accuracy_on_total") or {}).get("std", 0.0))),
                _safe_float(((metrics.get("parse_failure_rate_on_total") or {}).get("mean", 0.0))),
                _safe_float(((metrics.get("no_answer_rate_on_total") or {}).get("mean", 0.0))),
                _safe_float(((metrics.get("success_rate") or {}).get("mean", 0.0))),
            )
        )
    lines.append("")
    lines.append("## Gate Findings")
    lines.append("")
    if result.get("missing_variants"):
        lines.append(f"- Missing variants: `{', '.join(result['missing_variants'])}`")
    if result.get("missing_repeats"):
        for variant, repeats in sorted(result["missing_repeats"].items()):
            lines.append(f"- Missing repeats for `{variant}`: `{', '.join(str(x) for x in repeats)}`")
    if result.get("invalid_rate_spikes"):
        for variant, rates in sorted(result["invalid_rate_spikes"].items()):
            lines.append(
                "- Invalid-rate spike for `{0}`: parse_fail_total_mean={1:.4f}, no_answer_total_mean={2:.4f}".format(
                    variant,
                    _safe_float(rates.get("parse_failure_rate_on_total_mean", 0.0)),
                    _safe_float(rates.get("no_answer_rate_on_total_mean", 0.0)),
                )
            )
    if result.get("variance_warnings"):
        for variant, score_std in sorted(result["variance_warnings"].items()):
            lines.append(f"- Variance warning for `{variant}`: benchmark_score_on_total std={score_std:.4f}")
    if result.get("baseline_band_violation"):
        band = result["baseline_band_violation"]
        lines.append(
            "- Baseline out of band for `{0}`: mean_score={1:.4f}, lower_bound={2}, upper_bound={3}".format(
                str(band.get("variant", "")),
                _safe_float(band.get("mean_score", 0.0)),
                ("n/a" if band.get("lower_bound") is None else f"{_safe_float(band.get('lower_bound', 0.0)):.4f}"),
                ("n/a" if band.get("upper_bound") is None else f"{_safe_float(band.get('upper_bound', 0.0)):.4f}"),
            )
        )
    if result.get("ordering_reversals"):
        for variant, payload in sorted(result["ordering_reversals"].items()):
            delta_txt = ", ".join(f"{delta:.4f}" for delta in payload.get("paired_repeat_deltas", []))
            lines.append(
                "- Ordering reversal for `{0}` vs `{1}`: mean_delta={2:.4f}, paired_deltas=`{3}`, reasons=`{4}`".format(
                    variant,
                    str(payload.get("baseline_variant", "")),
                    _safe_float(payload.get("mean_delta", 0.0)),
                    delta_txt,
                    ", ".join(str(x) for x in payload.get("reasons", [])),
                )
            )
    if result.get("direction_conflicts"):
        for variant, deltas in sorted(result["direction_conflicts"].items()):
            delta_txt = ", ".join(f"{delta:.4f}" for delta in deltas)
            lines.append(f"- Direction conflict for `{variant}` vs baseline: `{delta_txt}`")
    if result.get("anomaly_codes"):
        lines.append(f"- Anomaly codes: `{', '.join(result['anomaly_codes'])}`")
    if not result.get("anomaly_codes"):
        lines.append("- No structural anomalies detected at this stage.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--workload-id", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--track-id", type=str, required=True)
    ap.add_argument("--stage", type=str, choices=("smoke", "pilot", "full"), required=True)
    ap.add_argument("--expected-variants", type=str, default="B0,B1,V1,F1,F2")
    ap.add_argument("--expected-repeats", type=str, default="0")
    ap.add_argument("--model-id-filter", type=str, default="")
    ap.add_argument("--invalid-rate-threshold", type=float, default=0.15)
    ap.add_argument("--score-std-threshold", type=float, default=0.10)
    ap.add_argument("--baseline-variant", type=str, default="")
    ap.add_argument("--must-beat-baseline-variants", type=str, default="")
    ap.add_argument("--min-mean-delta", type=float, default=0.0)
    ap.add_argument("--min-repeat-delta", type=float, default=0.0)
    ap.add_argument("--baseline-lower-bound", type=float, default=float("nan"))
    ap.add_argument("--baseline-upper-bound", type=float, default=float("nan"))
    ap.add_argument("--output-md", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    expected_variants = _parse_csv_list(args.expected_variants)
    expected_repeats = [int(token) for token in _parse_csv_list(args.expected_repeats)]
    must_beat_variants = _parse_csv_list(args.must_beat_baseline_variants)
    baseline_lower_bound = None if math.isnan(float(args.baseline_lower_bound)) else float(args.baseline_lower_bound)
    baseline_upper_bound = None if math.isnan(float(args.baseline_upper_bound)) else float(args.baseline_upper_bound)
    rows = _collect_rows(
        Path(args.runs_dir),
        workload_id=str(args.workload_id),
        dataset_id=str(args.dataset),
        track_id=str(args.track_id),
        model_id=str(args.model_id_filter),
    )
    result = _build_result(
        rows,
        dataset=str(args.dataset),
        track_id=str(args.track_id),
        workload_id=str(args.workload_id),
        stage=str(args.stage),
        expected_variants=expected_variants,
        expected_repeats=expected_repeats,
        invalid_rate_threshold=float(args.invalid_rate_threshold),
        std_threshold=float(args.score_std_threshold),
        baseline_variant=str(args.baseline_variant or ""),
        must_beat_baseline_variants=must_beat_variants,
        min_mean_delta=float(args.min_mean_delta),
        min_repeat_delta=float(args.min_repeat_delta),
        baseline_lower_bound=baseline_lower_bound,
        baseline_upper_bound=baseline_upper_bound,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(Path(args.output_md), result)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result.get("verdict") == "accepted" else 2)


if __name__ == "__main__":
    main()
