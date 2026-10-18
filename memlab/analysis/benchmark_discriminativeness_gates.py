from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memlab.pipeline.control_plane import accepted_for_official_reporting, infer_track_id_from_manifest


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _parse_dataset(tasks_path: str) -> str:
    norm = str(tasks_path or "").replace("\\", "/")
    if "/public_main/" not in norm:
        return "unknown"
    parts = norm.split("/public_main/", 1)[1].split("/")
    return parts[0] if parts else "unknown"


def _is_comparable(row: dict[str, Any]) -> bool:
    return bool(row.get("env_valid")) and bool(row.get("repro_valid")) and bool(row.get("llm_real"))


def _load_null_map(path: str) -> dict[str, str]:
    if not str(path or "").strip():
        return {}
    raw = _read_json(Path(path))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.strip():
            out[str(key).strip()] = str(value).strip()
        elif isinstance(value, dict):
            workload_id = str(value.get("workload_id", "") or "").strip()
            if workload_id:
                out[str(key).strip()] = workload_id
    return out


def _collect_rows(
    runs_dir: Path,
    *,
    workload_id: str = "",
    dataset_filter: str = "",
    model_id_filter: str = "",
    experiment_track_filter: str = "",
    track_id_filter: str = "",
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    updated_ts_by_key: dict[tuple[str, str, str, int], float] = {}
    if not runs_dir.exists():
        return []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        try:
            manifest = _read_json(manifest_path)
            summary = _read_json(summary_path)
        except Exception:
            continue
        wid = str(manifest.get("workload_id", "") or "")
        if not wid or "PREFLIGHT" in wid:
            continue
        if workload_id and wid != workload_id:
            continue
        extra = manifest.get("extra", {}) if isinstance(manifest.get("extra"), dict) else {}
        tasks_path = str(extra.get("tasks_path", "") or "")
        dataset = str(extra.get("dataset_id", "") or "").strip() or _parse_dataset(tasks_path)
        if dataset_filter and dataset != dataset_filter:
            continue
        model_id = str(manifest.get("model_id", "") or "")
        if model_id_filter and model_id != model_id_filter:
            continue
        if not accepted_for_official_reporting(manifest):
            continue
        experiment_track = str(extra.get("experiment_track", "mainline") or "mainline")
        if experiment_track_filter and experiment_track != experiment_track_filter:
            continue
        track_id = infer_track_id_from_manifest(manifest)
        if track_id_filter and track_id != track_id_filter:
            continue
        repeat_idx = int(manifest.get("repeat_idx", 0))
        variant = str(manifest.get("variant_id", "") or "")
        row = {
            "run_dir": str(run_dir),
            "mtime": float(run_dir.stat().st_mtime),
            "dataset": dataset,
            "workload_id": wid,
            "track_id": track_id,
            "variant": variant,
            "repeat_idx": repeat_idx,
            "seed": int(manifest.get("seed", 0)),
            "model_id": model_id,
            "experiment_track": experiment_track,
            "env_valid": bool(manifest.get("env_valid", False)),
            "repro_valid": bool(manifest.get("repro_valid", False)),
            "llm_real": bool(manifest.get("llm_real", False)),
            "success_rate": _safe_float(summary.get("success_rate", 0.0)),
            "strict_accuracy": _safe_float(summary.get("strict_accuracy", summary.get("accuracy", 0.0))),
            "relaxed_accuracy": _safe_float(summary.get("relaxed_accuracy", summary.get("accuracy", 0.0))),
            "benchmark_metric": str(summary.get("benchmark_metric", "") or ""),
            "benchmark_score": _safe_float(
                summary.get(
                    "benchmark_score_on_total",
                    summary.get("benchmark_score", summary.get("strict_accuracy", summary.get("accuracy", 0.0))),
                )
            ),
            "parse_failure_rate": _safe_float(summary.get("parse_failure_rate", 0.0)),
            "no_answer_rate": _safe_float(summary.get("no_answer_rate", 0.0)),
            "processing_p95_ms": _safe_float(((summary.get("latency") or {}).get("processing") or {}).get("p95", 0.0)),
            "end_to_end_p95_ms": _safe_float(((summary.get("latency") or {}).get("end_to_end") or {}).get("p95", 0.0)),
        }
        key = (wid, track_id, variant, repeat_idx)
        mtime = float(run_dir.stat().st_mtime)
        if mtime >= updated_ts_by_key.get(key, -1.0):
            updated_ts_by_key[key] = mtime
            rows_by_key[key] = row
    rows = list(rows_by_key.values())
    rows.sort(key=lambda item: (str(item.get("dataset", "")), str(item.get("workload_id", "")), str(item.get("track_id", "")), str(item.get("variant", "")), int(item.get("repeat_idx", 0))))
    return rows


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    sub = [r for r in rows if str(r.get("variant", "")) == variant]
    return {
        "variant": variant,
        "runs": len(sub),
        "benchmark_metric": str(sub[0].get("benchmark_metric", "")) if sub else "",
        "benchmark_score": _mean([_safe_float(r.get("benchmark_score", 0.0)) for r in sub]) if sub else 0.0,
        "success_rate": _mean([_safe_float(r.get("success_rate", 0.0)) for r in sub]) if sub else 0.0,
        "strict_accuracy": _mean([_safe_float(r.get("strict_accuracy", 0.0)) for r in sub]) if sub else 0.0,
        "relaxed_accuracy": _mean([_safe_float(r.get("relaxed_accuracy", 0.0)) for r in sub]) if sub else 0.0,
        "parse_failure_rate": _mean([_safe_float(r.get("parse_failure_rate", 0.0)) for r in sub]) if sub else 0.0,
        "no_answer_rate": _mean([_safe_float(r.get("no_answer_rate", 0.0)) for r in sub]) if sub else 0.0,
        "processing_p95_ms": _mean([_safe_float(r.get("processing_p95_ms", 0.0)) for r in sub]) if sub else 0.0,
        "end_to_end_p95_ms": _mean([_safe_float(r.get("end_to_end_p95_ms", 0.0)) for r in sub]) if sub else 0.0,
        "comparable_rate": _mean([1.0 if _is_comparable(r) else 0.0 for r in sub]) if sub else 0.0,
    }


def _is_main_smoke_workload(workload_id: str, dataset: str) -> bool:
    wid = str(workload_id or "").upper()
    prefix = f"PUB-{str(dataset).upper()}-SMOKE"
    return wid.startswith(prefix) and "NULLSANITY" not in wid and "MEMEXP" not in wid and "ALIGN" not in wid


def _find_latest_main_smoke_workload(rows: list[dict[str, Any]], dataset: str) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        wid = str(row.get("workload_id", "") or "")
        if not _is_main_smoke_workload(wid, dataset):
            continue
        grouped[wid].append(row)
    candidates: list[tuple[float, str, bool]] = []
    for wid, group_rows in grouped.items():
        latest_mtime = max(_safe_float(r.get("mtime", 0.0)) for r in group_rows)
        comparable_variants = {str(r.get("variant", "")) for r in group_rows if _is_comparable(r)}
        candidates.append((latest_mtime, wid, {"B1", "F1"}.issubset(comparable_variants)))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[2], item[0], item[1]), reverse=True)
    return str(candidates[0][1])


def _discover_null_workload(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    mapped_workload_id: str,
) -> str:
    if mapped_workload_id:
        return mapped_workload_id
    prefix = f"PUB-{str(dataset).upper()}-SMOKE-NULLSANITY-E550-K6"
    latest_mtime = -1.0
    latest_workload = ""
    for row in rows:
        wid = str(row.get("workload_id", "") or "")
        if not wid.startswith(prefix):
            continue
        if str(row.get("variant", "")) != "B1":
            continue
        if not _is_comparable(row):
            continue
        mtime = _safe_float(row.get("mtime", 0.0))
        if mtime >= latest_mtime:
            latest_mtime = mtime
            latest_workload = wid
    return latest_workload


def _recommended_disposition(*, dataset: str, ceiling: bool, too_hard: bool, refusal_dominated: bool, non_memory_dependent: bool, eligible_for_pilot: bool) -> str:
    if eligible_for_pilot:
        return "eligible_for_new_pilot"
    if ceiling or non_memory_dependent:
        return "do_not_scale"
    if too_hard:
        return "alignment_only"
    if refusal_dominated:
        return "memory_exposure_only"
    return "memory_exposure_only" if dataset in {"longbench_v2", "babilong"} else "alignment_only"


def _analyze_workload(
    *,
    dataset: str,
    workload_id: str,
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]] | None = None,
    null_workload_id: str = "",
) -> dict[str, Any]:
    main_rows = [r for r in rows if str(r.get("workload_id", "")) == workload_id]
    comparable_rows = [r for r in main_rows if _is_comparable(r)]
    b1 = _aggregate(comparable_rows, "B1")
    f1 = _aggregate(comparable_rows, "F1")
    if int(b1.get("runs", 0)) == 0 or int(f1.get("runs", 0)) == 0:
        return {
            "dataset": dataset,
            "workload_id": workload_id,
            "official_smoke_workload_id": workload_id,
            "rows": len(comparable_rows),
            "b1": b1,
            "f1": f1,
            "benchmark_metric": "",
            "benchmark_diff": None,
            "strict_diff": None,
            "no_answer_diff": None,
            "success_diff": None,
            "ceiling": False,
            "too_hard_or_bad_adaptation": False,
            "refusal_dominated": False,
            "needs_null_evidence_sanity": False,
            "null_evidence_available": False,
            "null_workload_id": "",
            "null_b1_benchmark_score": None,
            "null_retention": None,
            "non_memory_dependent": False,
            "eligible_for_pilot": False,
            "recommended_disposition": "insufficient_comparison_variants",
        }
    b1_score = _safe_float(b1.get("benchmark_score", 0.0))
    f1_score = _safe_float(f1.get("benchmark_score", 0.0))
    ceiling = b1_score > 0.90
    too_hard = b1_score < 0.05 and f1_score < 0.05
    refusal_dominated = _safe_float(b1.get("no_answer_rate", 0.0)) > 0.30 and _safe_float(f1.get("no_answer_rate", 0.0)) > 0.30
    needs_null_sanity = not ceiling and not too_hard and not refusal_dominated

    lookup_rows = all_rows if all_rows is not None else rows
    null_rows = [r for r in lookup_rows if str(r.get("workload_id", "")) == null_workload_id] if null_workload_id else []
    null_comparable_rows = [r for r in null_rows if _is_comparable(r)]
    null_b1 = _aggregate(null_comparable_rows, "B1")
    null_available = bool(null_workload_id) and int(null_b1.get("runs", 0)) > 0
    null_b1_score = _safe_float(null_b1.get("benchmark_score", 0.0))
    null_retention = (null_b1_score / max(b1_score, 1e-6)) if null_available else 0.0
    non_memory_dependent = bool(null_available and null_b1_score >= 0.30 and null_retention >= 0.80)
    eligible_for_pilot = bool(
        int(b1.get("runs", 0)) > 0
        and int(f1.get("runs", 0)) > 0
        and not ceiling
        and not too_hard
        and not refusal_dominated
        and (not needs_null_sanity or null_available)
        and not non_memory_dependent
    )
    disposition = _recommended_disposition(
        dataset=dataset,
        ceiling=ceiling,
        too_hard=too_hard,
        refusal_dominated=refusal_dominated,
        non_memory_dependent=non_memory_dependent,
        eligible_for_pilot=eligible_for_pilot,
    )
    return {
        "dataset": dataset,
        "workload_id": workload_id,
        "official_smoke_workload_id": workload_id,
        "rows": len(comparable_rows),
        "b1": b1,
        "f1": f1,
        "benchmark_metric": str(b1.get("benchmark_metric", "") or f1.get("benchmark_metric", "")),
        "benchmark_diff": f1_score - b1_score,
        "strict_diff": _safe_float(f1.get("strict_accuracy", 0.0)) - _safe_float(b1.get("strict_accuracy", 0.0)),
        "no_answer_diff": _safe_float(f1.get("no_answer_rate", 0.0)) - _safe_float(b1.get("no_answer_rate", 0.0)),
        "success_diff": _safe_float(f1.get("success_rate", 0.0)) - _safe_float(b1.get("success_rate", 0.0)),
        "ceiling": ceiling,
        "too_hard_or_bad_adaptation": too_hard,
        "refusal_dominated": refusal_dominated,
        "needs_null_evidence_sanity": needs_null_sanity,
        "null_evidence_available": null_available,
        "null_workload_id": null_workload_id if null_available else "",
        "null_b1_benchmark_score": null_b1_score if null_available else None,
        "null_retention": null_retention if null_available else None,
        "non_memory_dependent": non_memory_dependent,
        "eligible_for_pilot": eligible_for_pilot,
        "recommended_disposition": disposition,
    }


def _write_md(path: Path, results: list[dict[str, Any]], *, title: str) -> None:
    lines: list[str] = [
        f"# {title}",
        "",
        "| dataset | official_smoke_workload_id | metric | B1 score | F1 score | B1 strict | F1 strict | B1 no_answer | F1 no_answer | B1 success | F1 success | null_b1_score | null_retention | ceiling | too_hard/bad_adaptation | refusal_dominated | non_memory_dependent | eligible_for_pilot | recommended_disposition |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in results:
        b1 = row.get("b1", {})
        f1 = row.get("f1", {})
        null_b1 = row.get("null_b1_benchmark_score", None)
        null_retention = row.get("null_retention", None)
        lines.append(
            "| {dataset} | {workload_id} | {metric} | {b1_score:.3f} | {f1_score:.3f} | {b1_strict:.3f} | {f1_strict:.3f} | {b1_no_answer:.3f} | {f1_no_answer:.3f} | {b1_success:.3f} | {f1_success:.3f} | {null_b1} | {null_retention} | {ceiling} | {too_hard} | {refusal} | {non_memory} | {eligible} | {disposition} |".format(
                dataset=str(row.get("dataset", "")),
                workload_id=str(row.get("official_smoke_workload_id", "")),
                metric=str(row.get("benchmark_metric", "")),
                b1_score=_safe_float(b1.get("benchmark_score", 0.0)),
                f1_score=_safe_float(f1.get("benchmark_score", 0.0)),
                b1_strict=_safe_float(b1.get("strict_accuracy", 0.0)),
                f1_strict=_safe_float(f1.get("strict_accuracy", 0.0)),
                b1_no_answer=_safe_float(b1.get("no_answer_rate", 0.0)),
                f1_no_answer=_safe_float(f1.get("no_answer_rate", 0.0)),
                b1_success=_safe_float(b1.get("success_rate", 0.0)),
                f1_success=_safe_float(f1.get("success_rate", 0.0)),
                null_b1=("n/a" if null_b1 is None else f"{_safe_float(null_b1):.3f}"),
                null_retention=("n/a" if null_retention is None else f"{_safe_float(null_retention):.3f}"),
                ceiling=str(bool(row.get("ceiling", False))).lower(),
                too_hard=str(bool(row.get("too_hard_or_bad_adaptation", False))).lower(),
                refusal=str(bool(row.get("refusal_dominated", False))).lower(),
                non_memory=str(bool(row.get("non_memory_dependent", False))).lower(),
                eligible=str(bool(row.get("eligible_for_pilot", False))).lower(),
                disposition=str(row.get("recommended_disposition", "")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `score` uses `benchmark_score_on_total` when available, rather than forcing `strict_accuracy` on open-ended benchmarks.",
            "- `ceiling`: `B1 score > 0.90`.",
            "- `too_hard_or_bad_adaptation`: both B1/F1 score `< 0.05`.",
            "- `refusal_dominated`: both B1/F1 no_answer `> 0.30`.",
            "- `non_memory_dependent`: null-evidence B1 keeps strong score (`score >= 0.30` and retention `>= 0.80`).",
            "- `eligible_for_pilot` requires all gate flags to stay false and, when needed, null-evidence sanity must already exist.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--workload-id", type=str, default="")
    ap.add_argument("--stage", type=str, default="smoke")
    ap.add_argument("--model-id-filter", type=str, default="")
    ap.add_argument("--experiment-track-filter", type=str, default="")
    ap.add_argument("--track-id-filter", type=str, default="")
    ap.add_argument("--null-evidence-map-json", type=str, default="")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--output-md", type=str, required=True)
    args = ap.parse_args()

    null_map = _load_null_map(str(args.null_evidence_map_json or ""))
    rows = _collect_rows(
        Path(args.runs_dir),
        workload_id=str(args.workload_id or ""),
        dataset_filter=str(args.dataset or ""),
        model_id_filter=str(args.model_id_filter or ""),
        experiment_track_filter=str(args.experiment_track_filter or ""),
        track_id_filter=str(args.track_id_filter or ""),
    )
    rows_all_tracks = _collect_rows(
        Path(args.runs_dir),
        workload_id="",
        dataset_filter=str(args.dataset or ""),
        model_id_filter=str(args.model_id_filter or ""),
        experiment_track_filter="",
        track_id_filter="",
    )
    datasets = [d.strip() for d in str(args.datasets or "").split(",") if d.strip()]
    if str(args.dataset or "").strip():
        datasets = [str(args.dataset).strip()]
    if not datasets and not str(args.workload_id or "").strip():
        datasets = sorted({str(r.get("dataset", "")) for r in rows if str(r.get("dataset", "")) != "unknown"})

    results: list[dict[str, Any]] = []
    if str(args.workload_id or "").strip():
        if not rows:
            result = {
                "dataset": str(args.dataset or "unknown"),
                "workload_id": str(args.workload_id),
                "stage": str(args.stage or "smoke"),
                "eligible_for_pilot": False,
                "recommended_disposition": "do_not_scale",
                "reason": "missing rows",
            }
            out_json = Path(args.output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_md(Path(args.output_md), [result], title="Benchmark Discriminativeness Gates")
            print(json.dumps(result, ensure_ascii=False))
            return
        dataset = str(args.dataset or rows[0].get("dataset", "unknown"))
        null_workload_id = _discover_null_workload(rows_all_tracks, dataset=dataset, mapped_workload_id=null_map.get(dataset, ""))
        results = [
            _analyze_workload(
                dataset=dataset,
                workload_id=str(args.workload_id),
                rows=rows,
                all_rows=rows_all_tracks,
                null_workload_id=null_workload_id,
            )
        ]
    else:
        for dataset in datasets:
            workload_id = _find_latest_main_smoke_workload(rows, dataset)
            if not workload_id:
                continue
            null_workload_id = _discover_null_workload(rows_all_tracks, dataset=dataset, mapped_workload_id=null_map.get(dataset, ""))
            results.append(
                _analyze_workload(
                    dataset=dataset,
                    workload_id=workload_id,
                    rows=rows,
                    all_rows=rows_all_tracks,
                    null_workload_id=null_workload_id,
                )
            )

    report: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "model_id_filter": str(args.model_id_filter or ""),
        "experiment_track_filter": str(args.experiment_track_filter or ""),
        "track_id_filter": str(args.track_id_filter or ""),
        "results": results,
    }
    if len(results) == 1:
        report.update(results[0])

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(Path(args.output_md), results, title="Benchmark Discriminativeness Gates")
    print(json.dumps(report if len(results) == 1 else {"results": len(results), "output_json": str(args.output_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
