from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from memlab.pipeline.control_plane import (
    ControlPlane,
    BranchKey,
    accepted_for_official_reporting,
    fingerprint_paths,
    infer_branch_key_from_manifest,
    infer_dataset_id_from_manifest,
    infer_track_id_from_manifest,
    merge_control_fields,
    normalize_dataset_id,
    sha1_text,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_prepared_manifest_from_run_manifest(manifest: dict[str, Any]) -> Path | None:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    for key in ("tasks_path", "docs_path"):
        raw = str(extra.get(key, "") or "").strip()
        if not raw:
            continue
        candidate = Path(raw).parent / "manifest.json"
        if candidate.exists():
            return candidate
    return None


def _count_raw_samples(dataset_id: str, dataset_manifest: dict[str, Any]) -> int | None:
    dataset_id = normalize_dataset_id(dataset_id)
    raw_path = Path(str(dataset_manifest.get("raw_path", "")))
    if not raw_path.exists():
        return None
    if dataset_id in {"longbench_v2", "babilong"}:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        return len(raw) if isinstance(raw, list) else None
    if dataset_id == "longmemeval":
        import pandas as pd

        return int(len(pd.read_parquet(raw_path)))
    if dataset_id == "locomo":
        import pandas as pd

        df = pd.read_parquet(raw_path)
        total = 0
        for _, row in df.iterrows():
            total += min(len(list(row.get("questions", []))), len(list(row.get("answers", []))))
        return int(total)
    if dataset_id == "squad_v11":
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        total = 0
        for article in raw.get("data", []):
            for para in article.get("paragraphs", []):
                for qa in para.get("qas", []):
                    answers = qa.get("answers", [])
                    if qa.get("id") and qa.get("question") and answers and str(answers[0].get("text", "")).strip():
                        total += 1
        return total
    return None


def _resolve_dataset_audit(base_dir: Path, *, max_samples: int) -> tuple[dict[str, Any], list[str]]:
    manifest_path = base_dir / "manifest.json"
    tasks_path = base_dir / "tasks_internal.jsonl"
    docs_path = base_dir / "docs_internal.jsonl"
    report: dict[str, Any] = {
        "base_dir": base_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "tasks_path": tasks_path.as_posix(),
        "docs_path": docs_path.as_posix(),
        "checks": {},
        "anomaly_codes": [],
    }
    anomalies: list[str] = []
    if not manifest_path.exists() or not tasks_path.exists() or not docs_path.exists():
        anomalies.append("DATA_COUNT_MISMATCH")
        report["checks"]["required_files_present"] = False
        return report, anomalies
    report["checks"]["required_files_present"] = True
    manifest = _read_json(manifest_path)
    dataset_id = normalize_dataset_id(str(manifest.get("dataset", base_dir.name)))
    tasks = _read_jsonl(tasks_path)
    docs = _read_jsonl(docs_path)
    manifest_count = int(manifest.get("sample_count", 0) or 0)
    task_count = len(tasks)
    doc_count = len(docs)
    raw_count = _count_raw_samples(dataset_id, manifest)
    report["dataset_id"] = dataset_id
    report["raw_sample_count"] = raw_count
    report["prepared_sample_count"] = manifest_count
    report["task_count"] = task_count
    report["doc_count"] = doc_count
    report["requested_max_samples"] = int(max_samples)
    expected_full = int(max_samples) <= 0
    report["full_or_slice"] = "full" if expected_full else "slice"
    if manifest_count != task_count or manifest_count != doc_count:
        anomalies.append("DATA_COUNT_MISMATCH")
    if raw_count is not None and expected_full and manifest_count != raw_count:
        anomalies.append("FULL_SLICE_MIXED")
    task_ids = [str(row.get("task_id", "")).strip() for row in tasks]
    doc_ids = [str(row.get("doc_id", "")).strip() for row in docs]
    dup_task_ids = sorted([k for k, v in Counter(task_ids).items() if k and v > 1])
    empty_task_ids = [idx for idx, value in enumerate(task_ids) if not value]
    empty_doc_ids = [idx for idx, value in enumerate(doc_ids) if not value]
    empty_gold = []
    for idx, row in enumerate(tasks):
        gold = str(row.get("gold_answer", row.get("answer_key", "")) or "").strip()
        if not gold:
            empty_gold.append(idx)
    if dup_task_ids or empty_task_ids or empty_doc_ids:
        anomalies.append("DUPLICATE_OR_MISSING_SAMPLE")
    if empty_gold:
        anomalies.append("EMPTY_GOLD_OR_LABEL")
    report["checks"]["duplicate_task_ids"] = dup_task_ids[:20]
    report["checks"]["empty_task_id_rows"] = empty_task_ids[:20]
    report["checks"]["empty_doc_id_rows"] = empty_doc_ids[:20]
    report["checks"]["empty_gold_rows"] = empty_gold[:20]
    report["dataset_fingerprint"] = fingerprint_paths([manifest_path, tasks_path, docs_path])
    report["checks"]["counts_consistent"] = not {"DATA_COUNT_MISMATCH", "FULL_SLICE_MIXED"} & set(anomalies)
    report["anomaly_codes"] = sorted(set(anomalies))
    return report, sorted(set(anomalies))


def _find_latest_run(
    runs_dir: Path,
    *,
    workload_id: str,
    variant_id: str,
    repeat_idx: int,
    model_id: str = "",
    track_id: str = "",
) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if str(manifest.get("workload_id", "")) != workload_id:
            continue
        if str(manifest.get("variant_id", "")) != variant_id:
            continue
        if int(manifest.get("repeat_idx", 0)) != int(repeat_idx):
            continue
        if model_id and str(manifest.get("model_id", "")) != model_id:
            continue
        if track_id and infer_track_id_from_manifest(manifest) != track_id:
            continue
        candidates.append((float(run_dir.stat().st_mtime), run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _find_previous_matching_run(
    runs_dir: Path,
    *,
    current_run_dir: Path,
    workload_id: str,
    variant_id: str,
    repeat_idx: int,
    dataset_id: str,
    track_id: str,
) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.resolve() == current_run_dir.resolve():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if str(manifest.get("workload_id", "")) != workload_id:
            continue
        if str(manifest.get("variant_id", "")) != variant_id:
            continue
        if int(manifest.get("repeat_idx", 0)) != int(repeat_idx):
            continue
        if normalize_dataset_id(infer_dataset_id_from_manifest(manifest)) != normalize_dataset_id(dataset_id):
            continue
        if infer_track_id_from_manifest(manifest) != track_id:
            continue
        candidates.append((float(run_dir.stat().st_mtime), run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _protocol_fingerprint(manifest: dict[str, Any]) -> str:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    model_lock = extra.get("model_lock") if isinstance(extra.get("model_lock"), dict) else {}
    payload = {
        "model_base_url": manifest.get("model_base_url"),
        "model_id": manifest.get("model_id"),
        "expected_base_url": model_lock.get("expected_base_url"),
        "expected_model_id": model_lock.get("expected_model_id"),
        "prompt_mode": extra.get("prompt_mode"),
        "qa_prompt_profile": extra.get("qa_prompt_profile"),
        "qa_support_mode": extra.get("qa_support_mode"),
        "session_scope": extra.get("session_scope"),
        "max_evidence_chars": extra.get("max_evidence_chars"),
        "context_top_k": extra.get("context_top_k"),
        "context_item_max_chars": extra.get("context_item_max_chars"),
        "answer_format_mode": extra.get("answer_format_mode"),
        "structured_evidence": extra.get("structured_evidence"),
        "track_id": infer_track_id_from_manifest(manifest),
    }
    return sha1_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _audit_run_outputs(
    run_dir: Path,
    *,
    expected_prompt_mode: str = "",
    expected_qa_profile: str = "",
    expected_session_scope: str = "",
    expected_model_id: str = "",
    expected_base_url: str = "",
    expected_track_id: str = "",
    invalid_rate_threshold: float = 0.15,
) -> tuple[dict[str, Any], list[str], bool]:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    eventlog_path = run_dir / "eventlog.jsonl"
    report: dict[str, Any] = {
        "run_dir": run_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "eventlog_path": eventlog_path.as_posix(),
        "checks": {},
        "anomaly_codes": [],
    }
    anomalies: list[str] = []
    if not manifest_path.exists():
        anomalies.append("SUMMARY_CHAIN_INCONSISTENT")
        report["checks"]["manifest_present"] = False
        return report, anomalies, False
    manifest = _read_json(manifest_path)
    report["run_id"] = str(manifest.get("run_id", run_dir.name))
    report["branch_key"] = as_branch_dict(infer_branch_key_from_manifest(manifest))
    report["checks"]["manifest_present"] = True
    if not summary_path.exists():
        anomalies.append("SUMMARY_CHAIN_INCONSISTENT")
        report["checks"]["summary_present"] = False
        return report, sorted(set(anomalies)), False
    report["checks"]["summary_present"] = True
    summary = _read_json(summary_path)
    report["checks"]["eventlog_present"] = bool(eventlog_path.exists())
    if not eventlog_path.exists():
        anomalies.append("EVENTLOG_MISSING")
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    actual_prompt_mode = str(extra.get("prompt_mode", "")).strip()
    actual_qa_profile = str(extra.get("qa_prompt_profile", "")).strip()
    actual_session_scope = str(extra.get("session_scope", "")).strip()
    actual_track_id = infer_track_id_from_manifest(manifest)
    model_lock = extra.get("model_lock") if isinstance(extra.get("model_lock"), dict) else {}
    actual_model_id = str(manifest.get("model_id", "")).strip()
    actual_base_url = str(manifest.get("model_base_url", "")).strip().rstrip("/")
    expected_model_id = str(expected_model_id or "").strip()
    expected_base_url = str(expected_base_url or "").strip().rstrip("/")
    if expected_prompt_mode and actual_prompt_mode != expected_prompt_mode:
        anomalies.append("PROVIDER_PARAM_DRIFT")
    if expected_qa_profile and actual_qa_profile != expected_qa_profile:
        anomalies.append("PROMPT_PROFILE_MISMATCH")
    if expected_session_scope and actual_session_scope != expected_session_scope:
        anomalies.append("SESSION_SCOPE_DRIFT")
    if expected_track_id and actual_track_id != expected_track_id:
        anomalies.append("MIXED_TRACK_COMPARISON")
    if expected_model_id and actual_model_id != expected_model_id:
        anomalies.append("MODEL_LOCK_DRIFT")
    if expected_base_url and actual_base_url != expected_base_url:
        anomalies.append("MODEL_LOCK_DRIFT")
    if model_lock:
        if expected_model_id and str(model_lock.get("expected_model_id", "")).strip() != expected_model_id:
            anomalies.append("MODEL_LOCK_DRIFT")
        if expected_base_url and str(model_lock.get("expected_base_url", "")).strip().rstrip("/") != expected_base_url:
            anomalies.append("MODEL_LOCK_DRIFT")
    if str(summary.get("run_id", "")).strip() not in {"", str(manifest.get("run_id", ""))}:
        anomalies.append("SUMMARY_CHAIN_INCONSISTENT")
    if str(summary.get("variant_id", "")).strip() not in {"", str(manifest.get("variant_id", ""))}:
        anomalies.append("SUMMARY_CHAIN_INCONSISTENT")
    if str(summary.get("workload_id", "")).strip() not in {"", str(manifest.get("workload_id", ""))}:
        anomalies.append("SUMMARY_CHAIN_INCONSISTENT")
    invalid_rate = float(summary.get("parse_failure_rate_on_total", summary.get("parse_failure_rate", 0.0)) or 0.0)
    no_answer_rate = float(summary.get("no_answer_rate_on_total", summary.get("no_answer_rate", 0.0)) or 0.0)
    report["checks"]["invalid_rate"] = invalid_rate
    report["checks"]["no_answer_rate"] = no_answer_rate
    if invalid_rate > float(invalid_rate_threshold) or no_answer_rate > float(invalid_rate_threshold):
        anomalies.append("INVALID_RATE_SPIKE")
    report["summary_chain_valid"] = not bool({"SUMMARY_CHAIN_INCONSISTENT", "EVENTLOG_MISSING"} & set(anomalies))
    report["protocol_fingerprint"] = _protocol_fingerprint(manifest)
    report["anomaly_codes"] = sorted(set(anomalies))
    diagnostic_only = set(anomalies) <= {"INVALID_RATE_SPIKE"} and bool(anomalies)
    return report, sorted(set(anomalies)), diagnostic_only


def as_branch_dict(key: BranchKey) -> dict[str, Any]:
    return {
        "branch_id": key.branch_id,
        "dataset_id": key.dataset_id,
        "track_id": key.track_id,
        "variant_id": key.variant_id,
        "repeat_idx": int(key.repeat_idx),
    }


def command_audit_dataset(args: argparse.Namespace) -> int:
    key = BranchKey(
        dataset_id=normalize_dataset_id(str(args.dataset_id)),
        track_id=str(args.track_id),
        variant_id=str(args.variant_id),
        repeat_idx=int(args.repeat_idx),
    )
    cp = ControlPlane(artifacts_root=args.artifacts_root)
    cp.register_branch(key, metadata={"audit_type": "dataset"})
    report, anomalies = _resolve_dataset_audit(Path(args.base_dir), max_samples=int(args.max_samples))
    cp.write_audit_report(key, stage_id="audit_data", report=report)
    dataset_manifest_path = Path(args.base_dir) / "manifest.json"
    if dataset_manifest_path.exists():
        dataset_manifest = _read_json(dataset_manifest_path)
        dataset_manifest = merge_control_fields(
            dataset_manifest,
            {
                "dataset_fingerprint": report.get("dataset_fingerprint"),
            },
        )
        Path(dataset_manifest_path).write_text(json.dumps(dataset_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if anomalies:
        rerun_plan_id = cp.create_rerun_plan(
            key,
            rollback_root_stage="audit_data",
            anomaly_codes=anomalies,
            invalidated_run_ids=[],
            why="Dataset audit failed.",
            what="Rebuild prepared dataset artifacts and rerun downstream stages.",
        )
        cp.record_anomaly(
            key,
            run_id="",
            stage_id="audit_data",
            anomaly_codes=anomalies,
            reason="Dataset audit failed.",
            diagnostic_only=False,
            rerun_plan_id=rerun_plan_id,
        )
        cp.record_stage_status(
            key,
            stage_id="audit_data",
            status="invalid",
            note="Dataset audit failed.",
            anomaly_codes=anomalies,
            rollback_root_stage="audit_data",
            rerun_plan_id=rerun_plan_id,
        )
        return 2
    cp.record_stage_status(key, stage_id="audit_data", status="accepted", note="Dataset audit passed.")
    return 0


def command_audit_run(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run(
        runs_dir,
        workload_id=str(args.workload_id),
        variant_id=str(args.variant_id),
        repeat_idx=int(args.repeat_idx),
        model_id=str(args.model_id or ""),
        track_id=str(args.track_id or ""),
    )
    if run_dir is None:
        raise SystemExit(f"No matching run found for workload={args.workload_id}, variant={args.variant_id}, repeat={args.repeat_idx}")
    manifest = _read_json(run_dir / "manifest.json")
    key = BranchKey(
        dataset_id=normalize_dataset_id(str(args.dataset_id or infer_dataset_id_from_manifest(manifest))),
        track_id=str(args.track_id or infer_track_id_from_manifest(manifest)),
        variant_id=str(args.variant_id or manifest.get("variant_id", "")),
        repeat_idx=int(args.repeat_idx if args.repeat_idx is not None else manifest.get("repeat_idx", 0)),
    )
    cp = ControlPlane(artifacts_root=args.artifacts_root)
    cp.register_branch(key, metadata={"audit_type": "run"})
    prepared_manifest_path = _resolve_prepared_manifest_from_run_manifest(manifest)
    prepared_manifest = _read_json(prepared_manifest_path) if prepared_manifest_path and prepared_manifest_path.exists() else {}
    dataset_fingerprint = str(
        manifest.get("dataset_fingerprint")
        or prepared_manifest.get("dataset_fingerprint")
        or ""
    ).strip()
    report, anomalies, diagnostic_only = _audit_run_outputs(
        run_dir,
        expected_prompt_mode=str(args.expected_prompt_mode or ""),
        expected_qa_profile=str(args.expected_qa_profile or ""),
        expected_session_scope=str(args.expected_session_scope or ""),
        expected_model_id=str(args.model_id or ""),
        expected_base_url=str(args.base_url or ""),
        expected_track_id=str(args.track_id or ""),
        invalid_rate_threshold=float(args.invalid_rate_threshold),
    )
    report["run_id"] = str(manifest.get("run_id", run_dir.name))
    report["dataset_fingerprint"] = dataset_fingerprint or None
    cp.write_audit_report(key, stage_id=str(args.stage_id), report=report)
    manifest_updates = {
        "input_fingerprint": fingerprint_paths(
            [
                str((manifest.get("extra") or {}).get("tasks_path", "")),
                str((manifest.get("extra") or {}).get("docs_path", "")),
            ]
        ),
        "protocol_fingerprint": report.get("protocol_fingerprint"),
        "dataset_fingerprint": dataset_fingerprint or None,
        "track_id": key.track_id,
        "summary_chain_valid": bool(report.get("summary_chain_valid", False)),
    }
    if anomalies:
        cp.annotate_manifest(run_dir / "manifest.json", **manifest_updates)
        cp.invalidate_run(
            run_dir / "manifest.json",
            key=key,
            stage_id=str(args.stage_id),
            anomaly_codes=anomalies,
            reason=f"Run audit failed at {args.stage_id}.",
            diagnostic_only=diagnostic_only,
        )
        return 2
    previous_run_dir = _find_previous_matching_run(
        runs_dir,
        current_run_dir=run_dir,
        workload_id=str(manifest.get("workload_id", "")),
        variant_id=str(manifest.get("variant_id", "")),
        repeat_idx=int(manifest.get("repeat_idx", 0)),
        dataset_id=key.dataset_id,
        track_id=key.track_id,
    )
    supersedes_run_id = ""
    if previous_run_dir is not None:
        previous_manifest = _read_json(previous_run_dir / "manifest.json")
        supersedes_run_id = str(previous_manifest.get("run_id", previous_run_dir.name))
        cp.annotate_manifest(
            previous_run_dir / "manifest.json",
            acceptance_status="superseded",
            superseded_by_run_id=str(manifest.get("run_id", run_dir.name)),
        )
    cp.annotate_manifest(
        run_dir / "manifest.json",
        **manifest_updates,
        acceptance_status="accepted",
        diagnostic_only=False,
        invalid_reason_codes=[],
        anomaly_classification=[],
        supersedes_run_id=(supersedes_run_id or None),
        superseded_by_run_id=None,
    )
    cp.record_stage_status(
        key,
        stage_id=str(args.stage_id),
        status="accepted",
        run_id=str(manifest.get("run_id", run_dir.name)),
        note="Run audit passed.",
    )
    return 0


def command_audit_aggregate(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    cp = ControlPlane(artifacts_root=args.artifacts_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    run_dirs: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if not accepted_for_official_reporting(manifest):
            continue
        if str(args.workload_id or "").strip():
            if str(manifest.get("workload_id", "") or "") != str(args.workload_id):
                continue
        actual_dataset_id = normalize_dataset_id(str(infer_dataset_id_from_manifest(manifest)))
        actual_track_id = str(infer_track_id_from_manifest(manifest))
        if str(args.dataset_id or "").strip() and actual_dataset_id != normalize_dataset_id(str(args.dataset_id)):
            continue
        if str(args.track_id or "").strip() and actual_track_id != str(args.track_id):
            continue
        key = (actual_dataset_id, actual_track_id)
        grouped[key].append(
            {
                "run_dir": run_dir,
                "manifest": manifest,
                "summary": _read_json(summary_path),
            }
        )
        run_dirs[key].append(run_dir)
    anomalies_found = 0
    for (dataset_id, track_id), items in grouped.items():
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        prompt_modes = {str((item["manifest"].get("extra") or {}).get("prompt_mode", "")) for item in items}
        protocol_fingerprints = {_protocol_fingerprint(item["manifest"]) for item in items}
        if len([x for x in prompt_modes if x]) > 1 or len(protocol_fingerprints) > 1:
            anomalies_found += 1
            for item in items:
                manifest = item["manifest"]
                branch = BranchKey(
                    dataset_id=dataset_id,
                    track_id=track_id,
                    variant_id=str(manifest.get("variant_id", "")),
                    repeat_idx=int(manifest.get("repeat_idx", 0)),
                )
                cp.invalidate_run(
                    item["run_dir"] / "manifest.json",
                    key=branch,
                    stage_id="aggregate",
                    anomaly_codes=["MIXED_TRACK_COMPARISON", "PROVIDER_PARAM_DRIFT"],
                    reason="Aggregate audit found protocol drift inside one dataset/track group.",
                    diagnostic_only=False,
                )
            continue
        for item in items:
            by_variant[str(item["manifest"].get("variant_id", ""))].append(item)
        base_rows = by_variant.get("B1") or by_variant.get("B0") or []
        if not base_rows:
            continue
        base_scores = {
            int(row["manifest"].get("repeat_idx", 0)): float(row["summary"].get("benchmark_score_on_total", 0.0))
            for row in base_rows
        }
        for variant in ("V1", "F1", "F2"):
            variant_rows = by_variant.get(variant, [])
            if not variant_rows:
                continue
            deltas: list[float] = []
            for row in variant_rows:
                repeat_idx = int(row["manifest"].get("repeat_idx", 0))
                if repeat_idx not in base_scores:
                    continue
                delta = float(row["summary"].get("benchmark_score_on_total", 0.0)) - float(base_scores[repeat_idx])
                deltas.append(delta)
            if len(deltas) >= 2 and any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas):
                anomalies_found += 1
                for row in variant_rows:
                    manifest = row["manifest"]
                    branch = BranchKey(
                        dataset_id=dataset_id,
                        track_id=track_id,
                        variant_id=str(manifest.get("variant_id", "")),
                        repeat_idx=int(manifest.get("repeat_idx", 0)),
                    )
                    cp.invalidate_run(
                        row["run_dir"] / "manifest.json",
                        key=branch,
                        stage_id="aggregate",
                        anomaly_codes=["REPEAT_DIRECTION_CONFLICT"],
                        reason="Aggregate audit found conflicting delta directions across repeats.",
                        diagnostic_only=True,
                    )
    return 0 if anomalies_found == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ds = sub.add_parser("dataset")
    ds.add_argument("--base-dir", type=str, required=True)
    ds.add_argument("--dataset-id", type=str, required=True)
    ds.add_argument("--track-id", type=str, required=True)
    ds.add_argument("--variant-id", type=str, required=True)
    ds.add_argument("--repeat-idx", type=int, required=True)
    ds.add_argument("--max-samples", type=int, default=0)
    ds.add_argument("--artifacts-root", type=str, default="artifacts")

    run = sub.add_parser("run")
    run.add_argument("--runs-dir", type=str, default="runs")
    run.add_argument("--run-dir", type=str, default="")
    run.add_argument("--dataset-id", type=str, default="")
    run.add_argument("--track-id", type=str, default="")
    run.add_argument("--variant-id", type=str, required=True)
    run.add_argument("--repeat-idx", type=int, required=True)
    run.add_argument("--workload-id", type=str, required=True)
    run.add_argument("--stage-id", type=str, required=True, choices=["smoke", "pilot", "full", "aggregate", "promote"])
    run.add_argument("--model-id", type=str, default="")
    run.add_argument("--base-url", type=str, default="")
    run.add_argument("--expected-prompt-mode", type=str, default="")
    run.add_argument("--expected-qa-profile", type=str, default="")
    run.add_argument("--expected-session-scope", type=str, default="")
    run.add_argument("--invalid-rate-threshold", type=float, default=0.15)
    run.add_argument("--artifacts-root", type=str, default="artifacts")

    agg = sub.add_parser("aggregate")
    agg.add_argument("--runs-dir", type=str, default="runs")
    agg.add_argument("--dataset-id", type=str, default="")
    agg.add_argument("--track-id", type=str, default="")
    agg.add_argument("--workload-id", type=str, default="")
    agg.add_argument("--artifacts-root", type=str, default="artifacts")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "dataset":
        raise SystemExit(command_audit_dataset(args))
    if args.cmd == "run":
        raise SystemExit(command_audit_run(args))
    if args.cmd == "aggregate":
        raise SystemExit(command_audit_aggregate(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
