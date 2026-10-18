from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from memlab.pipeline.control_plane import BranchKey, ControlPlane, normalize_dataset_id


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _infer_track_goal(track_goal: str, *, track_id: str, experiment_track: str) -> str:
    value = str(track_goal or "auto").strip().lower()
    if value in {"standard", "memory_sensitive"}:
        return value
    track_text = f"{track_id}::{experiment_track}".lower()
    if "track-m" in track_text or "memory_exposure" in track_text:
        return "memory_sensitive"
    return "standard"


def _safe_hist(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter.keys()) if str(key)}


def resolve_design_audit(
    *,
    base_dir: Path,
    dataset_id: str,
    track_id: str,
    experiment_track: str,
    prompt_mode: str,
    qa_support_mode: str,
    qa_prompt_profile: str,
    track_goal: str,
    min_memory_needed_ratio: float,
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = base_dir / "manifest.json"
    tasks_path = base_dir / "tasks_internal.jsonl"
    docs_path = base_dir / "docs_internal.jsonl"
    report: dict[str, Any] = {
        "base_dir": base_dir.as_posix(),
        "dataset_id": normalize_dataset_id(dataset_id or base_dir.name),
        "track_id": str(track_id or ""),
        "experiment_track": str(experiment_track or ""),
        "prompt_mode": str(prompt_mode or ""),
        "qa_support_mode": str(qa_support_mode or ""),
        "qa_prompt_profile": str(qa_prompt_profile or ""),
        "track_goal": _infer_track_goal(
            track_goal,
            track_id=str(track_id or ""),
            experiment_track=str(experiment_track or ""),
        ),
        "checks": {},
        "anomaly_codes": [],
    }
    anomalies: list[str] = []
    if not manifest_path.exists() or not tasks_path.exists() or not docs_path.exists():
        report["checks"]["required_files_present"] = False
        anomalies.append("DATA_COUNT_MISMATCH")
        report["anomaly_codes"] = sorted(set(anomalies))
        return report, report["anomaly_codes"]
    report["checks"]["required_files_present"] = True
    manifest = _read_json(manifest_path)
    tasks = _read_jsonl(tasks_path)
    docs = _read_jsonl(docs_path)

    memory_needed_true = 0
    memory_needed_false = 0
    metric_hist: Counter[str] = Counter()
    task_profile_hist: Counter[str] = Counter()
    question_type_hist: Counter[str] = Counter()
    for row in tasks:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if bool(metadata.get("memory_needed")):
            memory_needed_true += 1
        else:
            memory_needed_false += 1
        metric_hist[str(metadata.get("answer_metric_hint", "")).strip()] += 1
        task_profile_hist[str(metadata.get("qa_prompt_profile", "")).strip()] += 1
        question_type_hist[str(metadata.get("question_type", metadata.get("task_type", ""))).strip()] += 1

    total = len(tasks)
    ratio = float(memory_needed_true) / float(total) if total else 0.0
    prepared_count = int(manifest.get("sample_count", total) or total)
    report["manifest_dataset"] = str(manifest.get("dataset", ""))
    report["prepared_sample_count"] = int(prepared_count)
    report["task_count"] = int(total)
    report["doc_count"] = int(len(docs))
    report["memory_needed_true"] = int(memory_needed_true)
    report["memory_needed_false"] = int(memory_needed_false)
    report["memory_needed_ratio"] = ratio
    report["answer_metric_histogram"] = _safe_hist(metric_hist)
    report["task_profile_histogram"] = _safe_hist(task_profile_hist)
    report["question_type_histogram"] = _safe_hist(question_type_hist)
    report["checks"]["counts_consistent"] = prepared_count == len(tasks) == len(docs)
    report["checks"]["memory_needed_ratio_meets_gate"] = ratio >= float(min_memory_needed_ratio)
    report["checks"]["qa_support_mode_safe_for_track_goal"] = True
    report["checks"]["prompt_mode_safe_for_track_goal"] = True

    goal = str(report["track_goal"])
    if prepared_count != len(tasks) or prepared_count != len(docs):
        anomalies.append("DATA_COUNT_MISMATCH")
    if goal == "memory_sensitive":
        if ratio < float(min_memory_needed_ratio):
            anomalies.append("NON_MEMORY_DEPENDENT_DATASET")
        if str(prompt_mode or "").strip() != "memory_beneficial":
            anomalies.append("TRACK_GOAL_MISMATCH")
            report["checks"]["prompt_mode_safe_for_track_goal"] = False
        if str(qa_support_mode or "").strip() != "none":
            anomalies.append("DIRECT_EVIDENCE_LEAKAGE")
            report["checks"]["qa_support_mode_safe_for_track_goal"] = False
    report["anomaly_codes"] = sorted(set(anomalies))
    return report, report["anomaly_codes"]


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Public Track Design Audit",
        "",
        f"- Dataset: `{report.get('dataset_id', '')}`",
        f"- Track: `{report.get('track_id', '')}`",
        f"- Experiment track: `{report.get('experiment_track', '')}`",
        f"- Track goal: `{report.get('track_goal', '')}`",
        f"- Prompt mode: `{report.get('prompt_mode', '')}`",
        f"- QA support mode: `{report.get('qa_support_mode', '')}`",
        f"- QA prompt profile: `{report.get('qa_prompt_profile', '')}`",
        f"- Prepared base dir: `{report.get('base_dir', '')}`",
        "",
        "## Counts",
        "",
        f"- Prepared sample count: `{report.get('prepared_sample_count', 0)}`",
        f"- Task count: `{report.get('task_count', 0)}`",
        f"- Doc count: `{report.get('doc_count', 0)}`",
        f"- Memory-needed ratio: `{float(report.get('memory_needed_ratio', 0.0)):.3f}`",
        "",
        "## Histograms",
        "",
        f"- answer_metric_histogram: `{json.dumps(report.get('answer_metric_histogram', {}), ensure_ascii=False)}`",
        f"- task_profile_histogram: `{json.dumps(report.get('task_profile_histogram', {}), ensure_ascii=False)}`",
        f"- question_type_histogram: `{json.dumps(report.get('question_type_histogram', {}), ensure_ascii=False)}`",
        "",
        "## Checks",
        "",
    ]
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    for key in sorted(checks.keys()):
        lines.append(f"- {key}: `{checks[key]}`")
    lines.extend(["", "## Anomalies", ""])
    anomalies = report.get("anomaly_codes", [])
    if anomalies:
        for code in anomalies:
            lines.append(f"- `{code}`")
    else:
        lines.append("- `none`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, required=True)
    ap.add_argument("--dataset-id", type=str, required=True)
    ap.add_argument("--track-id", type=str, required=True)
    ap.add_argument("--variant-id", type=str, required=True)
    ap.add_argument("--repeat-idx", type=int, required=True)
    ap.add_argument("--experiment-track", type=str, default="mainline")
    ap.add_argument("--prompt-mode", type=str, required=True)
    ap.add_argument("--qa-support-mode", type=str, required=True)
    ap.add_argument("--qa-prompt-profile", type=str, default="default")
    ap.add_argument("--track-goal", type=str, default="auto", choices=("auto", "standard", "memory_sensitive"))
    ap.add_argument("--min-memory-needed-ratio", type=float, default=0.5)
    ap.add_argument("--artifacts-root", type=str, default="artifacts")
    ap.add_argument("--output-json", type=str, default="")
    ap.add_argument("--output-md", type=str, default="")
    args = ap.parse_args()

    key = BranchKey(
        dataset_id=normalize_dataset_id(str(args.dataset_id)),
        track_id=str(args.track_id),
        variant_id=str(args.variant_id),
        repeat_idx=int(args.repeat_idx),
    )
    cp = ControlPlane(artifacts_root=args.artifacts_root)
    cp.register_branch(key)

    report, anomalies = resolve_design_audit(
        base_dir=Path(args.base_dir),
        dataset_id=str(args.dataset_id),
        track_id=str(args.track_id),
        experiment_track=str(args.experiment_track),
        prompt_mode=str(args.prompt_mode),
        qa_support_mode=str(args.qa_support_mode),
        qa_prompt_profile=str(args.qa_prompt_profile),
        track_goal=str(args.track_goal),
        min_memory_needed_ratio=float(args.min_memory_needed_ratio),
    )
    report["branch_key"] = {
        "branch_id": key.branch_id,
        "dataset_id": key.dataset_id,
        "track_id": key.track_id,
        "variant_id": key.variant_id,
        "repeat_idx": int(key.repeat_idx),
    }
    report["acceptance_status"] = "accepted" if not anomalies else "invalid"
    cp.write_audit_report(key, stage_id="audit_protocol", report=report)

    if str(args.output_json).strip():
        out_json = Path(str(args.output_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if str(args.output_md).strip():
        _write_md(Path(str(args.output_md)), report)

    if anomalies:
        reason = (
            "Public track design audit failed: "
            + ", ".join(sorted(set(anomalies)))
            + ". Fix dataset selection or protocol before smoke."
        )
        rerun_plan_id = cp.create_rerun_plan(
            key,
            rollback_root_stage="audit_protocol",
            anomaly_codes=anomalies,
            invalidated_run_ids=[],
            why=reason,
            what="Re-run protocol audit after fixing dataset suitability or track prompt configuration.",
        )
        cp.record_anomaly(
            key,
            run_id=f"audit::{key.branch_id}",
            stage_id="audit_protocol",
            anomaly_codes=anomalies,
            reason=reason,
            diagnostic_only=False,
            rerun_plan_id=rerun_plan_id,
        )
        cp.record_stage_status(
            key,
            stage_id="audit_protocol",
            status="invalid",
            run_id="",
            note=reason,
            anomaly_codes=anomalies,
            rollback_root_stage="audit_protocol",
            rerun_plan_id=rerun_plan_id,
        )
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(2)

    cp.record_stage_status(
        key,
        stage_id="audit_protocol",
        status="accepted",
        run_id="",
        note="Public track design audit passed.",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
