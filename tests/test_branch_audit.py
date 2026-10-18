from __future__ import annotations

import argparse
import json
from pathlib import Path

from memlab.analysis.branch_audit import _audit_run_outputs, _resolve_dataset_audit, command_audit_aggregate
from memlab.metrics.summary_builder import aggregate_runs


def test_dataset_audit_detects_full_slice_mixed(tmp_path: Path) -> None:
    base_dir = tmp_path / "benchmarks" / "public_main" / "babilong"
    base_dir.mkdir(parents=True)
    raw_path = base_dir / "raw.json"
    raw_path.write_text(json.dumps([{"x": 1}, {"x": 2}, {"x": 3}]), encoding="utf-8")
    (base_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "babilong",
                "raw_path": str(raw_path),
                "sample_count": 2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tasks = [
        {"task_id": "t1", "doc_id": "d1", "gold_answer": "a"},
        {"task_id": "t2", "doc_id": "d2", "gold_answer": "b"},
    ]
    docs = [
        {"doc_id": "d1", "body": "x"},
        {"doc_id": "d2", "body": "y"},
    ]
    (base_dir / "tasks_internal.jsonl").write_text("\n".join(json.dumps(x) for x in tasks) + "\n", encoding="utf-8")
    (base_dir / "docs_internal.jsonl").write_text("\n".join(json.dumps(x) for x in docs) + "\n", encoding="utf-8")

    report, anomalies = _resolve_dataset_audit(base_dir, max_samples=0)
    assert "FULL_SLICE_MIXED" in anomalies
    assert report["full_or_slice"] == "full"


def test_run_audit_detects_missing_eventlog(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "f1-e2e-r0"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "f1-e2e-r0",
        "variant_id": "F1",
        "workload_id": "PUB-LONGBENCH_V2-FULL-ARCH",
        "repeat_idx": 0,
        "model_id": "Pro/deepseek-ai/DeepSeek-V3.2",
        "model_base_url": "https://example.test/v1",
        "extra": {
            "tasks_path": "benchmarks/public_main/longbench_v2/tasks_internal.jsonl",
            "docs_path": "benchmarks/public_main/longbench_v2/docs_internal.jsonl",
            "prompt_mode": "memory_beneficial",
            "qa_prompt_profile": "multiple_choice_letter",
            "session_scope": "task",
            "experiment_track": "mainline",
            "track_id": "mainline:memory_beneficial",
            "model_lock": {
                "expected_base_url": "https://example.test/v1",
                "expected_model_id": "Pro/deepseek-ai/DeepSeek-V3.2",
            },
        },
    }
    summary = {
        "run_id": "f1-e2e-r0",
        "variant_id": "F1",
        "workload_id": "PUB-LONGBENCH_V2-FULL-ARCH",
        "parse_failure_rate_on_total": 0.0,
        "no_answer_rate_on_total": 0.0,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report, anomalies, diagnostic_only = _audit_run_outputs(
        run_dir,
        expected_prompt_mode="memory_beneficial",
        expected_qa_profile="multiple_choice_letter",
        expected_session_scope="task",
        expected_model_id="Pro/deepseek-ai/DeepSeek-V3.2",
        expected_base_url="https://example.test/v1",
        expected_track_id="mainline:memory_beneficial",
    )
    assert "EVENTLOG_MISSING" in anomalies
    assert not diagnostic_only
    assert not report["summary_chain_valid"]


def test_aggregate_runs_excludes_nonaccepted_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    accepted_dir = runs_dir / "run-accepted"
    rejected_dir = runs_dir / "run-rejected"
    accepted_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)

    base_manifest = {
        "variant_id": "B1",
        "workload_id": "E2E-UNIT",
        "repeat_idx": 0,
        "seed": 42,
        "metrics_schema_version": 5,
        "env_valid": True,
        "repro_valid": True,
        "llm_real": True,
        "track_id": "mainline:memory_neutral",
        "summary_chain_valid": True,
        "extra": {
            "tasks_path": "workloads/e2e/tasks.jsonl",
            "prompt_mode": "memory_neutral",
            "experiment_track": "mainline",
        },
    }
    accepted_manifest = {
        **base_manifest,
        "run_id": "run-accepted",
        "acceptance_status": "accepted",
        "diagnostic_only": False,
        "invalid_reason_codes": [],
    }
    rejected_manifest = {
        **base_manifest,
        "run_id": "run-rejected",
        "acceptance_status": "invalid",
        "diagnostic_only": False,
        "invalid_reason_codes": ["MODEL_LOCK_DRIFT"],
    }
    summary = {
        "run_id": "run-accepted",
        "variant_id": "B1",
        "workload_id": "E2E-UNIT",
        "success_rate": 1.0,
        "latency": {"processing": {"p95": 10.0}, "end_to_end": {"p95": 11.0}},
        "strict_accuracy_on_total": 0.5,
        "relaxed_accuracy_on_total": 0.5,
        "benchmark_score_on_total": 0.5,
        "accuracy_applicable": True,
        "throughput_tasks_per_min": 1.0,
        "error_breakdown": {"by_category": {}},
    }
    (accepted_dir / "manifest.json").write_text(json.dumps(accepted_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (accepted_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (rejected_dir / "manifest.json").write_text(json.dumps(rejected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rejected_summary = dict(summary)
    rejected_summary["run_id"] = "run-rejected"
    (rejected_dir / "summary.json").write_text(json.dumps(rejected_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    result = aggregate_runs(
        runs_dir=runs_dir,
        summary_csv=tmp_path / "summary.csv",
        stats_json=tmp_path / "summary.stats.json",
        report_md=tmp_path / "experiment_report.md",
        report_alias_md=tmp_path / "report.md",
    )
    assert result["rows"] == 1
    assert result["excluded_rows"] == 1
    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "run-accepted" in csv_text
    assert "run-rejected" not in csv_text


def test_aggregate_audit_scopes_to_workload_id(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    def write_run(name: str, workload_id: str, prompt_mode: str) -> None:
        run_dir = runs_dir / name
        run_dir.mkdir()
        manifest = {
            "run_id": name,
            "variant_id": "B0",
            "workload_id": workload_id,
            "repeat_idx": 0,
            "model_id": "Pro/deepseek-ai/DeepSeek-V3.2",
            "model_base_url": "https://example.test/v1",
            "acceptance_status": "accepted",
            "diagnostic_only": False,
            "summary_chain_valid": True,
            "extra": {
                "dataset_id": "longbench_v2",
                "track_id": "track-s:memory_neutral",
                "prompt_mode": prompt_mode,
                "qa_prompt_profile": "multiple_choice_letter",
                "session_scope": "task",
                "model_lock": {
                    "expected_base_url": "https://example.test/v1",
                    "expected_model_id": "Pro/deepseek-ai/DeepSeek-V3.2",
                },
            },
        }
        summary = {
            "run_id": name,
            "variant_id": "B0",
            "workload_id": workload_id,
            "benchmark_score_on_total": 0.5,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_run("run-smoke", "PUB-LONGBENCH_V2-SMOKE-A", "memory_neutral")
    write_run("run-old", "PUB-LONGBENCH_V2-SMOKE-OLD", "memory_beneficial")

    rc = command_audit_aggregate(
        argparse.Namespace(
            runs_dir=str(runs_dir),
            dataset_id="longbench_v2",
            track_id="track-s:memory_neutral",
            workload_id="PUB-LONGBENCH_V2-SMOKE-A",
            artifacts_root=str(tmp_path / "artifacts"),
        )
    )
    assert rc == 0
