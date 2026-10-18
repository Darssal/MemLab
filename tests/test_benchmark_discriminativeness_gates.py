from __future__ import annotations

import json
from pathlib import Path

from memlab.analysis.benchmark_discriminativeness_gates import _analyze_workload, _collect_rows


def test_collect_rows_honors_track_id_filter(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    base_summary = {
        "success_rate": 1.0,
        "strict_accuracy": 0.5,
        "relaxed_accuracy": 0.5,
        "benchmark_score_on_total": 0.5,
        "benchmark_metric": "exact_option",
        "latency": {"processing": {"p95": 10.0}, "end_to_end": {"p95": 12.0}},
    }
    for track_id in ("mainline:memory_neutral", "mainline:memory_beneficial"):
        run_dir = runs_dir / track_id.replace(":", "-")
        run_dir.mkdir()
        manifest = {
            "run_id": run_dir.name,
            "variant_id": "B1",
            "workload_id": "PUB-LONGBENCH_V2-SMOKE-MAINLINE",
            "repeat_idx": 0,
            "seed": 42,
            "model_id": "model-x",
            "extra": {
                "tasks_path": "benchmarks/public_main/longbench_v2/tasks_internal.jsonl",
                "experiment_track": "mainline",
                "track_id": track_id,
            },
            "track_id": track_id,
            "env_valid": True,
            "repro_valid": True,
            "llm_real": True,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(base_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _collect_rows(
        runs_dir,
        workload_id="PUB-LONGBENCH_V2-SMOKE-MAINLINE",
        dataset_filter="longbench_v2",
        model_id_filter="model-x",
        experiment_track_filter="mainline",
        track_id_filter="mainline:memory_beneficial",
    )
    assert len(rows) == 1
    assert rows[0]["track_id"] == "mainline:memory_beneficial"


def test_collect_rows_excludes_nonaccepted_and_dedupes_latest(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    summary_old = {
        "success_rate": 1.0,
        "strict_accuracy": 0.2,
        "relaxed_accuracy": 0.2,
        "benchmark_score_on_total": 0.2,
        "benchmark_metric": "exact_option",
        "latency": {"processing": {"p95": 10.0}, "end_to_end": {"p95": 12.0}},
    }
    summary_new = {
        "success_rate": 1.0,
        "strict_accuracy": 0.8,
        "relaxed_accuracy": 0.8,
        "benchmark_score_on_total": 0.8,
        "benchmark_metric": "exact_option",
        "latency": {"processing": {"p95": 10.0}, "end_to_end": {"p95": 12.0}},
    }
    accepted_manifest = {
        "variant_id": "B1",
        "workload_id": "PUB-LONGBENCH_V2-SMOKE-TRACK-M",
        "repeat_idx": 0,
        "seed": 42,
        "model_id": "model-x",
        "extra": {
            "tasks_path": "benchmarks/public_main/longbench_v2/tasks_internal.jsonl",
            "experiment_track": "mainline",
            "track_id": "track-m:memory_beneficial",
        },
        "track_id": "track-m:memory_beneficial",
        "env_valid": True,
        "repro_valid": True,
        "llm_real": True,
        "acceptance_status": "accepted",
        "diagnostic_only": False,
        "summary_chain_valid": True,
    }
    invalid_manifest = dict(accepted_manifest)
    invalid_manifest["acceptance_status"] = "invalid"

    old_run = runs_dir / "old-accepted"
    old_run.mkdir()
    (old_run / "manifest.json").write_text(json.dumps({**accepted_manifest, "run_id": old_run.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    (old_run / "summary.json").write_text(json.dumps(summary_old, ensure_ascii=False, indent=2), encoding="utf-8")

    invalid_run = runs_dir / "invalid-run"
    invalid_run.mkdir()
    (invalid_run / "manifest.json").write_text(json.dumps({**invalid_manifest, "run_id": invalid_run.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    (invalid_run / "summary.json").write_text(json.dumps(summary_new, ensure_ascii=False, indent=2), encoding="utf-8")

    new_run = runs_dir / "new-accepted"
    new_run.mkdir()
    (new_run / "manifest.json").write_text(json.dumps({**accepted_manifest, "run_id": new_run.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    (new_run / "summary.json").write_text(json.dumps(summary_new, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _collect_rows(
        runs_dir,
        workload_id="PUB-LONGBENCH_V2-SMOKE-TRACK-M",
        dataset_filter="longbench_v2",
        model_id_filter="model-x",
        experiment_track_filter="mainline",
        track_id_filter="track-m:memory_beneficial",
    )
    assert len(rows) == 1
    assert rows[0]["benchmark_score"] == 0.8


def test_analyze_workload_marks_missing_b1_f1_as_insufficient() -> None:
    result = _analyze_workload(
        dataset="longbench_v2",
        workload_id="PUB-LONGBENCH_V2-SMOKE-TRACK-S",
        rows=[
            {
                "workload_id": "PUB-LONGBENCH_V2-SMOKE-TRACK-S",
                "variant": "B0",
                "env_valid": True,
                "repro_valid": True,
                "llm_real": True,
                "benchmark_score": 0.25,
                "strict_accuracy": 0.25,
                "relaxed_accuracy": 0.25,
                "no_answer_rate": 0.0,
                "success_rate": 1.0,
                "parse_failure_rate": 0.0,
                "processing_p95_ms": 10.0,
                "end_to_end_p95_ms": 12.0,
                "benchmark_metric": "exact_option",
            }
        ],
        all_rows=[],
        null_workload_id="",
    )
    assert result["eligible_for_pilot"] is False
    assert result["recommended_disposition"] == "insufficient_comparison_variants"
    assert result["benchmark_diff"] is None
