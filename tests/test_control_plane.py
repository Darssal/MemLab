from __future__ import annotations

import json
from pathlib import Path

from memlab.metrics.manifest import build_manifest
from memlab.pipeline.control_plane import (
    BranchKey,
    ControlPlane,
    accepted_for_official_reporting,
    default_manifest_control_fields,
    earliest_rollback_stage,
    sanitize_path_token,
)


def test_build_manifest_includes_control_plane_defaults() -> None:
    manifest = build_manifest(
        run_id="run-1",
        variant_id="F1",
        workload_id="PUB-TEST-FULL",
        seed=42,
        repeat_idx=0,
        env_valid=True,
        env_invalid_reasons=[],
        repro_valid=True,
        repro_invalid_reasons=[],
        llm_real=True,
        llm_invalid_reasons=[],
        usage_unavailable=False,
        usage_unavailable_reasons=[],
        model_base_url="https://example.test/v1",
        model_id="model-x",
        model_fallback_used=False,
        comparable_mode=True,
        comparability_invalid_reasons=[],
        metrics_schema_version=5,
        memlab_git_commit="abc",
        deerflow_git_commit="def",
        docker_image_digests={},
        infra_versions={},
        extra={"experiment_track": "mainline", "prompt_mode": "memory_beneficial"},
    )
    defaults = default_manifest_control_fields()
    for key, value in defaults.items():
        assert key in manifest
        if key not in {"track_id"}:
            assert manifest[key] == value
    assert manifest["track_id"] == "mainline:memory_beneficial"


def test_control_plane_invalidate_run_updates_manifest_and_ledgers(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest = build_manifest(
        run_id="run-1",
        variant_id="F1",
        workload_id="PUB-LONGBENCH_V2-FULL",
        seed=42,
        repeat_idx=0,
        env_valid=True,
        env_invalid_reasons=[],
        repro_valid=True,
        repro_invalid_reasons=[],
        llm_real=True,
        llm_invalid_reasons=[],
        usage_unavailable=False,
        usage_unavailable_reasons=[],
        model_base_url="https://example.test/v1",
        model_id="Pro/deepseek-ai/DeepSeek-V3.2",
        model_fallback_used=False,
        comparable_mode=True,
        comparability_invalid_reasons=[],
        metrics_schema_version=5,
        memlab_git_commit="abc",
        deerflow_git_commit="def",
        docker_image_digests={},
        infra_versions={},
        extra={
            "tasks_path": "benchmarks/public_main/longbench_v2/tasks_internal.jsonl",
            "prompt_mode": "memory_beneficial",
            "experiment_track": "mainline",
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    cp = ControlPlane(artifacts_root=tmp_path / "artifacts")
    key = BranchKey("longbench_v2", "mainline:memory_beneficial", "F1", 0)
    cp.register_branch(key)
    cp.invalidate_run(
        manifest_path,
        key=key,
        stage_id="pilot",
        anomaly_codes=["REPEAT_DIRECTION_CONFLICT"],
        reason="repeat deltas disagree",
        diagnostic_only=True,
    )
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["acceptance_status"] == "diagnostic_only"
    assert updated["rollback_root_stage"] == "pilot"
    assert updated["rerun_plan_id"]
    assert not accepted_for_official_reporting(updated)
    invalid_ledger = (tmp_path / "artifacts" / "ledger" / "invalid_run_ledger.jsonl").read_text(encoding="utf-8")
    assert "REPEAT_DIRECTION_CONFLICT" in invalid_ledger


def test_earliest_rollback_stage_uses_earliest_upstream_stage() -> None:
    assert earliest_rollback_stage(["SUMMARY_CHAIN_INCONSISTENT", "MODEL_LOCK_DRIFT"]) == "audit_protocol"
    assert earliest_rollback_stage(["SUMMARY_CHAIN_INCONSISTENT"]) == "smoke"


def test_sanitize_path_token_removes_windows_invalid_chars() -> None:
    assert sanitize_path_token("mainline:memory_beneficial") == "mainline-memory_beneficial"
