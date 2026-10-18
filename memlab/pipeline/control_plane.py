from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memlab.io import now_ms


BRANCH_STATUSES = (
    "pending",
    "running",
    "accepted",
    "invalid",
    "diagnostic_only",
    "blocked",
    "superseded",
)

STAGE_IDS = (
    "audit_data",
    "audit_protocol",
    "smoke",
    "pilot",
    "full",
    "aggregate",
    "promote",
)

ANOMALY_TO_ROLLBACK_STAGE = {
    "DATA_COUNT_MISMATCH": "audit_data",
    "FULL_SLICE_MIXED": "audit_data",
    "DUPLICATE_OR_MISSING_SAMPLE": "audit_data",
    "EMPTY_GOLD_OR_LABEL": "audit_data",
    "NON_MEMORY_DEPENDENT_DATASET": "audit_data",
    "PROMPT_PROFILE_MISMATCH": "audit_protocol",
    "METRIC_BINDING_ERROR": "audit_protocol",
    "MODEL_LOCK_DRIFT": "audit_protocol",
    "PROVIDER_PARAM_DRIFT": "audit_protocol",
    "SESSION_SCOPE_DRIFT": "audit_protocol",
    "MIXED_TRACK_COMPARISON": "audit_protocol",
    "DIRECT_EVIDENCE_LEAKAGE": "audit_protocol",
    "TRACK_GOAL_MISMATCH": "audit_protocol",
    "CROSS_SAMPLE_MEMORY_CONTAMINATION": "smoke",
    "EVENTLOG_MISSING": "smoke",
    "SUMMARY_CHAIN_INCONSISTENT": "smoke",
    "INVALID_RATE_SPIKE": "smoke",
    "BASELINE_OUT_OF_BAND": "pilot",
    "RAW_RUN_TABLE_MISMATCH": "full",
    "REPEAT_DIRECTION_CONFLICT": "pilot",
    "REPEAT_VARIANCE_ABNORMAL": "pilot",
    "UNEXPLAINED_ORDERING_REVERSAL": "pilot",
}

STRUCTURAL_ANOMALIES = {
    "DATA_COUNT_MISMATCH",
    "FULL_SLICE_MIXED",
    "DUPLICATE_OR_MISSING_SAMPLE",
    "EMPTY_GOLD_OR_LABEL",
    "NON_MEMORY_DEPENDENT_DATASET",
    "PROMPT_PROFILE_MISMATCH",
    "METRIC_BINDING_ERROR",
    "MODEL_LOCK_DRIFT",
    "PROVIDER_PARAM_DRIFT",
    "SESSION_SCOPE_DRIFT",
    "CROSS_SAMPLE_MEMORY_CONTAMINATION",
    "EVENTLOG_MISSING",
    "SUMMARY_CHAIN_INCONSISTENT",
    "MIXED_TRACK_COMPARISON",
    "DIRECT_EVIDENCE_LEAKAGE",
    "TRACK_GOAL_MISMATCH",
    "RAW_RUN_TABLE_MISMATCH",
}

DATASET_ALIASES = {
    "locomo_preprocess": "locomo",
    "longmemeval_24k": "longmemeval",
    "squad_v11_dev": "squad_v11",
}


@dataclass(frozen=True)
class BranchKey:
    dataset_id: str
    track_id: str
    variant_id: str
    repeat_idx: int

    @property
    def branch_id(self) -> str:
        parts = [self.dataset_id, self.track_id, self.variant_id, f"r{int(self.repeat_idx)}"]
        return "::".join([str(p).strip() for p in parts])


@dataclass
class StageStatusRecord:
    branch_id: str
    dataset_id: str
    track_id: str
    variant_id: str
    repeat_idx: int
    stage_id: str
    status: str
    ts_ms: int
    run_id: str = ""
    note: str = ""
    anomaly_codes: list[str] | None = None
    rollback_root_stage: str | None = None
    rerun_plan_id: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{now_ms()}")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="ignore")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_paths(paths: list[str | Path]) -> str:
    tokens: list[str] = []
    for raw in paths:
        path = Path(raw)
        tokens.append(f"{path.as_posix()}::{sha1_file(path)}")
    return sha1_text("|".join(tokens))


def normalize_dataset_id(dataset_id: str) -> str:
    raw = str(dataset_id or "").strip().lower()
    return DATASET_ALIASES.get(raw, raw)


def sanitize_path_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    clean = "".join(ch if ch not in '<>:"/\\|?*' else "-" for ch in raw)
    clean = clean.strip().strip(".")
    return clean or "unknown"


def infer_dataset_id_from_manifest(manifest: dict[str, Any]) -> str:
    extra = manifest.get("extra") or {}
    if extra.get("dataset_id"):
        return normalize_dataset_id(str(extra.get("dataset_id")))
    tasks_path = Path(str(extra.get("tasks_path", "")))
    if tasks_path.parent.name:
        return normalize_dataset_id(tasks_path.parent.name)
    workload_id = str(manifest.get("workload_id", ""))
    if workload_id.upper().startswith("PUB-"):
        return normalize_dataset_id(workload_id.split("-", 2)[1].lower())
    return normalize_dataset_id(str(manifest.get("dataset", "unknown")))


def infer_track_id_from_manifest(manifest: dict[str, Any]) -> str:
    if manifest.get("track_id"):
        return str(manifest.get("track_id"))
    extra = manifest.get("extra") or {}
    if extra.get("track_id"):
        return str(extra.get("track_id"))
    experiment_track = str(extra.get("experiment_track", "mainline") or "mainline").strip()
    prompt_mode = str(extra.get("prompt_mode", "memory_neutral") or "memory_neutral").strip()
    return f"{experiment_track}:{prompt_mode}"


def infer_branch_key_from_manifest(manifest: dict[str, Any]) -> BranchKey:
    return BranchKey(
        dataset_id=infer_dataset_id_from_manifest(manifest),
        track_id=infer_track_id_from_manifest(manifest),
        variant_id=str(manifest.get("variant_id", "")).strip(),
        repeat_idx=int(manifest.get("repeat_idx", 0)),
    )


def accepted_for_official_reporting(manifest: dict[str, Any]) -> bool:
    acceptance_status = str(manifest.get("acceptance_status", "") or "").strip().lower()
    diagnostic_only = bool(manifest.get("diagnostic_only", False))
    summary_chain_valid = manifest.get("summary_chain_valid", None)
    superseded_by = str(manifest.get("superseded_by_run_id", "") or "").strip()
    if acceptance_status:
        return (
            acceptance_status == "accepted"
            and not diagnostic_only
            and not superseded_by
            and bool(summary_chain_valid)
        )
    # Legacy fallback for older runs that predate control-plane metadata.
    return (
        bool(manifest.get("env_valid", False))
        and bool(manifest.get("repro_valid", False))
        and bool(manifest.get("llm_real", False))
    )


def merge_control_fields(manifest: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(manifest)
    for key, value in updates.items():
        merged[key] = value
    return merged


class ControlPlane:
    def __init__(self, *, artifacts_root: str | Path = "artifacts") -> None:
        self.artifacts_root = Path(artifacts_root)
        self.ledger_dir = self.artifacts_root / "ledger"
        self.audit_dir = self.artifacts_root / "audit"
        self.branch_registry_path = self.ledger_dir / "branch_registry.json"
        self.invalid_run_ledger_path = self.ledger_dir / "invalid_run_ledger.jsonl"
        self.rollback_ledger_path = self.ledger_dir / "rollback_ledger.jsonl"
        self.rerun_plan_ledger_path = self.ledger_dir / "rerun_plan_ledger.jsonl"
        self.acceptance_status_path = self.ledger_dir / "acceptance_status.jsonl"
        self.anomaly_classification_path = self.ledger_dir / "anomaly_classification.jsonl"
        self.stage_status_path = self.ledger_dir / "stage_status.jsonl"

    def _load_registry(self) -> dict[str, Any]:
        if not self.branch_registry_path.exists():
            return {"branches": {}}
        return _read_json(self.branch_registry_path)

    def _write_registry(self, registry: dict[str, Any]) -> None:
        _write_json(self.branch_registry_path, registry)

    def register_branch(self, key: BranchKey, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        registry = self._load_registry()
        branches = registry.setdefault("branches", {})
        record = branches.get(key.branch_id) or {
            "branch_id": key.branch_id,
            "dataset_id": key.dataset_id,
            "track_id": key.track_id,
            "variant_id": key.variant_id,
            "repeat_idx": int(key.repeat_idx),
            "status": "pending",
            "created_at_ms": now_ms(),
            "updated_at_ms": now_ms(),
            "metadata": {},
            "latest_run_id": "",
            "latest_stage_id": "",
            "latest_rerun_plan_id": "",
            "anomaly_count": 0,
        }
        if metadata:
            record["metadata"] = {**record.get("metadata", {}), **metadata}
        record["updated_at_ms"] = now_ms()
        branches[key.branch_id] = record
        self._write_registry(registry)
        return record

    def update_branch(
        self,
        key: BranchKey,
        *,
        status: str | None = None,
        latest_run_id: str | None = None,
        latest_stage_id: str | None = None,
        latest_rerun_plan_id: str | None = None,
        increment_anomaly_count: bool = False,
    ) -> None:
        registry = self._load_registry()
        branches = registry.setdefault("branches", {})
        record = branches.get(key.branch_id) or self.register_branch(key)
        if status:
            record["status"] = status
        if latest_run_id is not None:
            record["latest_run_id"] = latest_run_id
        if latest_stage_id is not None:
            record["latest_stage_id"] = latest_stage_id
        if latest_rerun_plan_id is not None:
            record["latest_rerun_plan_id"] = latest_rerun_plan_id
        if increment_anomaly_count:
            record["anomaly_count"] = int(record.get("anomaly_count", 0)) + 1
        record["updated_at_ms"] = now_ms()
        branches[key.branch_id] = record
        self._write_registry(registry)

    def record_stage_status(
        self,
        key: BranchKey,
        *,
        stage_id: str,
        status: str,
        run_id: str = "",
        note: str = "",
        anomaly_codes: list[str] | None = None,
        rollback_root_stage: str | None = None,
        rerun_plan_id: str | None = None,
    ) -> None:
        record = StageStatusRecord(
            branch_id=key.branch_id,
            dataset_id=key.dataset_id,
            track_id=key.track_id,
            variant_id=key.variant_id,
            repeat_idx=int(key.repeat_idx),
            stage_id=stage_id,
            status=status,
            ts_ms=now_ms(),
            run_id=run_id,
            note=note,
            anomaly_codes=list(anomaly_codes or []),
            rollback_root_stage=rollback_root_stage,
            rerun_plan_id=rerun_plan_id,
        )
        _append_jsonl(self.stage_status_path, asdict(record))
        _append_jsonl(
            self.acceptance_status_path,
            {
                "branch_id": key.branch_id,
                "run_id": run_id,
                "stage_id": stage_id,
                "status": status,
                "ts_ms": record.ts_ms,
                "note": note,
            },
        )
        self.update_branch(
            key,
            status=status,
            latest_run_id=run_id or None,
            latest_stage_id=stage_id,
            latest_rerun_plan_id=rerun_plan_id,
            increment_anomaly_count=bool(anomaly_codes),
        )

    def audit_report_path(self, key: BranchKey, stage_id: str) -> Path:
        return (
            self.audit_dir
            / sanitize_path_token(key.dataset_id)
            / sanitize_path_token(key.track_id)
            / sanitize_path_token(key.variant_id)
            / f"r{int(key.repeat_idx)}"
            / f"{sanitize_path_token(stage_id)}.json"
        )

    def write_audit_report(self, key: BranchKey, *, stage_id: str, report: dict[str, Any]) -> Path:
        path = self.audit_report_path(key, stage_id)
        _write_json(path, report)
        self.register_branch(key, metadata={"latest_audit_report": path.as_posix()})
        return path

    def annotate_manifest(self, manifest_path: str | Path, **updates: Any) -> dict[str, Any]:
        path = Path(manifest_path)
        manifest = _read_json(path)
        manifest = merge_control_fields(manifest, updates)
        _write_json(path, manifest)
        return manifest

    def create_rerun_plan(
        self,
        key: BranchKey,
        *,
        rollback_root_stage: str,
        anomaly_codes: list[str],
        invalidated_run_ids: list[str],
        why: str,
        what: str = "",
        downstream_impact: list[str] | None = None,
    ) -> str:
        rerun_plan_id = f"rerun-{sha1_text(key.branch_id + why + str(now_ms()))[:12]}"
        row = {
            "rerun_plan_id": rerun_plan_id,
            "branch_id": key.branch_id,
            "dataset_id": key.dataset_id,
            "track_id": key.track_id,
            "variant_id": key.variant_id,
            "repeat_idx": int(key.repeat_idx),
            "rollback_root_stage": rollback_root_stage,
            "anomaly_codes": list(sorted(set(anomaly_codes))),
            "invalidated_run_ids": list(invalidated_run_ids),
            "why": why,
            "what": what,
            "downstream_impact": list(downstream_impact or []),
            "status": "pending",
            "created_at_ms": now_ms(),
        }
        _append_jsonl(self.rerun_plan_ledger_path, row)
        self.update_branch(key, latest_rerun_plan_id=rerun_plan_id)
        return rerun_plan_id

    def record_anomaly(
        self,
        key: BranchKey,
        *,
        run_id: str,
        stage_id: str,
        anomaly_codes: list[str],
        reason: str,
        diagnostic_only: bool = False,
        rerun_plan_id: str | None = None,
    ) -> str:
        rollback_root_stage = earliest_rollback_stage(anomaly_codes)
        row = {
            "branch_id": key.branch_id,
            "dataset_id": key.dataset_id,
            "track_id": key.track_id,
            "variant_id": key.variant_id,
            "repeat_idx": int(key.repeat_idx),
            "run_id": run_id,
            "stage_id": stage_id,
            "anomaly_codes": list(sorted(set(anomaly_codes))),
            "reason": reason,
            "diagnostic_only": bool(diagnostic_only),
            "rollback_root_stage": rollback_root_stage,
            "rerun_plan_id": rerun_plan_id,
            "ts_ms": now_ms(),
        }
        _append_jsonl(self.anomaly_classification_path, row)
        _append_jsonl(
            self.rollback_ledger_path,
            {
                **row,
                "action": "rollback",
            },
        )
        return rollback_root_stage

    def invalidate_run(
        self,
        manifest_path: str | Path,
        *,
        key: BranchKey,
        stage_id: str,
        anomaly_codes: list[str],
        reason: str,
        diagnostic_only: bool = False,
        supersedes_run_id: str | None = None,
    ) -> tuple[dict[str, Any], str, str]:
        path = Path(manifest_path)
        manifest = _read_json(path)
        run_id = str(manifest.get("run_id", path.parent.name))
        rollback_root_stage = earliest_rollback_stage(anomaly_codes)
        rerun_plan_id = self.create_rerun_plan(
            key,
            rollback_root_stage=rollback_root_stage,
            anomaly_codes=anomaly_codes,
            invalidated_run_ids=[run_id],
            why=reason,
            what=f"Re-run branch from {rollback_root_stage} after fixing {', '.join(sorted(set(anomaly_codes)))}.",
        )
        status = "diagnostic_only" if diagnostic_only else "invalid"
        manifest = self.annotate_manifest(
            path,
            acceptance_status=status,
            diagnostic_only=bool(diagnostic_only),
            invalid_reason_codes=list(sorted(set(anomaly_codes))),
            anomaly_classification=list(sorted(set(anomaly_codes))),
            rollback_root_stage=rollback_root_stage,
            rerun_plan_id=rerun_plan_id,
            supersedes_run_id=supersedes_run_id,
            summary_chain_valid=bool(manifest.get("summary_chain_valid", False)),
        )
        self.record_anomaly(
            key,
            run_id=run_id,
            stage_id=stage_id,
            anomaly_codes=anomaly_codes,
            reason=reason,
            diagnostic_only=diagnostic_only,
            rerun_plan_id=rerun_plan_id,
        )
        _append_jsonl(
            self.invalid_run_ledger_path,
            {
                "run_id": run_id,
                "branch_id": key.branch_id,
                "status": status,
                "stage_id": stage_id,
                "anomaly_codes": list(sorted(set(anomaly_codes))),
                "reason": reason,
                "rollback_root_stage": rollback_root_stage,
                "rerun_plan_id": rerun_plan_id,
                "ts_ms": now_ms(),
            },
        )
        self.record_stage_status(
            key,
            stage_id=stage_id,
            status=status,
            run_id=run_id,
            note=reason,
            anomaly_codes=anomaly_codes,
            rollback_root_stage=rollback_root_stage,
            rerun_plan_id=rerun_plan_id,
        )
        return manifest, rollback_root_stage, rerun_plan_id


def earliest_rollback_stage(anomaly_codes: list[str]) -> str:
    stages = [ANOMALY_TO_ROLLBACK_STAGE.get(code, "audit_protocol") for code in anomaly_codes]
    order = {name: idx for idx, name in enumerate(STAGE_IDS)}
    return sorted(stages, key=lambda name: order.get(name, 999))[0] if stages else "audit_protocol"


def default_manifest_control_fields() -> dict[str, Any]:
    return {
        "acceptance_status": "pending",
        "diagnostic_only": False,
        "invalid_reason_codes": [],
        "anomaly_classification": [],
        "rollback_parent_run_id": None,
        "rollback_root_stage": None,
        "rerun_plan_id": None,
        "supersedes_run_id": None,
        "superseded_by_run_id": None,
        "input_fingerprint": None,
        "protocol_fingerprint": None,
        "dataset_fingerprint": None,
        "summary_chain_valid": False,
        "track_id": "",
    }
