from __future__ import annotations

import os
import platform
import subprocess
import ctypes
from pathlib import Path
from typing import Any

from memlab.pipeline.control_plane import default_manifest_control_fields


def resolve_git_commit(repo_path: str | Path) -> str | None:
    repo = Path(repo_path)
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return out.decode("utf-8").strip()


def _get_total_ram_bytes() -> int:
    if platform.system().lower().startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
    try:
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_PHYS_PAGES"))  # type: ignore[attr-defined]
            page_size = int(os.sysconf("SC_PAGE_SIZE"))  # type: ignore[attr-defined]
            return pages * page_size
    except Exception:
        pass
    return 0


def build_manifest(
    *,
    run_id: str,
    variant_id: str,
    workload_id: str,
    seed: int,
    repeat_idx: int,
    env_valid: bool,
    env_invalid_reasons: list[str],
    repro_valid: bool,
    repro_invalid_reasons: list[str],
    llm_real: bool | None,
    llm_invalid_reasons: list[str],
    usage_unavailable: bool | None,
    usage_unavailable_reasons: list[str],
    model_base_url: str | None,
    model_id: str | None,
    model_fallback_used: bool,
    comparable_mode: bool,
    comparability_invalid_reasons: list[str],
    metrics_schema_version: int,
    memlab_git_commit: str | None,
    deerflow_git_commit: str | None,
    docker_image_digests: dict[str, str],
    infra_versions: dict[str, str],
    extra: dict[str, Any] | None = None,
    track_id: str | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "variant_id": variant_id,
        "workload_id": workload_id,
        "seed": int(seed),
        "repeat_idx": int(repeat_idx),
        "env_valid": bool(env_valid),
        "env_invalid_reasons": list(env_invalid_reasons),
        "repro_valid": bool(repro_valid),
        "repro_invalid_reasons": list(repro_invalid_reasons),
        "llm_real": llm_real if llm_real is None else bool(llm_real),
        "llm_invalid_reasons": list(llm_invalid_reasons),
        "usage_unavailable": usage_unavailable if usage_unavailable is None else bool(usage_unavailable),
        "usage_unavailable_reasons": list(usage_unavailable_reasons),
        "model_base_url": model_base_url,
        "model_id": model_id,
        "model_fallback_used": bool(model_fallback_used),
        "comparable_mode": bool(comparable_mode),
        "comparability_invalid_reasons": list(comparability_invalid_reasons),
        "metrics_schema_version": int(metrics_schema_version),
        "memlab_git_commit": memlab_git_commit,
        "deerflow_git_commit": deerflow_git_commit,
        "docker_image_digests": docker_image_digests,
        "infra_versions": infra_versions,
        "host_info": {
            "os": platform.platform(),
            "cpu_cores": os.cpu_count() or 0,
            "ram_bytes": _get_total_ram_bytes(),
            "python_version": platform.python_version(),
        },
        "extra": extra or {},
        "track_id": str(track_id or ""),
    }
    for key, value in default_manifest_control_fields().items():
        manifest.setdefault(key, value)
    if control:
        manifest.update(control)
    if not manifest.get("track_id"):
        extra_map = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
        prompt_mode = str((extra_map or {}).get("prompt_mode", "") or "").strip()
        experiment_track = str((extra_map or {}).get("experiment_track", "") or "").strip()
        if experiment_track or prompt_mode:
            manifest["track_id"] = f"{experiment_track or 'mainline'}:{prompt_mode or 'memory_neutral'}"
    return manifest
