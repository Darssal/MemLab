from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from memlab.metrics.manifest import resolve_git_commit
from memlab.runtime_paths import deerflow_root


INFRA_REQUIRED_VARIANTS = {"V1", "F1", "F2", "A1", "A2"}


@dataclass
class EnvCheckResult:
    env_valid: bool
    env_invalid_reasons: list[str] = field(default_factory=list)
    memlab_git_commit: str | None = None
    deerflow_git_commit: str | None = None
    infra_versions: dict[str, str] = field(default_factory=dict)
    docker_image_digests: dict[str, str] = field(default_factory=dict)


def _run(cmd: list[str], *, timeout_s: float = 20.0) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        return True, out.strip()
    except Exception as exc:
        text = ""
        if hasattr(exc, "output"):
            if isinstance(exc.output, bytes):
                text = exc.output.decode("utf-8", errors="ignore").strip()
            elif isinstance(exc.output, str):
                text = exc.output.strip()
        return False, text or str(exc)


def _tcp_reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", "infra/docker-compose.yml", *args]


def _extract_host_port(url: str) -> tuple[str, int]:
    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    if p.port:
        return host, p.port
    return host, 443 if p.scheme == "https" else 80


def _docker_engine_ok() -> bool:
    ok, out = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout_s=10.0)
    if not ok:
        return False
    txt = out.strip().lower()
    if not txt:
        return False
    if "error" in txt or "request returned" in txt or "cannot connect" in txt:
        return False
    return True


def _service_running(service: str) -> bool:
    ok, out = _run(_compose_cmd("ps", "-q", service))
    if not ok or not out.strip():
        return False
    cid = out.strip()
    ok, state = _run(["docker", "inspect", "--format", "{{.State.Status}}", cid])
    return ok and state.strip() == "running"


def _service_healthy(service: str) -> bool:
    ok, out = _run(_compose_cmd("ps", "-q", service))
    if not ok or not out.strip():
        return False
    cid = out.strip()
    ok, health = _run(["docker", "inspect", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}", cid])
    if not ok:
        return False
    status = health.strip().lower()
    return status in {"healthy", "none"}


def _require_docker_infra(reasons: list[str]) -> None:
    if not _docker_engine_ok():
        reasons.append("Docker engine is not running.")
        return
    for svc in ("redis", "postgres", "qdrant"):
        if not _service_running(svc):
            reasons.append(f"Container '{svc}' is not running.")
            continue
        if not _service_healthy(svc):
            reasons.append(f"Container '{svc}' is not healthy.")
    if not _tcp_reachable("127.0.0.1", 6379):
        reasons.append("Redis is unreachable on 127.0.0.1:6379.")
    if not _tcp_reachable("127.0.0.1", 5432):
        reasons.append("Postgres is unreachable on 127.0.0.1:5432.")
    if not _tcp_reachable("127.0.0.1", 6333):
        reasons.append("Qdrant is unreachable on 127.0.0.1:6333.")


def _probe_infra_versions_and_digests() -> tuple[dict[str, str], dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    digests: dict[str, str] = {}
    errors: list[str] = []

    ok, redis_info = _run(_compose_cmd("exec", "-T", "redis", "redis-cli", "-h", "127.0.0.1", "-p", "6379", "INFO", "server"))
    if ok:
        redis_ver = ""
        for line in redis_info.splitlines():
            if line.startswith("redis_version:"):
                redis_ver = line.split(":", 1)[1].strip()
                break
        if redis_ver:
            versions["redis"] = redis_ver
        else:
            errors.append("Unable to parse Redis version.")
    else:
        errors.append(f"Unable to query Redis version: {redis_info}")

    ok, pg_ver = _run(_compose_cmd(
        "exec", "-T", "postgres", "sh", "-c",
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SHOW server_version;"',
    ))
    if ok and pg_ver:
        versions["postgres"] = pg_ver.strip()
    else:
        errors.append(f"Unable to query Postgres version: {pg_ver}")

    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            resp = client.get("http://127.0.0.1:6333/")
            resp.raise_for_status()
            body = resp.json()
        versions["qdrant"] = str(body.get("version", ""))
        if not versions["qdrant"]:
            errors.append("Unable to parse Qdrant version.")
    except Exception as exc:
        errors.append(f"Unable to query Qdrant version: {exc}")

    for svc in ("redis", "postgres", "qdrant"):
        ok, cid = _run(_compose_cmd("ps", "-q", svc))
        if not ok or not cid:
            errors.append(f"Unable to resolve container id for '{svc}'.")
            continue
        ok, image_ref = _run(["docker", "inspect", "--format", "{{.Config.Image}}", cid.strip()])
        if not ok or not image_ref:
            errors.append(f"Unable to resolve image reference for '{svc}'.")
            continue
        ok, digest = _run(["docker", "image", "inspect", image_ref.strip(), "--format", "{{index .RepoDigests 0}}"])
        digests[svc] = digest.strip() if ok and digest else image_ref.strip()

    return versions, digests, errors


def _probe_redis_lfu_support(reasons: list[str]) -> None:
    ok, out = _run(
        _compose_cmd(
            "exec",
            "-T",
            "redis",
            "redis-cli",
            "-h",
            "127.0.0.1",
            "-p",
            "6379",
            "CONFIG",
            "SET",
            "maxmemory-policy",
            "allkeys-lfu",
        )
    )
    if not ok or "OK" not in out:
        reasons.append("Redis does not support allkeys-lfu; refusing to run comparable variants.")


def _check_deerflow(
    *,
    langgraph_url: str,
    gateway_url: str,
    reasons: list[str],
) -> None:
    lg_host, lg_port = _extract_host_port(langgraph_url)
    gw_host, gw_port = _extract_host_port(gateway_url)
    if not _tcp_reachable(lg_host, lg_port):
        reasons.append(f"DeerFlow LangGraph endpoint is unreachable at {lg_host}:{lg_port}.")
    if not _tcp_reachable(gw_host, gw_port):
        reasons.append(f"DeerFlow Gateway endpoint is unreachable at {gw_host}:{gw_port}.")


def _read_simple_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _read_local_deerflow_runtime_lock(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "base_url": str(data.get("base_url", "") or "").strip().rstrip("/"),
        "model_id": str(data.get("model_id", "") or "").strip(),
        "fallback_model": str(data.get("fallback_model", "") or "").strip(),
        "gateway_pid": str(data.get("gateway_pid", "") or "").strip(),
        "langgraph_pid": str(data.get("langgraph_pid", "") or "").strip(),
    }


def _check_local_deerflow_lock_alignment(
    *,
    langgraph_url: str,
    gateway_url: str,
    expected_base_url: str,
    expected_model_id: str,
    reasons: list[str],
) -> None:
    lg_host, _ = _extract_host_port(langgraph_url)
    gw_host, _ = _extract_host_port(gateway_url)
    if {lg_host.lower(), gw_host.lower()} - {"127.0.0.1", "localhost"}:
        return

    expected_base = str(expected_base_url or os.getenv("OPENAI_BASE_URL", "") or "").strip().rstrip("/")
    expected_model = str(expected_model_id or os.getenv("OPENAI_MODEL", "") or "").strip()
    if not expected_base and not expected_model:
        return

    runtime_lock = _read_local_deerflow_runtime_lock(deerflow_root() / "logs/runtime.lock.json")
    if runtime_lock:
        runtime_mismatches: list[str] = []
        runtime_base = runtime_lock.get("base_url", "")
        runtime_model = runtime_lock.get("model_id", "")
        if expected_base and runtime_base and runtime_base != expected_base:
            runtime_mismatches.append(f"base_url={runtime_base} != {expected_base}")
        if expected_model and runtime_model and runtime_model != expected_model:
            runtime_mismatches.append(f"model_id={runtime_model} != {expected_model}")
        if runtime_mismatches:
            reasons.append(
                "Local DeerFlow runtime lock mismatch in deer-flow/logs/runtime.lock.json: "
                + "; ".join(runtime_mismatches)
            )
        return

    for env_path in (deerflow_root() / "backend/.env", deerflow_root() / ".env"):
        env_map = _read_simple_env_file(env_path)
        if not env_map:
            continue
        file_base = str(env_map.get("OPENAI_BASE_URL", "") or "").strip().rstrip("/")
        file_model = str(env_map.get("OPENAI_MODEL", "") or "").strip()
        mismatches: list[str] = []
        if expected_base and file_base and file_base != expected_base:
            mismatches.append(f"OPENAI_BASE_URL={file_base} != {expected_base}")
        if expected_model and file_model and file_model != expected_model:
            mismatches.append(f"OPENAI_MODEL={file_model} != {expected_model}")
        if mismatches:
            reasons.append(f"Local DeerFlow env drift in {env_path.as_posix()}: {'; '.join(mismatches)}")


def run_env_checks(
    *,
    variant_id: str,
    require_deerflow: bool,
    deerflow_langgraph_url: str,
    deerflow_gateway_url: str,
    locked_base_url: str | None = None,
    locked_model_id: str | None = None,
) -> EnvCheckResult:
    reasons: list[str] = []
    memlab_commit = resolve_git_commit(Path.cwd())
    deerflow_commit = resolve_git_commit(deerflow_root())
    if deerflow_commit is None:
        pinned = Path("artifacts/deerflow_commit.txt")
        if pinned.exists():
            txt = pinned.read_text(encoding="utf-8").strip()
            deerflow_commit = txt if txt else None
    versions: dict[str, str] = {}
    digests: dict[str, str] = {}
    infra_reasons: list[str] = []
    _require_docker_infra(infra_reasons)
    if not infra_reasons:
        versions, digests, infra_errors = _probe_infra_versions_and_digests()
        infra_reasons.extend(infra_errors)
        _probe_redis_lfu_support(infra_reasons)
    reasons.extend(infra_reasons)
    if require_deerflow:
        _check_deerflow(
            langgraph_url=deerflow_langgraph_url,
            gateway_url=deerflow_gateway_url,
            reasons=reasons,
        )
        _check_local_deerflow_lock_alignment(
            langgraph_url=deerflow_langgraph_url,
            gateway_url=deerflow_gateway_url,
            expected_base_url=str(locked_base_url or "").strip().rstrip("/"),
            expected_model_id=str(locked_model_id or "").strip(),
            reasons=reasons,
        )

    return EnvCheckResult(
        env_valid=len(reasons) == 0,
        env_invalid_reasons=reasons,
        memlab_git_commit=memlab_commit,
        deerflow_git_commit=deerflow_commit,
        infra_versions=versions,
        docker_image_digests=digests,
    )


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
