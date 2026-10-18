from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path
from typing import Any

from memlab.config import load_config
from memlab.envcheck import run_env_checks, write_manifest
from memlab.io import json_dump, jsonl_read, now_ms
from memlab.memory.types import ContextBudget, MemoryItem
from memlab.memory_manager.manager import MultiLevelMemoryManager
from memlab.metrics.eventlog import EventLogWriter
from memlab.metrics.manifest import build_manifest
from memlab.runner_core import apply_variant, compute_run_summary
from memlab.runner_core import normalize_variant
from memlab.session_logging import current_agent_id as _CURRENT_AGENT_ID
from memlab.session_logging import make_operation_logger as _make_op_logger


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/memlab.default.yaml")
    ap.add_argument("--variant", type=str, required=True, help="B0|B1|V1|F1|F2 (legacy aliases: A1|A2)")
    ap.add_argument("--n_sessions", type=int, default=200)
    ap.add_argument("--llm_max_inflight", type=int, default=0)
    ap.add_argument("--worker_processes", type=int, default=1)
    ap.add_argument("--sessions_per_process", type=int, default=200)
    ap.add_argument("--repeat_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workload_id", type=str, default="SYN-200")
    ap.add_argument("--deerflow_langgraph_url", type=str, default="http://127.0.0.1:2024")
    ap.add_argument("--deerflow_gateway_url", type=str, default="http://127.0.0.1:8001")
    ap.add_argument("--trace_path", type=str, default="")
    ap.add_argument("--ops_per_session", type=int, default=1000)
    ap.add_argument("--track_id", type=str, default="stub:default")
    return ap.parse_args()


def _build_deterministic_stub_tasks(*, n_sessions: int, ops_per_session: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    total = n_sessions * ops_per_session
    hotset = max(1000, n_sessions * 20)
    for i in range(total):
        key_idx = rng.randrange(hotset)
        payload_bytes = rng.randint(128, 8192)
        tasks.append(
            {
                "task_id": f"syn_{i:07d}",
                "key": f"k-{key_idx:06d}",
                "bytes": payload_bytes,
                "importance": rng.random(),
            }
        )
    return tasks


def _load_trace_tasks(path: str) -> list[dict[str, Any]]:
    rows = jsonl_read(path)
    tasks: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        tasks.append(
            {
                "task_id": f"trace_{i:07d}",
                "key": str(row.get("key", f"k-{i:06d}")),
                "bytes": int(row.get("bytes", 256)),
                "importance": float(row.get("importance", 0.5)),
            }
        )
    return tasks


def _fake_text(key: str, n_bytes: int) -> str:
    prefix = f"key={key} "
    fill = max(0, n_bytes - len(prefix.encode("utf-8")))
    return prefix + ("x" * fill)


async def _session_worker(
    *, queue, session_index, run_id, memory, event_writer, task_rows, mem_rows, cfg,
) -> None:
    token = _CURRENT_AGENT_ID.set(f"agent_{session_index + 1:04d}")
    try:
        await _session_worker_body(
            queue=queue, session_index=session_index, run_id=run_id,
            memory=memory, event_writer=event_writer, task_rows=task_rows,
            mem_rows=mem_rows, cfg=cfg,
        )
    finally:
        _CURRENT_AGENT_ID.reset(token)


async def _session_worker_body(
    *,
    queue: asyncio.Queue[dict[str, Any]],
    session_index: int,
    run_id: str,
    memory: MultiLevelMemoryManager,
    event_writer: EventLogWriter,
    task_rows: list[dict[str, Any]],
    mem_rows: list[dict[str, Any]],
    cfg,
) -> None:
    agent_id = f"agent_{session_index + 1:04d}"
    session_id = f"{run_id}-{agent_id}"

    while True:
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        task_id = str(task["task_id"])
        key = str(task["key"])
        payload_size = int(task.get("bytes", 256))
        task_enqueued_ts = int(task.get("task_enqueued_ts", now_ms()))
        task_start_ts = now_ms()
        status = "success"
        err = ""
        accuracy: float | None = None
        tokens_in = 0
        tokens_out = 0
        injected_tokens = 0
        injected_items = 0
        retrieve_skipped_reason: str | None = None
        embedding_cache_hit: bool | None = None
        retrieval_cache_hit: bool | None = None
        try:
            text = _fake_text(key, payload_size)
            item = MemoryItem(
                key=key,
                agent_id=agent_id,
                kind="fact",
                text=text,
                bytes=payload_size,
                importance=float(task.get("importance", 0.5)),
                created_at_ms=now_ms(),
                last_access_ms=now_ms(),
                access_count=1,
                metadata={"role": "system", "workload_hint": 1.0},
            )
            await memory.put(item, ttl_s=cfg.memory.l1.ttl_s if cfg.memory.l1.enabled else None)
            _ = await memory.get(agent_id=agent_id, query=key, k=1, mode="hybrid")
            packed = await memory.assemble_context(
                agent_id=agent_id,
                user_query=f"lookup {key}",
                budget=ContextBudget(
                    max_prompt_tokens=cfg.memory.l0.max_prompt_tokens,
                    max_memory_injection_tokens=cfg.memory.l0.max_memory_injection_tokens,
                    keep_recent_messages=cfg.memory.l0.keep_recent_messages,
                ),
            )
            injected_tokens = int(packed.trace.get("injection_tokens", 0))
            injected_items = int(packed.trace.get("injection_item_count", 0))
            retrieve_skipped_reason = packed.trace.get("retrieve_skipped_reason")
            embedding_cache_hit = packed.trace.get("embedding_cache_hit")
            retrieval_cache_hit = packed.trace.get("retrieval_cache_hit")
            tokens_in = max(1, int(packed.trace.get("total_tokens", 0)))
            tokens_out = 1
        except Exception as exc:
            status = "error"
            err = type(exc).__name__
        task_end_ts = now_ms()
        queue_wait_ms = max(0.0, float(task_start_ts - task_enqueued_ts))
        raw_processing_ms = max(0.0, float(task_end_ts - task_start_ts))
        processing_ms = raw_processing_ms
        proc_llm_ms = 0.0
        proc_non_llm_ms = processing_ms
        end_to_end_ms = max(0.0, float(task_end_ts - task_enqueued_ts))
        row = {
            "run_id": run_id,
            "op_type": "task",
            "op_name": "task_result",
            "task_id": task_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "start_ts": task_start_ts,
            "end_ts": task_end_ts,
            "task_enqueued_ts": task_enqueued_ts,
            "task_start_ts": task_start_ts,
            "task_end_ts": task_end_ts,
            "queue_wait_ms": float(queue_wait_ms),
            "processing_ms": float(processing_ms),
            "proc_whole_ms": float(processing_ms),
            "proc_llm_ms": float(proc_llm_ms),
            "proc_non_llm_ms": float(proc_non_llm_ms),
            "end_to_end_ms": float(end_to_end_ms),
            "status": status,
            "accuracy_applicable": False,
            "accuracy": accuracy,
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "latency_ms": float(end_to_end_ms),
            "context_injected_tokens": int(injected_tokens),
            "context_injected_items": int(injected_items),
            "retrieve_skipped_reason": retrieve_skipped_reason,
            "embedding_cache_hit": embedding_cache_hit,
            "retrieval_cache_hit": retrieval_cache_hit,
            "llm_queue_wait_ms": 0.0,
            "llm_latency_ms": None,
            "llm_request_latency_ms": None,
            "error": err,
        }
        task_rows.append(row)
        event_writer.append(row)
        queue.task_done()


async def _run() -> int:
    args = _parse_args()
    variant, legacy_alias = normalize_variant(args.variant)
    run_id = f"{variant.lower()}-syn-r{args.repeat_idx}-{now_ms()}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = apply_variant(cfg, variant, legacy_alias=legacy_alias)
    cfg.experiment.seed = int(args.seed)
    cfg.experiment.n_agents = int(args.n_sessions)

    env = run_env_checks(
        variant_id=variant,
        require_deerflow=False,
        deerflow_langgraph_url=args.deerflow_langgraph_url,
        deerflow_gateway_url=args.deerflow_gateway_url,
    )
    manifest = build_manifest(
        run_id=run_id,
        variant_id=variant,
        workload_id=str(args.workload_id),
        seed=int(args.seed),
        repeat_idx=int(args.repeat_idx),
        env_valid=env.env_valid,
        env_invalid_reasons=env.env_invalid_reasons,
        repro_valid=bool(env.memlab_git_commit),
        repro_invalid_reasons=[] if env.memlab_git_commit else ["memlab git commit hash is unavailable."],
        llm_real=None,
        llm_invalid_reasons=[],
        usage_unavailable=None,
        usage_unavailable_reasons=[],
        model_base_url=None,
        model_id=None,
        model_fallback_used=False,
        comparable_mode=True,
        comparability_invalid_reasons=[],
        metrics_schema_version=3,
        memlab_git_commit=env.memlab_git_commit,
        deerflow_git_commit=env.deerflow_git_commit,
        docker_image_digests=env.docker_image_digests,
        infra_versions=env.infra_versions,
        track_id=str(args.track_id or "stub:default"),
        extra={
            "runner": "runner_stub",
            "track_id": str(args.track_id or "stub:default"),
            "worker_processes": int(args.worker_processes),
            "sessions_per_process": int(args.sessions_per_process),
            "ops_per_session": int(args.ops_per_session),
            "trace_path": str(args.trace_path or ""),
            "legacy_variant_alias": legacy_alias,
        },
    )
    write_manifest(run_dir / "manifest.json", manifest)

    fatal_reasons = list(env.env_invalid_reasons)
    if fatal_reasons:
        print("STOP: required services missing; refusing fallback.")
        for reason in fatal_reasons:
            print(f"- {reason}")
        return 1

    if args.trace_path and Path(args.trace_path).exists():
        tasks = _load_trace_tasks(args.trace_path)
    else:
        tasks = _build_deterministic_stub_tasks(
            n_sessions=int(args.n_sessions),
            ops_per_session=int(args.ops_per_session),
            seed=int(args.seed) + int(args.repeat_idx),
        )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for task in tasks:
        enriched = dict(task)
        enriched["task_enqueued_ts"] = now_ms()
        await q.put(enriched)

    event_writer = EventLogWriter(run_dir / "eventlog.jsonl")
    task_rows: list[dict[str, Any]] = []
    mem_rows: list[dict[str, Any]] = []
    memory = MultiLevelMemoryManager(
        cfg, op_logger=_make_op_logger(run_id=run_id, mem_rows=mem_rows, event_writer=event_writer)
    )
    await memory.init()

    capacity = max(1, int(args.worker_processes) * int(args.sessions_per_process))
    if capacity < int(args.n_sessions):
        reason = (
            "Invalid session topology: "
            f"worker_processes*sessions_per_process={capacity} is smaller than n_sessions={int(args.n_sessions)}."
        )
        manifest["env_valid"] = False
        manifest.setdefault("env_invalid_reasons", [])
        manifest["env_invalid_reasons"].append(reason)
        write_manifest(run_dir / "manifest.json", manifest)
        print("STOP: required services missing; refusing fallback.")
        print(f"- {reason}")
        await memory.close()
        return 1
    worker_count = int(args.n_sessions)
    workers = [
        _session_worker(
            queue=q,
            session_index=i,
            run_id=run_id,
            memory=memory,
            event_writer=event_writer,
            task_rows=task_rows,
            mem_rows=mem_rows,
            cfg=cfg,
        )
        for i in range(worker_count)
    ]
    await asyncio.gather(*workers)
    await memory.close()

    summary = compute_run_summary(
        run_id=run_id,
        variant_id=variant,
        workload_id=str(args.workload_id),
        n_sessions=int(args.n_sessions),
        tasks=task_rows,
        mem_ops=mem_rows,
    )
    json_dump(run_dir / "summary.json", summary)
    manifest["summary_chain_valid"] = bool((run_dir / "eventlog.jsonl").exists() and (run_dir / "summary.json").exists())
    write_manifest(run_dir / "manifest.json", manifest)

    print(f"Run complete: {run_id}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Event log: {run_dir / 'eventlog.jsonl'}")
    print(f"Summary: {run_dir / 'summary.json'}")
    return 0


def main() -> None:
    rc = asyncio.run(_run())
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
