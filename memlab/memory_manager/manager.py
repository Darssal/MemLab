from __future__ import annotations

import asyncio
import contextlib
import copy
import re
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from typing import Any, Callable

from memlab.config import MemlabConfig
from memlab.io import now_ms
from memlab.memory.api import MemoryManager
from memlab.memory.types import ContextBudget, ContextPack, MemoryItem, PlacementDecision
from memlab.tools.embedding import cosine, hash_embedding

from .l0 import assemble_l0_context
from .l1 import L1RedisClient
from .l2 import L2PostgresStore
from .l3 import L3ObjectStore
from .vector import QdrantVectorStore

OpLogger = Callable[[dict[str, Any]], None]
DIRECT_KEY_RE = re.compile(r"^(?:lookup\s+)?(k-\d{6})$", re.IGNORECASE)
RECALL_MARKERS = ("previous", "remember", "earlier", "as we said", "last time", "before")
DIRECT_EVIDENCE_QUERY_MARKERS = (
    "find the secret_code",
    "find the secret code",
    "document titled",
)


def _point_id(agent_id: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memlab:{agent_id}:{key}"))


class MultiLevelMemoryManager(MemoryManager):
    def __init__(self, config: MemlabConfig, *, op_logger: OpLogger | None = None) -> None:
        self.cfg = config
        self.op_logger = op_logger
        self.l1 = L1RedisClient(config.services.redis_url) if config.memory.l1.enabled else None
        self.l2 = L2PostgresStore(config.services.pg_uri) if config.memory.l2.enabled else None
        self.vector = (
            QdrantVectorStore(
                config.services.qdrant_url,
                config.services.qdrant_collection,
                vector_size=64,
                shard_number=int(getattr(config.memory.l2, "qdrant_shard_number", 1)),
            )
            if config.memory.l2.enabled and bool(getattr(config.memory.l2, "vector_enabled", True))
            else None
        )
        self.l3 = (
            L3ObjectStore(config.memory.l3.root_dir)
            if getattr(config.memory.l3, "enabled", False)
            else None
        )
        self._recent_messages: dict[str, list[MemoryItem]] = defaultdict(list)
        self._l1_write_queue: asyncio.Queue[MemoryItem] | None = (
            asyncio.Queue(maxsize=32768)
            if self.cfg.memory.l1.enabled and self.l1 is not None
            else None
        )
        self._l1_writer_task: asyncio.Task[None] | None = None
        self._l1_writer_error: RuntimeError | None = None
        self._l1_batch_size = 256
        self._l1_flush_interval_ms = 20
        self._l2_write_queue: asyncio.Queue[MemoryItem] | None = (
            asyncio.Queue(maxsize=16384)
            if self.cfg.memory.l2.enabled and self.l2 is not None
            else None
        )
        self._l2_writer_task: asyncio.Task[None] | None = None
        self._l2_writer_error: RuntimeError | None = None
        self._l2_batch_size = max(1, int(self.cfg.memory.l2.write_batch_size))
        self._l2_flush_interval_ms = max(1, int(self.cfg.memory.l2.write_flush_interval_ms))
        self._l2_max_batch_per_flush = max(1, int(getattr(self.cfg.memory.l2, "max_batch_per_flush", self._l2_batch_size)))
        self._l2_max_flush_ms = max(1, int(getattr(self.cfg.memory.l2, "max_flush_ms", self._l2_flush_interval_ms)))
        self._l2_min_importance = float(self.cfg.memory.l2.min_write_importance)
        self._l2_write_retry_count = max(1, int(getattr(self.cfg.memory.l2, "write_retry_count", 3)))
        self._l2_write_retry_backoff_ms = max(50, int(getattr(self.cfg.memory.l2, "write_retry_backoff_ms", 250)))
        self._l2_drop_queue_depth_threshold = max(1, int(getattr(self.cfg.memory.l2, "drop_queue_depth_threshold", 12000)))
        self._l2_backpressure_drop_importance_threshold = float(
            getattr(self.cfg.memory.l2, "backpressure_drop_importance_threshold", max(self._l2_min_importance, 0.85))
        )
        self._dropped_writes_count = 0
        self._query_embedding_cache: OrderedDict[tuple[str, str], tuple[list[float], int]] = OrderedDict()
        self._query_embedding_cache_limit = 8192
        self._embedding_cache_ttl_ms = max(1000, int(getattr(self.cfg.memory.l2, "embedding_cache_ttl_ms", 120000)))
        self._retrieval_cache: OrderedDict[tuple[str, str, int, str], tuple[list[MemoryItem], int]] = OrderedDict()
        self._retrieval_cache_limit = 8192
        self._retrieval_cache_ttl_ms = max(1000, int(getattr(self.cfg.memory.l2, "retrieval_cache_ttl_ms", 15000)))
        self._known_l1_keys: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._known_l1_keys_limit = 500000
        self._known_l2_keys: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._known_l2_keys_limit = 500000
        self._agents_with_l1_history: set[str] = set()
        self._agents_with_l2_history: set[str] = set()
        self._close_join_timeout_s = 15.0

    def _emit_op(
        self,
        *,
        op_name: str,
        tier: str,
        latency_ms: float,
        hit: bool | None,
        n_bytes: int | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.op_logger is None:
            return
        row = {
            "ts_ms": now_ms(),
            "op_type": "memory",
            "op_name": op_name,
            "tier": tier,
            "hit": hit,
            "latency_ms": float(latency_ms),
            "bytes": n_bytes,
            "extra": extra or {},
        }
        self.op_logger(row)

    def _raise_l2_writer_error(self) -> None:
        if self._l2_writer_error is not None:
            raise self._l2_writer_error

    def _raise_l1_writer_error(self) -> None:
        if self._l1_writer_error is not None:
            raise self._l1_writer_error

    def _raise_background_errors(self) -> None:
        self._raise_l1_writer_error()
        self._raise_l2_writer_error()

    @staticmethod
    def _is_transient_l2_write_error(exc: Exception) -> bool:
        msg = str(exc or "").lower()
        markers = (
            "timeout",
            "timed out",
            "readtimeout",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "broken pipe",
        )
        return any(marker in msg for marker in markers)

    async def _persist_l1_batch(self, batch: list[MemoryItem], *, flush_interval_ms: float, queue_depth: int) -> None:
        if not batch or self.l1 is None:
            return
        total_bytes = sum(item.bytes for item in batch)
        ttl_s = self.cfg.memory.l1.ttl_s if self.cfg.memory.l1.enabled else None
        t0 = time.perf_counter()
        await self.l1.put_many(batch, ttl_s=ttl_s)
        self._emit_op(
            op_name="put",
            tier="L1",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            hit=None,
            n_bytes=total_bytes,
            extra={
                "batch_size": len(batch),
                "batch_size_used": len(batch),
                "flush_interval_ms": float(flush_interval_ms),
                "queue_depth": int(queue_depth),
            },
        )

    async def _l1_writer_loop(self) -> None:
        assert self._l1_write_queue is not None
        while True:
            try:
                item = await self._l1_write_queue.get()
            except asyncio.CancelledError:
                break
            batch = [item]
            t_batch_start = time.perf_counter()
            cancelled = False
            while len(batch) < self._l1_batch_size:
                elapsed = (time.perf_counter() - t_batch_start) * 1000.0
                remaining_ms = self._l1_flush_interval_ms - elapsed
                if remaining_ms <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._l1_write_queue.get(), timeout=remaining_ms / 1000.0)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    break
                batch.append(nxt)
            flush_interval_ms = (time.perf_counter() - t_batch_start) * 1000.0
            queue_depth = self._l1_write_queue.qsize()
            try:
                await self._persist_l1_batch(batch, flush_interval_ms=flush_interval_ms, queue_depth=queue_depth)
            except Exception as exc:
                self._l1_writer_error = RuntimeError(f"L1 background writer failed: {type(exc).__name__}")
                for _ in batch:
                    self._l1_write_queue.task_done()
                while True:
                    try:
                        _ = self._l1_write_queue.get_nowait()
                        self._l1_write_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                break
            for _ in batch:
                self._l1_write_queue.task_done()
            if cancelled:
                break

    async def _persist_l2_batch(self, batch: list[MemoryItem], *, flush_interval_ms: float, queue_depth: int) -> None:
        if not batch:
            return
        assert self.l2 is not None

        total_bytes = sum(item.bytes for item in batch)
        t0 = time.perf_counter()
        await self.l2.put_many(batch)
        self._emit_op(
            op_name="put",
            tier="L2",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            hit=None,
            n_bytes=total_bytes,
            extra={
                "batch_size": len(batch),
                "batch_size_used": len(batch),
                "flush_interval_ms": float(flush_interval_ms),
                "queue_depth": int(queue_depth),
                "dropped_writes_count": int(self._dropped_writes_count),
            },
        )

        if self.vector is None:
            self._emit_op(
                op_name="vector_upsert_skip",
                tier="L2",
                latency_ms=0.0,
                hit=False,
                n_bytes=total_bytes,
                extra={"reason": "vector_unavailable"},
            )
            return

        points = [
            {
                "id": _point_id(item.agent_id, item.key),
                "vector": item.embedding or hash_embedding(item.text),
                "payload": {"agent_id": item.agent_id, "key": item.key, "kind": item.kind},
            }
            for item in batch
        ]
        t1 = time.perf_counter()
        await self.vector.upsert_many(points)
        self._emit_op(
            op_name="vector_upsert",
            tier="L2",
            latency_ms=(time.perf_counter() - t1) * 1000.0,
            hit=None,
            n_bytes=total_bytes,
            extra={
                "batch_size": len(batch),
                "batch_size_used": len(batch),
                "flush_interval_ms": float(flush_interval_ms),
                "queue_depth": int(queue_depth),
                "dropped_writes_count": int(self._dropped_writes_count),
            },
        )

    def _query_embedding(self, agent_id: str, query: str) -> tuple[list[float], bool]:
        key = (agent_id, query.strip().lower())
        cached = self._query_embedding_cache.get(key)
        now = now_ms()
        if cached is not None:
            emb, expires_ms = cached
            if expires_ms >= now:
                self._query_embedding_cache.move_to_end(key)
                return emb, True
            self._query_embedding_cache.pop(key, None)
        emb = hash_embedding(query)
        self._query_embedding_cache[key] = (emb, now + self._embedding_cache_ttl_ms)
        if len(self._query_embedding_cache) > self._query_embedding_cache_limit:
            self._query_embedding_cache.popitem(last=False)
        return emb, False

    def _get_retrieval_cache(self, agent_id: str, query: str, k: int, mode: str) -> list[MemoryItem] | None:
        key = (agent_id, query.strip().lower(), int(k), str(mode))
        cached = self._retrieval_cache.get(key)
        now = now_ms()
        if cached is None:
            return None
        items, expires_ms = cached
        if expires_ms < now:
            self._retrieval_cache.pop(key, None)
            return None
        self._retrieval_cache.move_to_end(key)
        return [copy.deepcopy(item) for item in items]

    def _set_retrieval_cache(self, agent_id: str, query: str, k: int, mode: str, items: list[MemoryItem]) -> None:
        key = (agent_id, query.strip().lower(), int(k), str(mode))
        self._retrieval_cache[key] = ([copy.deepcopy(item) for item in items], now_ms() + self._retrieval_cache_ttl_ms)
        self._retrieval_cache.move_to_end(key)
        if len(self._retrieval_cache) > self._retrieval_cache_limit:
            self._retrieval_cache.popitem(last=False)

    def _has_memory_history(self, agent_id: str) -> bool:
        if self._recent_messages.get(agent_id):
            return True
        if agent_id in self._agents_with_l1_history:
            return True
        if agent_id in self._agents_with_l2_history:
            return True
        return False

    def _should_retrieve(self, agent_id: str, query: str) -> tuple[bool, str | None]:
        direct_key = self._extract_direct_key(query)
        if direct_key:
            return True, None
        q = query.strip().lower()
        if not q:
            return False, "empty_query"
        if any(marker in q for marker in RECALL_MARKERS):
            return True, None
        has_history = self._has_memory_history(agent_id)
        token_count = len(q.split())
        if not has_history and token_count <= 8 and len(q) <= 80:
            return False, "short_query_without_history"
        if not has_history:
            return False, "no_memory_history"
        return True, None

    @staticmethod
    def _has_direct_tool_evidence(recent: list[MemoryItem], query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        if not any(marker in q for marker in DIRECT_EVIDENCE_QUERY_MARKERS):
            return False
        # Use a wider tail window because retrieval shards may occupy the latest slots.
        for item in reversed(recent[-256:]):
            metadata = item.metadata or {}
            if str(metadata.get("role", "")).lower() != "tool":
                continue
            if bool(metadata.get("retrieval_shard", False)):
                continue
            # The main evidence body is usually much longer than hint snippets.
            if len(str(item.text or "")) < 80:
                continue
            return True
        return False

    @staticmethod
    def _finalize_results(items: list[MemoryItem], query: str, query_embedding: list[float], k: int) -> list[MemoryItem]:
        scored = sorted(items, key=lambda x: MultiLevelMemoryManager._relevance_score(x, query, query_embedding), reverse=True)
        for item in scored:
            item.metadata["retrieval_score"] = MultiLevelMemoryManager._relevance_score(item, query, query_embedding)
        return scored[:k]

    @staticmethod
    def _extract_direct_key(query: str) -> str | None:
        m = DIRECT_KEY_RE.match(query.strip())
        if not m:
            return None
        return m.group(1).lower()

    def _mark_known_l2_key(self, agent_id: str, key: str) -> None:
        token = (agent_id, key)
        self._known_l2_keys[token] = None
        self._known_l2_keys.move_to_end(token)
        self._agents_with_l2_history.add(agent_id)
        if len(self._known_l2_keys) > self._known_l2_keys_limit:
            removed = self._known_l2_keys.popitem(last=False)[0]
            removed_agent = removed[0]
            if not any(t[0] == removed_agent for t in self._known_l2_keys.keys()):
                self._agents_with_l2_history.discard(removed_agent)

    def _mark_known_l1_key(self, agent_id: str, key: str) -> None:
        token = (agent_id, key)
        self._known_l1_keys[token] = None
        self._known_l1_keys.move_to_end(token)
        self._agents_with_l1_history.add(agent_id)
        if len(self._known_l1_keys) > self._known_l1_keys_limit:
            removed = self._known_l1_keys.popitem(last=False)[0]
            removed_agent = removed[0]
            if not any(t[0] == removed_agent for t in self._known_l1_keys.keys()):
                self._agents_with_l1_history.discard(removed_agent)

    async def _l2_writer_loop(self) -> None:
        assert self._l2_write_queue is not None
        while True:
            try:
                item = await self._l2_write_queue.get()
            except asyncio.CancelledError:
                break
            batch = [item]
            t_batch_start = time.perf_counter()
            cancelled = False
            target_batch = min(self._l2_batch_size, self._l2_max_batch_per_flush)
            while len(batch) < target_batch:
                elapsed = (time.perf_counter() - t_batch_start) * 1000.0
                remaining_budget = min(self._l2_flush_interval_ms, self._l2_max_flush_ms)
                remaining_ms = remaining_budget - elapsed
                if remaining_ms <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._l2_write_queue.get(), timeout=remaining_ms / 1000.0)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    break
                batch.append(nxt)
            flush_interval_ms = (time.perf_counter() - t_batch_start) * 1000.0
            queue_depth = self._l2_write_queue.qsize()
            try:
                last_exc: Exception | None = None
                persisted = False
                for attempt in range(1, self._l2_write_retry_count + 1):
                    try:
                        await self._persist_l2_batch(batch, flush_interval_ms=flush_interval_ms, queue_depth=queue_depth)
                        persisted = True
                        break
                    except Exception as exc:
                        last_exc = exc
                        if attempt >= self._l2_write_retry_count or not self._is_transient_l2_write_error(exc):
                            raise
                        await asyncio.sleep((self._l2_write_retry_backoff_ms / 1000.0) * attempt)
                        try:
                            if self.l2 is not None:
                                await self.l2.connect()
                        except Exception:
                            pass
                if not persisted and last_exc is not None:
                    raise last_exc
            except Exception as exc:
                self._l2_writer_error = RuntimeError(f"L2 background writer failed: {type(exc).__name__}")
                for _ in batch:
                    self._l2_write_queue.task_done()
                while True:
                    try:
                        _ = self._l2_write_queue.get_nowait()
                        self._l2_write_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                break
            for _ in batch:
                self._l2_write_queue.task_done()
            if cancelled:
                break

    async def init(self) -> None:
        if self._l1_write_queue is not None and self._l1_writer_task is None:
            self._l1_writer_task = asyncio.create_task(self._l1_writer_loop())
        if self.l2 is not None:
            t0 = time.perf_counter()
            await self.l2.connect()
            self._emit_op(op_name="connect", tier="L2", latency_ms=(time.perf_counter() - t0) * 1000.0, hit=None, n_bytes=None)
        if self.vector is not None:
            try:
                t0 = time.perf_counter()
                await self.vector.ensure_collection()
                self._emit_op(op_name="ensure_collection", tier="L2", latency_ms=(time.perf_counter() - t0) * 1000.0, hit=None, n_bytes=None)
            except Exception as exc:
                self._emit_op(
                    op_name="degrade",
                    tier="L2",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=None,
                    extra={"reason": "vector_unavailable", "detail": type(exc).__name__},
                )
                self.vector = None
        if self.l3 is not None:
            t0 = time.perf_counter()
            await self.l3.ensure_ready()
            self._emit_op(op_name="ensure_ready", tier="L3", latency_ms=(time.perf_counter() - t0) * 1000.0, hit=None, n_bytes=None)
        if self._l2_write_queue is not None and self._l2_writer_task is None:
            self._l2_writer_task = asyncio.create_task(self._l2_writer_loop())

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[MemoryItem]) -> int:
        drained = 0
        while True:
            try:
                _ = queue.get_nowait()
                queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        return drained

    async def close(self) -> None:
        close_errors: list[str] = []
        if self._l1_write_queue is not None:
            try:
                await asyncio.wait_for(self._l1_write_queue.join(), timeout=self._close_join_timeout_s)
            except asyncio.TimeoutError:
                drained = self._drain_queue(self._l1_write_queue)
                close_errors.append(
                    f"L1 write queue drain timed out after {self._close_join_timeout_s:.1f}s; drained={drained}"
                )
        if self._l1_writer_task is not None:
            self._l1_writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._l1_writer_task
            self._l1_writer_task = None
        if self._l2_write_queue is not None:
            try:
                await asyncio.wait_for(self._l2_write_queue.join(), timeout=self._close_join_timeout_s)
            except asyncio.TimeoutError:
                drained = self._drain_queue(self._l2_write_queue)
                close_errors.append(
                    f"L2 write queue drain timed out after {self._close_join_timeout_s:.1f}s; drained={drained}"
                )
        if self._l2_writer_task is not None:
            self._l2_writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._l2_writer_task
            self._l2_writer_task = None
        if self.l1 is not None:
            await self.l1.close()
        if self.l2 is not None:
            await self.l2.close()
        if self.vector is not None:
            await self.vector.close()
        if self._l1_writer_error is not None:
            close_errors.append(str(self._l1_writer_error))
        if self._l2_writer_error is not None:
            close_errors.append(str(self._l2_writer_error))
        if close_errors:
            raise RuntimeError("; ".join(close_errors))

    async def put(self, item: MemoryItem, *, ttl_s: int | None = None) -> PlacementDecision:
        self._raise_background_errors()
        tiers: list[str] = []
        reasons: list[str] = []
        degrade_mode: str | None = None
        admission_reason = "tiered_ingest"

        if item.embedding is None:
            item.embedding = hash_embedding(item.text)

        if self.l3 is not None and int(item.bytes) >= int(self.cfg.memory.l3.pointer_threshold_bytes):
            source_kind = item.kind
            try:
                t_l3 = time.perf_counter()
                pointer = await self.l3.put(
                    agent_id=item.agent_id,
                    key=item.key,
                    text=item.text,
                    metadata={"kind": source_kind, "created_at_ms": int(item.created_at_ms)},
                )
                self._emit_op(
                    op_name="put",
                    tier="L3",
                    latency_ms=(time.perf_counter() - t_l3) * 1000.0,
                    hit=None,
                    n_bytes=int(pointer.get("bytes", item.bytes)),
                )
                preview_chars = max(80, int(self.cfg.memory.l3.pointer_preview_chars))
                preview = item.text[:preview_chars]
                marker = f"[BLOB_REF:{source_kind}:{item.key}] "
                pointer_meta = dict(item.metadata or {})
                pointer_meta["l3_pointer"] = pointer
                pointer_meta["pointerized"] = True
                pointer_meta["source_kind"] = source_kind
                item = MemoryItem(
                    key=item.key,
                    agent_id=item.agent_id,
                    kind="blob_ref",
                    text=marker + preview,
                    bytes=len((marker + preview).encode("utf-8")),
                    importance=item.importance,
                    created_at_ms=item.created_at_ms,
                    last_access_ms=item.last_access_ms,
                    access_count=item.access_count,
                    embedding=item.embedding,
                    metadata=pointer_meta,
                )
                tiers.append("L3")
                reasons.append("cold_large_object")
                admission_reason = "pointerized_to_l3"
            except Exception as exc:
                degrade_mode = "l3_unavailable"
                self._emit_op(
                    op_name="put_skip",
                    tier="L3",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=item.bytes,
                    extra={"reason": "l3_put_failed", "detail": type(exc).__name__},
                )
                reasons.append("l3_put_failed_fallback")

        if item.kind in {"message", "fact", "summary", "tool_result", "blob_ref"}:
            buf = self._recent_messages[item.agent_id]
            buf.append(item)
            if len(buf) > 512:
                del buf[:-512]
            tiers.append("L0")
            reasons.append("eligible_for_context")

        if self.cfg.memory.l1.enabled and self.l1 is not None and item.bytes <= self.cfg.memory.l1.max_object_bytes:
            if float(item.importance) >= float(self.cfg.memory.l1.reuse_threshold):
                if self._l1_write_queue is not None:
                    t0 = time.perf_counter()
                    await self._l1_write_queue.put(item)
                    self._emit_op(
                        op_name="enqueue_put",
                        tier="L1",
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        hit=None,
                        n_bytes=item.bytes,
                        extra={"queue_depth": self._l1_write_queue.qsize()},
                    )
                else:
                    t0 = time.perf_counter()
                    await self.l1.put(item, ttl_s=ttl_s or self.cfg.memory.l1.ttl_s)
                    self._emit_op(
                        op_name="put",
                        tier="L1",
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        hit=None,
                        n_bytes=item.bytes,
                    )
                self._mark_known_l1_key(item.agent_id, item.key)
                tiers.append("L1")
                reasons.append("hot_cache")
            else:
                self._emit_op(
                    op_name="skip_put",
                    tier="L1",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=item.bytes,
                    extra={
                        "importance": float(item.importance),
                        "reuse_threshold": float(self.cfg.memory.l1.reuse_threshold),
                    },
                )
                reasons.append("below_l1_reuse_threshold")

        if self.cfg.memory.l2.enabled and self.l2 is not None:
            if float(item.importance) >= self._l2_min_importance:
                if self._l2_write_queue is not None:
                    queue_depth = self._l2_write_queue.qsize()
                    if (
                        queue_depth >= self._l2_drop_queue_depth_threshold
                        and float(item.importance) < self._l2_backpressure_drop_importance_threshold
                    ):
                        self._dropped_writes_count += 1
                        self._emit_op(
                            op_name="drop_put",
                            tier="L2",
                            latency_ms=0.0,
                            hit=False,
                            n_bytes=item.bytes,
                            extra={
                                "queue_depth": int(queue_depth),
                                "drop_queue_depth_threshold": int(self._l2_drop_queue_depth_threshold),
                                "importance": float(item.importance),
                                "drop_importance_threshold": float(self._l2_backpressure_drop_importance_threshold),
                                "dropped_writes_count": int(self._dropped_writes_count),
                            },
                        )
                        reasons.append("l2_backpressure_drop")
                        return PlacementDecision(
                            key=item.key,
                            tiers=tiers,
                            ttl_s=ttl_s,
                            reasons=reasons,
                            admission_reason=admission_reason,
                            degrade_mode=degrade_mode,
                        )
                    t2 = time.perf_counter()
                    await self._l2_write_queue.put(item)
                    self._mark_known_l2_key(item.agent_id, item.key)
                    self._emit_op(
                        op_name="enqueue_put",
                        tier="L2",
                        latency_ms=(time.perf_counter() - t2) * 1000.0,
                        hit=None,
                        n_bytes=item.bytes,
                        extra={
                            "queue_depth": self._l2_write_queue.qsize(),
                            "dropped_writes_count": int(self._dropped_writes_count),
                        },
                    )
                else:
                    await self._persist_l2_batch([item], flush_interval_ms=0.0, queue_depth=0)
                    self._mark_known_l2_key(item.agent_id, item.key)
                tiers.append("L2")
                reasons.append("durable_with_vector_index" if self.vector is not None else "durable_without_vector")
            else:
                self._emit_op(
                    op_name="skip_put",
                    tier="L2",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=item.bytes,
                    extra={
                        "importance": float(item.importance),
                        "min_importance": self._l2_min_importance,
                    },
                )
                reasons.append("below_l2_importance_threshold")

        return PlacementDecision(
            key=item.key,
            tiers=tiers,
            ttl_s=ttl_s,
            reasons=reasons,
            admission_reason=admission_reason,
            degrade_mode=degrade_mode,
        )

    async def get(
        self,
        agent_id: str,
        query: str,
        *,
        k: int,
        mode: str = "hybrid",
        telemetry: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self._raise_background_errors()
        if telemetry is not None:
            telemetry.setdefault("embedding_cache_hit", None)
            telemetry.setdefault("retrieval_cache_hit", None)
            telemetry.setdefault("degrade_mode", "none")
            telemetry.setdefault("pointer_fetch", 0)
        out: list[MemoryItem] = []
        seen: set[str] = set()
        degrade_modes: list[str] = []
        direct_key = self._extract_direct_key(query)
        if direct_key:
            recent = self._recent_messages.get(agent_id, [])
            for item in reversed(recent):
                if item.key == direct_key:
                    item.metadata["retrieval_score"] = 1.0
                    self._emit_op(
                        op_name="get_key_recent",
                        tier="L0",
                        latency_ms=0.0,
                        hit=True,
                        n_bytes=item.bytes,
                    )
                    result = await self._resolve_blob_refs([item], telemetry=telemetry, degrade_modes=degrade_modes)
                    return result
        if telemetry is not None:
            cached = self._get_retrieval_cache(agent_id, query, k, mode)
            if cached is not None:
                telemetry["retrieval_cache_hit"] = True
                return cached
            telemetry["retrieval_cache_hit"] = False
        t_embed = time.perf_counter()
        query_embedding, cache_hit = self._query_embedding(agent_id, query)
        if telemetry is not None:
            telemetry["embedding_cache_hit"] = cache_hit
        if self.cfg.memory.l2.enabled:
            self._emit_op(
                op_name="embed_query",
                tier="L2",
                latency_ms=(time.perf_counter() - t_embed) * 1000.0,
                hit=cache_hit,
                n_bytes=None,
            )
        best_l1_score = 0.0

        if self.cfg.memory.l1.enabled and self.l1 is not None and mode in {"hybrid", "l1_first"}:
            try:
                if direct_key:
                    if (agent_id, direct_key) in self._known_l1_keys:
                        t_key = time.perf_counter()
                        exact = await self.l1.get_key(agent_id, direct_key)
                        key_lat = (time.perf_counter() - t_key) * 1000.0
                        self._emit_op(op_name="get_key", tier="L1", latency_ms=key_lat, hit=exact is not None, n_bytes=None)
                        if exact is not None:
                            exact.metadata["retrieval_score"] = 1.0
                            result = [exact]
                            result = await self._resolve_blob_refs(result, telemetry=telemetry, degrade_modes=degrade_modes)
                            self._set_retrieval_cache(agent_id, query, k, mode, result)
                            return result
                    else:
                        self._emit_op(
                            op_name="get_key_skip",
                            tier="L1",
                            latency_ms=0.0,
                            hit=False,
                            n_bytes=None,
                            extra={"reason": "unknown_direct_key"},
                        )
                if not direct_key:
                    t0 = time.perf_counter()
                    l1_recent = await self.l1.get_recent(agent_id, limit=min(64, max(8, k * 2)))
                    lat = (time.perf_counter() - t0) * 1000.0
                    for item in l1_recent:
                        if item.key in seen:
                            continue
                        score = self._relevance_score(item, query, query_embedding)
                        if score > best_l1_score:
                            best_l1_score = score
                        if score > 0.05:
                            item.metadata["retrieval_score"] = score
                            out.append(item)
                            seen.add(item.key)
                        if len(out) >= k:
                            self._emit_op(op_name="get", tier="L1", latency_ms=lat, hit=True, n_bytes=None)
                            result = out[:k]
                            result = await self._resolve_blob_refs(result, telemetry=telemetry, degrade_modes=degrade_modes)
                            self._set_retrieval_cache(agent_id, query, k, mode, result)
                            return result
                    self._emit_op(op_name="get", tier="L1", latency_ms=lat, hit=bool(out), n_bytes=None)
            except Exception as exc:
                degrade_modes.append("l1_unavailable")
                self._emit_op(
                    op_name="degrade",
                    tier="L1",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=None,
                    extra={"reason": type(exc).__name__},
                )

        if self.cfg.memory.l2.enabled and self.l2 is not None and mode in {"hybrid", "l2_only", "l1_first"}:
            try:
                if direct_key:
                    self._emit_op(
                        op_name="vector_search_skip",
                        tier="L2",
                        latency_ms=0.0,
                        hit=True,
                        n_bytes=None,
                        extra={"reason": "direct_key_query"},
                    )
                    if (agent_id, direct_key) in self._known_l2_keys:
                        t1 = time.perf_counter()
                        pg_items = await self.l2.get_by_keys(agent_id, [direct_key])
                        self._emit_op(
                            op_name="get_by_keys",
                            tier="L2",
                            latency_ms=(time.perf_counter() - t1) * 1000.0,
                            hit=bool(pg_items),
                            n_bytes=None,
                            extra={"key_count": 1, "direct_key": True},
                        )
                        out.extend(pg_items)
                    else:
                        self._emit_op(
                            op_name="get_by_keys_skip",
                            tier="L2",
                            latency_ms=0.0,
                            hit=False,
                            n_bytes=None,
                            extra={"reason": "unknown_direct_key"},
                        )
                elif self.vector is None:
                    degrade_modes.append("vector_unavailable")
                    self._emit_op(
                        op_name="vector_search_skip",
                        tier="L2",
                        latency_ms=0.0,
                        hit=False,
                        n_bytes=None,
                        extra={"reason": "vector_unavailable"},
                    )
                elif best_l1_score >= 0.75 and out:
                    self._emit_op(
                        op_name="vector_search_skip",
                        tier="L2",
                        latency_ms=0.0,
                        hit=True,
                        n_bytes=None,
                        extra={"reason": "high_confidence_l1_hit"},
                    )
                    result = self._finalize_results(out, query, query_embedding, k)
                    result = await self._resolve_blob_refs(result, telemetry=telemetry, degrade_modes=degrade_modes)
                    self._set_retrieval_cache(agent_id, query, k, mode, result)
                    return result
                else:
                    t0 = time.perf_counter()
                    q_hits = await self.vector.search(query_embedding, limit=max(k * 2, 16), agent_id=agent_id)  # type: ignore[union-attr]
                    qdrant_ms = (time.perf_counter() - t0) * 1000.0
                    self._emit_op(
                        op_name="vector_search",
                        tier="L2",
                        latency_ms=qdrant_ms,
                        hit=bool(q_hits),
                        n_bytes=None,
                        extra={"qdrant_search_time_ms": qdrant_ms},
                    )
                    q_keys = []
                    for hit in q_hits:
                        payload = hit.get("payload", {}) or {}
                        key = payload.get("key")
                        if key and key not in seen:
                            q_keys.append(key)
                            seen.add(key)
                    if q_keys:
                        deduped_keys = list(dict.fromkeys(q_keys))
                        t1 = time.perf_counter()
                        pg_items = await self.l2.get_by_keys(agent_id, deduped_keys)
                        self._emit_op(
                            op_name="get_by_keys",
                            tier="L2",
                            latency_ms=(time.perf_counter() - t1) * 1000.0,
                            hit=bool(pg_items),
                            n_bytes=None,
                            extra={"key_count": len(deduped_keys)},
                        )
                        for item in pg_items:
                            item.metadata["retrieval_score"] = self._relevance_score(item, query, query_embedding)
                        out.extend(pg_items)
                # For direct-key lookups, avoid expensive broad text scans.
                if direct_key:
                    result = self._finalize_results(out, query, query_embedding, k)
                    result = await self._resolve_blob_refs(result, telemetry=telemetry, degrade_modes=degrade_modes)
                    self._set_retrieval_cache(agent_id, query, k, mode, result)
                    return result
                if len(out) < k:
                    t2 = time.perf_counter()
                    extra = await self.l2.search_text(agent_id, query=query, limit=(k - len(out)))
                    self._emit_op(op_name="search_text", tier="L2", latency_ms=(time.perf_counter() - t2) * 1000.0, hit=bool(extra), n_bytes=None)
                    for item in extra:
                        if item.key in seen:
                            continue
                        item.metadata["retrieval_score"] = self._relevance_score(item, query, query_embedding)
                        out.append(item)
                        seen.add(item.key)
            except Exception as exc:
                degrade_modes.append("l2_unavailable")
                self._emit_op(
                    op_name="degrade",
                    tier="L2",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=None,
                    extra={"reason": type(exc).__name__},
                )

        result = self._finalize_results(out, query, query_embedding, k)
        result = await self._resolve_blob_refs(result, telemetry=telemetry, degrade_modes=degrade_modes)
        if telemetry is not None:
            telemetry["degrade_mode"] = "|".join(sorted(set(degrade_modes))) if degrade_modes else "none"
        self._set_retrieval_cache(agent_id, query, k, mode, result)
        return result

    async def assemble_context(self, agent_id: str, user_query: str, budget: ContextBudget) -> ContextPack:
        self._raise_background_errors()
        recent = list(self._recent_messages.get(agent_id, []))
        if not recent and self.cfg.memory.l2.enabled and self.l2 is not None:
            t0 = time.perf_counter()
            recent = await self.l2.get_recent(agent_id, limit=max(2 * budget.keep_recent_messages, 20))
            recent = list(reversed(recent))
            self._emit_op(op_name="get_recent", tier="L2", latency_ms=(time.perf_counter() - t0) * 1000.0, hit=bool(recent), n_bytes=None)

        retrieval_k = 8
        if self.cfg.memory.l2.enabled:
            retrieval_k = max(1, int(self.cfg.memory.l2.default_top_k))
        retrieve_allowed, retrieve_skip_reason = self._should_retrieve(agent_id, user_query)
        if retrieve_allowed and bool(getattr(self.cfg.memory.policy, "direct_evidence_retrieval_bypass", True)):
            if self._has_direct_tool_evidence(recent, user_query):
                retrieve_allowed = False
                retrieve_skip_reason = "direct_tool_evidence_present"
        retrieval_telemetry: dict[str, Any] = {}
        if retrieve_allowed:
            candidates = await self.get(agent_id, user_query, k=retrieval_k, mode="hybrid", telemetry=retrieval_telemetry)
        else:
            candidates = []
            self._emit_op(
                op_name="retrieve_skip",
                tier="L2",
                latency_ms=0.0,
                hit=False,
                n_bytes=None,
                extra={"reason": retrieve_skip_reason or "heuristic_skip"},
            )
        inject_budget = min(
            int(self.cfg.memory.l0.max_memory_injection_tokens),
            int(self.cfg.memory.l0.max_injected_tokens),
        )
        packed = assemble_l0_context(
            recent_messages=recent,
            candidates=candidates,
            user_query=user_query,
            budget=ContextBudget(
                max_prompt_tokens=budget.max_prompt_tokens,
                max_memory_injection_tokens=max(0, inject_budget),
                keep_recent_messages=budget.keep_recent_messages,
            ),
            summarization_on=self.cfg.experiment.summarization_on,
            summarize_trigger_tokens=self.cfg.memory.l0.summarize_trigger_tokens,
            summarize_target_tokens=self.cfg.memory.l0.summarize_target_tokens,
            min_relevance_score=float(self.cfg.memory.l0.min_relevance_score),
        )
        self._emit_op(
            op_name="assemble_context",
            tier="L0",
            latency_ms=0.0,
            hit=True,
            n_bytes=None,
            extra={
                "injected_tokens": packed.injected_tokens,
                "injected_items": packed.injected_item_count,
                "total_tokens": packed.total_tokens,
                "summarized": packed.summarized,
            },
        )
        return ContextPack(
            messages=packed.messages,
            injected_memories=packed.selected,
            trace={
                "selected_keys": [i.key for i in packed.selected],
                "dropped_keys": packed.dropped_keys,
                "summarized": packed.summarized,
                "total_tokens": packed.total_tokens,
                "packed_tokens": packed.total_tokens,
                "injection_tokens": packed.injected_tokens,
                "injection_item_count": packed.injected_item_count,
                "admission_reason": "retrieval_then_pack",
                "eviction_reason": "budget_trim" if packed.dropped_keys else "",
                "pointer_fetch": int(retrieval_telemetry.get("pointer_fetch", 0) or 0),
                "degrade_mode": retrieval_telemetry.get("degrade_mode", "none"),
                "retrieve_skipped_reason": retrieve_skip_reason,
                "embedding_cache_hit": retrieval_telemetry.get("embedding_cache_hit"),
                "retrieval_cache_hit": retrieval_telemetry.get("retrieval_cache_hit"),
            },
        )

    async def checkpoint(self, agent_id: str, thread_id: str, state: dict) -> None:
        self._raise_background_errors()
        if str(getattr(self.cfg.memory.policy, "checkpoint_policy", "l2_only")) == "disabled":
            self._emit_op(
                op_name="checkpoint_skip",
                tier="L2",
                latency_ms=0.0,
                hit=False,
                n_bytes=None,
                extra={"reason": "policy_disabled"},
            )
            return
        if self.l2 is None:
            return
        t0 = time.perf_counter()
        await self.l2.checkpoint(agent_id, thread_id, now_ms(), state)
        self._emit_op(op_name="checkpoint", tier="L2", latency_ms=(time.perf_counter() - t0) * 1000.0, hit=None, n_bytes=None)

    async def healthcheck(self) -> dict[str, Any]:
        return {
            "redis": await self.l1.ping() if self.l1 is not None else True,
            "postgres": await self.l2.ping() if self.l2 is not None else True,
            "qdrant": await self.vector.ping() if self.vector is not None else True,
            "l3_object_store": await self.l3.ping() if self.l3 is not None else True,
            "postgres_backend": "postgres" if self.l2 is not None else "none",
            "qdrant_mode": "remote" if self.vector is not None else "none",
            "l3_mode": "local_object_store" if self.l3 is not None else "none",
        }

    async def _resolve_blob_refs(
        self,
        items: list[MemoryItem],
        *,
        telemetry: dict[str, Any] | None,
        degrade_modes: list[str],
    ) -> list[MemoryItem]:
        if not items or self.l3 is None:
            return items
        pointer_fetch = int(telemetry.get("pointer_fetch", 0) or 0) if telemetry is not None else 0
        for item in items:
            pointer = (item.metadata or {}).get("l3_pointer")
            if item.kind != "blob_ref" or not isinstance(pointer, dict):
                continue
            try:
                t0 = time.perf_counter()
                blob_text = await self.l3.get(pointer)
                lat = (time.perf_counter() - t0) * 1000.0
                self._emit_op(op_name="get", tier="L3", latency_ms=lat, hit=bool(blob_text), n_bytes=None)
                if not blob_text:
                    continue
                pointer_fetch += 1
                preview_chars = max(80, int(self.cfg.memory.l3.pointer_preview_chars))
                preview = blob_text[:preview_chars]
                item.metadata["pointer_preview"] = preview
                item.text = f"{item.text}\n[POINTER_PREVIEW] {preview}"
                item.bytes = len(item.text.encode("utf-8"))
            except Exception as exc:
                degrade_modes.append("l3_unavailable")
                self._emit_op(
                    op_name="degrade",
                    tier="L3",
                    latency_ms=0.0,
                    hit=False,
                    n_bytes=None,
                    extra={"reason": type(exc).__name__},
                )
        if telemetry is not None:
            telemetry["pointer_fetch"] = pointer_fetch
        return items

    @staticmethod
    def _relevance_score(item: MemoryItem, query: str, query_embedding: list[float]) -> float:
        emb = item.embedding or hash_embedding(item.text)
        sem = max(0.0, cosine(emb, query_embedding))
        lex = 1.0 if any(tok in item.text.lower() for tok in query.lower().split()) else 0.0
        return 0.7 * sem + 0.3 * lex

    @staticmethod
    def to_dict(item: MemoryItem) -> dict[str, Any]:
        return asdict(item)
