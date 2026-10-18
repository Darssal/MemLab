from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

from memlab.config import MemlabConfig
from memlab.memory.types import MemoryItem
from memlab.memory_manager.manager import MultiLevelMemoryManager
from memlab.runner_core import apply_variant, normalize_variant


def test_variant_alias_mapping() -> None:
    canonical, alias = normalize_variant("A1")
    assert canonical == "V1"
    assert alias == "A1"

    cfg = apply_variant(MemlabConfig(), "A2")
    assert cfg.experiment.variant_id == "F1"
    assert cfg.experiment.redis_policy == "allkeys-lru"


def test_pointerization_to_l3() -> None:
    async def _run() -> None:
        cfg = MemlabConfig()
        cfg.memory.l1.enabled = False
        cfg.memory.l2.enabled = False
        cfg.memory.l3.enabled = True
        cfg.memory.l3.pointer_threshold_bytes = 32
        with TemporaryDirectory() as td:
            cfg.memory.l3.root_dir = td
            mgr = MultiLevelMemoryManager(cfg)
            await mgr.init()
            try:
                item = MemoryItem(
                    key="k-000001",
                    agent_id="agent-1",
                    kind="fact",
                    text="x" * 256,
                    bytes=256,
                    importance=1.0,
                    created_at_ms=1,
                    last_access_ms=1,
                    access_count=1,
                    metadata={"role": "system"},
                )
                decision = await mgr.put(item)
                assert "L3" in decision.tiers
                assert decision.admission_reason == "pointerized_to_l3"
                recent = mgr._recent_messages["agent-1"][-1]
                assert recent.kind == "blob_ref"
                assert "l3_pointer" in recent.metadata
            finally:
                await mgr.close()

    asyncio.run(_run())


def test_degrade_mode_trace_when_l1_fails() -> None:
    class _FailingL1:
        async def get_recent(self, agent_id: str, limit: int):  # noqa: ARG002
            raise RuntimeError("injected_l1_failure")

        async def get_key(self, agent_id: str, key: str):  # noqa: ARG002
            raise RuntimeError("injected_l1_failure")

    async def _run() -> None:
        cfg = MemlabConfig()
        cfg.memory.l1.enabled = True
        cfg.memory.l2.enabled = False
        cfg.memory.l3.enabled = False
        mgr = MultiLevelMemoryManager(cfg)
        mgr.l1 = _FailingL1()  # type: ignore[assignment]
        telemetry: dict[str, object] = {}
        rows = await mgr.get(agent_id="agent-1", query="what happened before", k=4, mode="hybrid", telemetry=telemetry)
        assert rows == []
        assert telemetry.get("degrade_mode") == "l1_unavailable"

    asyncio.run(_run())
