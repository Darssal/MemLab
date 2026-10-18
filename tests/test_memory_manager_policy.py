from __future__ import annotations

import asyncio

from memlab.config import MemlabConfig
from memlab.memory.types import ContextBudget, MemoryItem
from memlab.memory_manager.manager import MultiLevelMemoryManager


def _mk_item(*, key: str, role: str, text: str, retrieval_shard: bool = False) -> MemoryItem:
    return MemoryItem(
        key=key,
        agent_id="agent_0001",
        kind="tool_result" if role == "tool" else "message",
        text=text,
        bytes=len(text.encode("utf-8")),
        importance=0.9,
        created_at_ms=1,
        last_access_ms=1,
        access_count=1,
        metadata={
            "role": role,
            "retrieval_shard": retrieval_shard,
        },
    )


def test_direct_tool_evidence_detector() -> None:
    recent = [
        _mk_item(
            key="tool-main",
            role="tool",
            text="SECRET_CODE: ABC123. " + ("evidence " * 32),
        )
    ]
    assert MultiLevelMemoryManager._has_direct_tool_evidence(
        recent, 'Find the SECRET_CODE in the document titled "Document 00013".'
    )


def test_direct_tool_evidence_ignores_shard_only() -> None:
    recent = [
        _mk_item(
            key="tool-shard",
            role="tool",
            text="SECRET_CODE: ABC123. " + ("evidence " * 32),
            retrieval_shard=True,
        )
    ]
    assert not MultiLevelMemoryManager._has_direct_tool_evidence(
        recent, 'Find the SECRET_CODE in the document titled "Document 00013".'
    )


def test_assemble_context_bypasses_retrieval_when_direct_evidence_present() -> None:
    cfg = MemlabConfig()
    cfg.memory.l1.enabled = False
    cfg.memory.l2.enabled = False
    cfg.memory.l3.enabled = False
    cfg.memory.policy.direct_evidence_retrieval_bypass = True

    manager = MultiLevelMemoryManager(cfg)
    manager._recent_messages["agent_0001"] = [
        _mk_item(key="user-1", role="user", text="Task t00013: Find the secret code."),
        _mk_item(
            key="tool-1",
            role="tool",
            text="Document body with SECRET_CODE: ABC123 and extra evidence context. " + ("x" * 180),
        ),
    ]

    pack = asyncio.run(
        manager.assemble_context(
            "agent_0001",
            'Task t00013: Find the SECRET_CODE in the document titled "Document 00013".',
            ContextBudget(max_prompt_tokens=4000, max_memory_injection_tokens=256, keep_recent_messages=10),
        )
    )

    assert pack.trace.get("retrieve_skipped_reason") == "direct_tool_evidence_present"
