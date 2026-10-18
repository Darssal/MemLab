"""Interleave two worker sessions over one fake memory manager."""
import asyncio
from types import SimpleNamespace

from memlab.config import MemlabConfig
from memlab.runner_stub import _CURRENT_AGENT_ID, _make_op_logger, _session_worker


def test_shared_memory_logger_keeps_interleaved_session_ownership():
    async def run():
        rows, task_rows, written = [], [], []
        writer = SimpleNamespace(append=written.append)
        both_started = asyncio.Event()

        class Memory:
            def __init__(self):
                self.started = 0
                self.op_logger = _make_op_logger(run_id="demo", mem_rows=rows, event_writer=writer)

            async def put(self, item, **_):
                self.started += 1
                if self.started == 2:
                    both_started.set()
                await both_started.wait()
                await asyncio.sleep(0)
                self.op_logger({"op_type": "memory", "expected_agent": item.agent_id})

            async def get(self, **_):
                return []

            async def assemble_context(self, **_):
                return SimpleNamespace(trace={})

        memory = Memory()
        queue = asyncio.Queue()
        for i in range(2):
            queue.put_nowait({"task_id": f"demo-{i}", "key": f"k-{i:06d}", "bytes": 64})
        cfg = MemlabConfig()
        await asyncio.gather(*[
            _session_worker(
                queue=queue, session_index=i, run_id="demo", memory=memory,
                event_writer=writer, task_rows=task_rows, mem_rows=rows, cfg=cfg,
            )
            for i in range(2)
        ])
        assert len(rows) == 2
        assert {row["agent_id"] for row in rows} == {"agent_0001", "agent_0002"}
        for row in rows:
            assert row["agent_id"] == row["expected_agent"]
            assert row["session_id"] == f'demo-{row["expected_agent"]}'
        assert all(row["status"] == "success" for row in task_rows)
        assert _CURRENT_AGENT_ID.get() is None

    asyncio.run(run())


def test_e2e_worker_context_keeps_interleaved_session_ownership(monkeypatch):
    from memlab import runner_e2e

    async def run():
        rows = []
        writer = SimpleNamespace(append=lambda _: None)
        logger = _make_op_logger(run_id="demo", mem_rows=rows, event_writer=writer)
        both_started = asyncio.Event()
        started = 0

        async def body(**kwargs):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            await asyncio.sleep(0)
            logger({"expected_agent": f"agent_{kwargs['session_index'] + 1:04d}"})

        monkeypatch.setattr(runner_e2e, "_session_worker_body", body)
        await asyncio.gather(*[
            runner_e2e._session_worker(session_index=i) for i in range(2)
        ])
        assert len(rows) == 2
        for row in rows:
            assert row["agent_id"] == row["expected_agent"]
            assert row["session_id"] == f'demo-{row["expected_agent"]}'
        assert _CURRENT_AGENT_ID.get() is None

    asyncio.run(run())
