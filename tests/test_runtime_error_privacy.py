"""Remote error payloads must not become persistent diagnostic strings."""
import asyncio

import httpx
import pytest

from memlab import runner_e2e
from memlab.runners import llm


SENTINEL = "SYNTHETIC_REMOTE_BODY_MUST_NOT_BE_PERSISTED"


class Client:
    status = 422

    def __init__(self, **_):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def post(self, *_, **__):
        return httpx.Response(self.status, text=SENTINEL)


@pytest.mark.parametrize("status", [403, 422])
def test_direct_provider_exception_excludes_error_body(status):
    client = Client()
    client.status = status
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(runner_e2e._run_provider_prompt_direct(
            client=client, base_url="https://example.test/v1", model_id="demo-model",
            api_key="placeholder", prompt="synthetic prompt", llm_sema=asyncio.Semaphore(1),
        ))
    assert SENTINEL not in str(caught.value)
    assert str(status) in str(caught.value)


@pytest.mark.parametrize("preflight", [
    runner_e2e._provider_chat_preflight, runner_e2e._provider_tool_call_preflight,
])
def test_preflight_detail_excludes_error_body(monkeypatch, preflight):
    monkeypatch.setattr(runner_e2e.httpx, "AsyncClient", Client)
    ok, detail = asyncio.run(preflight(
        base_url="https://example.test/v1", model_id="demo-model", api_key="placeholder",
    ))
    assert not ok
    assert detail == "HTTP 422"
    assert SENTINEL not in detail


def test_compatible_adapter_exception_excludes_remote_body(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")
    monkeypatch.setattr(llm.httpx, "AsyncClient", Client)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(llm.run_openai_compatible("demo", []))
    assert str(caught.value) == "HTTP 422"


def test_transport_exception_detail_excludes_payload(monkeypatch):
    class FailedClient(Client):
        async def post(self, *_, **__):
            raise httpx.ConnectError(SENTINEL)

    monkeypatch.setattr(runner_e2e.httpx, "AsyncClient", FailedClient)
    ok, detail = asyncio.run(runner_e2e._provider_chat_preflight(
        base_url="https://example.test/v1", model_id="demo-model", api_key="placeholder",
    ))
    assert not ok
    assert detail == "ConnectError"


def test_memory_failure_events_exclude_exception_payload():
    import json

    from memlab.config import MemlabConfig
    from memlab.memory_manager.manager import MultiLevelMemoryManager

    class FailedL1:
        async def get_recent(self, *_args, **_kwargs):
            raise RuntimeError(SENTINEL)

        async def get_key(self, *_args, **_kwargs):
            raise RuntimeError(SENTINEL)

    cfg = MemlabConfig()
    cfg.memory.l2.enabled = False
    cfg.memory.l3.enabled = False
    events = []
    manager = MultiLevelMemoryManager(cfg, op_logger=events.append)
    manager.l1 = FailedL1()
    asyncio.run(manager.get(agent_id="demo-agent", query="what happened before", k=4))
    assert events
    assert SENTINEL not in json.dumps(events)


@pytest.mark.parametrize("balance_error", [True, False])
def test_deerflow_stream_error_excludes_remote_payload(balance_error):
    import json

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield "event: error"
            reason = "insufficient balance" if balance_error else "request rejected"
            yield "data: " + json.dumps({"error": reason, "echo": SENTINEL})

    class StreamClient:
        def stream(self, *_, **__):
            return Stream()

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(runner_e2e._run_deerflow_prompt(
            client=StreamClient(), langgraph_url="https://example.test",
            thread_id="synthetic-thread", prompt="synthetic prompt",
            llm_sema=asyncio.Semaphore(1),
        ))
    expected = "provider_balance_rejected" if balance_error else "deerflow_stream_rejected"
    assert str(caught.value) == expected
    assert SENTINEL not in str(caught.value)
    assert SENTINEL not in runner_e2e._safe_error_detail(caught.value)
