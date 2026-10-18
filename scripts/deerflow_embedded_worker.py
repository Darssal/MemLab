from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEERFLOW_ROOT = Path(os.getenv("MEMLAB_DEERFLOW_ROOT", "") or "deer-flow").expanduser().resolve()
BACKEND_ROOT = DEERFLOW_ROOT / "backend"
CONFIG_PATH = DEERFLOW_ROOT / "config.yaml"
DEERFLOW_HOME = BACKEND_ROOT / ".deer-flow"

os.environ.setdefault("DEER_FLOW_CONFIG_PATH", str(CONFIG_PATH))
os.environ.setdefault("DEER_FLOW_HOME", str(DEERFLOW_HOME))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agents.lead_agent.agent import make_lead_agent
from src.config.app_config import reset_app_config
import src.config.paths as paths_mod


_AGENT_CACHE: dict[tuple[str, str], Any] = {}
JSON_PREFIX = "__MEMLAB_JSON__:"


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(JSON_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _agent_key(model_override: str | None, agent_name: str | None) -> tuple[str, str]:
    return (str(model_override or "").strip(), str(agent_name or "").strip())


def _build_config(*, model_override: str | None, agent_name: str | None, thread_id: str) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "thinking_enabled": False,
        "is_plan_mode": False,
        "subagent_enabled": False,
    }
    if model_override:
        configurable["model_name"] = model_override
        configurable["model"] = model_override
    if agent_name:
        configurable["agent_name"] = agent_name
    return RunnableConfig(configurable=configurable, recursion_limit=100)


def _get_agent(*, model_override: str | None, agent_name: str | None):
    key = _agent_key(model_override, agent_name)
    cached = _AGENT_CACHE.get(key)
    if cached is not None:
        return cached
    config = _build_config(model_override=model_override, agent_name=agent_name, thread_id="worker-bootstrap")
    agent = make_lead_agent(config)
    _AGENT_CACHE[key] = agent
    return agent


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


def _run_prompt(*, thread_id: str, prompt: str, model_override: str | None, agent_name: str | None) -> dict[str, Any]:
    agent = _get_agent(model_override=model_override, agent_name=agent_name)
    config = _build_config(model_override=model_override, agent_name=agent_name, thread_id=thread_id)
    state: dict[str, Any] = {"messages": [HumanMessage(content=prompt)]}
    context = {"thread_id": thread_id}
    seen_ids: set[str] = set()
    ai_chunks: list[str] = []
    started = time.perf_counter()
    for chunk in agent.stream(state, config=config, context=context, stream_mode="values"):
        messages = chunk.get("messages", [])
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            msg_id = getattr(msg, "id", None)
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)
            text = _extract_text(getattr(msg, "content", ""))
            if text:
                ai_chunks.append(text)
    return {
        "content": ai_chunks[-1] if ai_chunks else "{}",
        "request_latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> int:
    reset_app_config()
    if hasattr(paths_mod, "_paths"):
        paths_mod._paths = None
    _emit(
        {
            "ready": True,
            "pid": os.getpid(),
            "config_path": os.environ.get("DEER_FLOW_CONFIG_PATH", ""),
            "deerflow_home": os.environ.get("DEER_FLOW_HOME", ""),
        }
    )
    for raw_line in sys.stdin:
        line = str(raw_line or "").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "error": f"invalid_request_json: {exc}"})
            continue
        req_type = str(request.get("type", "run") or "run").strip().lower()
        if req_type == "shutdown":
            _emit({"ok": True, "shutdown": True})
            return 0
        try:
            result = _run_prompt(
                thread_id=str(request.get("thread_id", "") or "embedded-thread"),
                prompt=str(request.get("prompt", "") or ""),
                model_override=str(request.get("model_override", "") or "") or None,
                agent_name=str(request.get("agent_name", "") or "") or None,
            )
            _emit({"ok": True, **result})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
