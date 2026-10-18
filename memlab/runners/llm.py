from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from .common import extract_secret_code


SECRET_JSON_RE = re.compile(r'"secret_code"\s*:\s*"([A-Z0-9]+)"')


def _build_stub_json(task_id: str, secret: str) -> str:
    return json.dumps({"task_id": task_id, "secret_code": secret}, ensure_ascii=False)


async def run_stub_llm(task_id: str, messages: list[dict[str, Any]], evidence_text: str) -> dict[str, Any]:
    secret = extract_secret_code(evidence_text)
    content = _build_stub_json(task_id, secret)
    return {
        "content": content,
        "prompt_tokens": sum(max(1, len(m.get("content", "")) // 4) for m in messages),
        "completion_tokens": max(1, len(content) // 4),
    }


async def run_openai_compatible(task_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "")
    if not base_url or not api_key or not model:
        raise RuntimeError("OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL must be set for openai_compatible mode")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    except Exception as exc:
        raise RuntimeError(type(exc).__name__) from None
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}")
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {}) or {}
    return {
        "content": content,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def parse_model_secret(response_content: str) -> str:
    try:
        obj = json.loads(response_content)
        if isinstance(obj, dict):
            sc = obj.get("secret_code")
            return sc if isinstance(sc, str) else ""
    except Exception:
        pass
    m = SECRET_JSON_RE.search(response_content)
    return m.group(1) if m else ""
