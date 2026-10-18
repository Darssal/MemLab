from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalObjectStore:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)

    async def ensure_ready(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def ping(self) -> bool:
        try:
            await self.ensure_ready()
            return self.root.exists()
        except Exception:
            return False

    @staticmethod
    def _safe_chunk(value: str) -> str:
        return "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})

    def _blob_path(self, agent_id: str, key: str) -> Path:
        safe_agent = self._safe_chunk(agent_id) or "agent"
        safe_key = self._safe_chunk(key) or "key"
        return self.root / safe_agent / f"{safe_key}.json"

    async def put(self, *, agent_id: str, key: str, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        await self.ensure_ready()
        path = self._blob_path(agent_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"text": text, "metadata": metadata or {}}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return {
            "path": str(path.as_posix()),
            "bytes": len(text.encode("utf-8")),
        }

    async def get(self, pointer: dict[str, Any]) -> str | None:
        raw = str(pointer.get("path", "")).strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / raw
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        text = payload.get("text")
        return str(text) if isinstance(text, str) else None
