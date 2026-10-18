from __future__ import annotations

from typing import Any

from memlab.backends.object_store import LocalObjectStore


class L3ObjectStore:
    def __init__(self, root_dir: str) -> None:
        self.store = LocalObjectStore(root_dir=root_dir)

    async def ensure_ready(self) -> None:
        await self.store.ensure_ready()

    async def ping(self) -> bool:
        return await self.store.ping()

    async def put(self, *, agent_id: str, key: str, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.store.put(agent_id=agent_id, key=key, text=text, metadata=metadata)

    async def get(self, pointer: dict[str, Any]) -> str | None:
        return await self.store.get(pointer)
