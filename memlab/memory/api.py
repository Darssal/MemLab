from __future__ import annotations

from abc import ABC, abstractmethod

from .types import ContextBudget, ContextPack, MemoryItem, PlacementDecision


class MemoryManager(ABC):
    @abstractmethod
    async def put(self, item: MemoryItem, *, ttl_s: int | None = None) -> PlacementDecision:
        raise NotImplementedError

    @abstractmethod
    async def get(
        self,
        agent_id: str,
        query: str,
        *,
        k: int,
        mode: str = "hybrid",
    ) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    async def assemble_context(
        self,
        agent_id: str,
        user_query: str,
        budget: ContextBudget,
    ) -> ContextPack:
        raise NotImplementedError

    @abstractmethod
    async def checkpoint(self, agent_id: str, thread_id: str, state: dict) -> None:
        raise NotImplementedError
