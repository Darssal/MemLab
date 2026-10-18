from .types import ContextBudget, ContextPack, MemoryItem, PlacementDecision

__all__ = [
    "TieredMemoryManager",
    "ContextBudget",
    "ContextPack",
    "MemoryItem",
    "PlacementDecision",
]


def __getattr__(name: str):
    if name == "TieredMemoryManager":
        from .manager import TieredMemoryManager

        return TieredMemoryManager
    raise AttributeError(name)
