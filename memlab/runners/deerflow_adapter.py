from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass


@dataclass
class DeerFlowResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int


class DeerFlowAdapter:
    def __init__(self) -> None:
        backend_path = os.getenv("DEERFLOW_BACKEND_PATH", "").strip()
        if not backend_path:
            raise RuntimeError("DEERFLOW_BACKEND_PATH must point to DeerFlow backend directory")
        if backend_path not in sys.path:
            sys.path.append(backend_path)

        from src.client import DeerFlowClient  # type: ignore

        checkpointer = None
        if os.getenv("MEMLAB_ENABLE_CHECKPOINTER", "1") == "1":
            from memlab.checkpoint.langgraph_pg import get_checkpointer

            checkpointer = get_checkpointer()

        self.client = DeerFlowClient(checkpointer=checkpointer)

    async def chat(self, prompt: str, thread_id: str) -> DeerFlowResponse:
        def _call() -> str:
            resp = self.client.chat(prompt, thread_id=thread_id)
            return resp if isinstance(resp, str) else str(resp)

        content = await asyncio.to_thread(_call)
        # DeerFlow client does not always expose usage in plain chat return.
        return DeerFlowResponse(
            content=content,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(content) // 4),
        )
