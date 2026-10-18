"""Attribute shared-manager operations to the current async session."""
from contextvars import ContextVar
from typing import Any

current_agent_id: ContextVar[str | None] = ContextVar("memory_agent_id", default=None)


def make_operation_logger(*, run_id, mem_rows, event_writer):
    def log_operation(row: dict[str, Any]) -> None:
        event = dict(row)
        agent_id = event.get("agent_id") or current_agent_id.get()
        event["run_id"] = run_id
        event["agent_id"] = agent_id
        event["session_id"] = f"{run_id}-{agent_id}" if agent_id else None
        mem_rows.append(event)
        event_writer.append(event)

    return log_operation
