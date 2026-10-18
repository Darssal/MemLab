from __future__ import annotations

import os


def get_checkpointer():
    """
    Optional helper for DeerFlow/LangGraph multi-turn persistence.
    Returns a configured PostgresSaver instance when dependency exists.
    """
    pg_uri = os.environ.get("PG_URI")
    if not pg_uri:
        raise RuntimeError("PG_URI is required for checkpointer")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except Exception as exc:
        raise RuntimeError(
            "langgraph-checkpoint-postgres is not installed; install it in the DeerFlow environment"
        ) from exc
    saver = PostgresSaver.from_conn_string(pg_uri)
    saver.setup()
    return saver
