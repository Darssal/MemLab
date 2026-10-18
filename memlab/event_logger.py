from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from memlab.io import json_dump, jsonl_append
from memlab.types import EventLog


class EventLogger:
    def __init__(self, run_id: str, event_path: str | Path) -> None:
        self.run_id = run_id
        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: EventLog) -> None:
        row = asdict(event)
        row["extra"] = row.get("extra") or {}
        jsonl_append(self.event_path, row)

    def emit_raw(self, row: dict[str, Any]) -> None:
        jsonl_append(self.event_path, row)

    @staticmethod
    def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
        json_dump(path, manifest)
