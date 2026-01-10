import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Set

from ..config import settings


@lru_cache(maxsize=1)
def _trace_note_ids() -> Set[str]:
    """
    Читает переменную окружения SGR_TRACE_NOTES (разделитель — запятая)
    и возвращает множество note_id, для которых нужно дополнительно логировать пайплайн.
    """
    raw = os.getenv("SGR_TRACE_NOTES") or ""
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items


def should_trace_note(note_id: str) -> bool:
    notes = _trace_note_ids()
    if not notes:
        return False
    return "*" in notes or note_id in notes


def log_pipeline_stage(note_id: str, stage: str, payload: Any) -> None:
    """
    Сохраняет JSON с результатом конкретного этапа пайплайна в log_tracing/people_pipeline/<note_id>/.
    """
    if not should_trace_note(note_id):
        return

    trace_dir: Path = settings.project_root / "log_tracing" / "people_pipeline" / note_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = trace_dir / f"{stage}_{timestamp}.json"

    record = {
        "note_id": note_id,
        "stage": stage,
        "timestamp": timestamp,
        "payload": payload,
    }

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
