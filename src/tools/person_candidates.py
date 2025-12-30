from pathlib import Path
from typing import Iterable, List

import orjson
from rich import print

from ..config import settings
from ..models import PersonCandidate


CACHE_DIR = settings.project_root / "cache"
CACHE_PERSONS_LOCAL = CACHE_DIR / "persons_local_normalized.jsonl"


def load_person_candidates() -> List[PersonCandidate]:
    """
    Читает cache/persons_local_normalized.jsonl и возвращает список PersonCandidate.
    """
    path = CACHE_PERSONS_LOCAL
    if not path.exists():
        raise RuntimeError(f"Кэш кандидатов не найден: {path} (сначала запусти scan-people)")

    candidates: List[PersonCandidate] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            data = orjson.loads(line)
            candidates.append(PersonCandidate.model_validate(data))
    return candidates


def debug_print_candidates(limit: int = 20) -> None:
    """
    Быстрый просмотр первых N кандидатов.
    """
    candidates = load_person_candidates()
    print(f"[bold]Всего кандидатов:[/bold] {len(candidates)}")
    for c in candidates[:limit]:
        print(
            f"- {c.candidate_id}: {c.normalized_full_name!r} "
            f"(canonical={c.canonical_name_in_note!r}, year={c.note_year_context}, "
            f"conf={c.person_confidence}, note={c.note_id})"
            )