import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Tuple

import orjson
from rich import print
from tqdm.auto import tqdm

from ..config import settings
from ..models import PersonNormalizationResponse, confidence_to_bucket
from .name_normalizer import normalize_people_in_file_with_warnings_async


CACHE_DIR = settings.cache_dir
CACHE_PERSONS_LOCAL = CACHE_DIR / "persons_local_normalized.jsonl"
MIN_CONFIDENCE_TO_SAVE = 0.2
logger = logging.getLogger(__name__)


def iter_markdown_pages() -> Iterable[Path]:
    pages_dir: Path = settings.pages_dir
    yield from sorted(pages_dir.glob("*.md"))


def _serialize_people(norm_resp: PersonNormalizationResponse) -> List[bytes]:
    note_id = norm_resp.note_id
    payloads: List[bytes] = []
    for person in norm_resp.people:
        if not person.is_person:
            continue
        if person.confidence < MIN_CONFIDENCE_TO_SAVE:
            continue

        candidate_id = f"{note_id}:{person.local_person_id}"
        snippet = person.snippet_evidence[0].snippet if person.snippet_evidence else None
        bucket = confidence_to_bucket(person.confidence)

        record = {
            "candidate_id": candidate_id,
            "note_id": person.note_id,
            "local_person_id": person.local_person_id,
            "normalized_full_name": person.normalized_full_name,
            "normalized_last_name": person.normalized_last_name,
            "first_initial": person.first_initial,
            "patronymic_initial": person.patronymic_initial,
            "canonical_name_in_note": person.canonical_name_in_note,
            "surface_forms": person.surface_forms,
            "name_parts": person.name_parts.model_dump(),
            "note_year_context": person.note_year_context,
            "person_confidence": person.confidence,
            "confidence_bucket": bucket,
            "role": person.role,
            "role_confidence": person.role_confidence,
            "is_person": person.is_person,
            "snippet_preview": snippet,
            }
        payloads.append(orjson.dumps(record))
    return payloads


async def _process_note(path: Path) -> Tuple[List[bytes], List[str]]:
    norm_resp, warnings = await normalize_people_in_file_with_warnings_async(path)
    records = _serialize_people(norm_resp)
    return records, warnings


async def scan_people_over_pages_async(
    overwrite_cache: bool = False,
    workers: int = 8,
    ) -> None:
    """
    Проходит по всем страницам в data/pages, запускает нормализованное извлечение людей
    и пишет результаты в cache/persons_local_normalized.jsonl (async-версия).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PERSONS_LOCAL.exists() and not overwrite_cache:
        print(f"[yellow]Кэш уже существует:[/yellow] {CACHE_PERSONS_LOCAL} (используй overwrite, если нужно пересчитать)")
        return

    pages = list(iter_markdown_pages())
    if not pages:
        print(f"[red]В {settings.pages_dir} нет md-файлов[/red]")
        return

    limiter = asyncio.Semaphore(max(1, workers))
    started_at = perf_counter()
    queue: asyncio.Queue[Tuple[Path, List[bytes], List[str], Exception | None]] = asyncio.Queue()

    async def run_with_limit(path: Path):
        async with limiter:
            try:
                records, warnings = await _process_note(path)
                await queue.put((path, records, warnings, None))
            except Exception as exc:
                await queue.put((path, [], [], exc))

    tasks = [asyncio.create_task(run_with_limit(path)) for path in pages]

    async def writer():
        with open(CACHE_PERSONS_LOCAL, "wb") as f_out, tqdm(
            total=len(pages),
            desc="Сканирование заметок",
            unit="note",
            leave=True,
        ) as pbar:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                path, records, warnings, error = item
                if error:
                    print(f"[red]Ошибка при обработке {path.name}:[/red] {error}")
                else:
                    if warnings:
                        for warning in warnings:
                            logger.warning("Note %s: %s", path.stem, warning)
                    for payload in records:
                        f_out.write(payload)
                        f_out.write(b"\n")
                pbar.update(1)
                queue.task_done()

    writer_task = asyncio.create_task(writer())

    await asyncio.gather(*tasks)
    await queue.put(None)
    await queue.join()
    await writer_task

    elapsed = perf_counter() - started_at
    per_note = elapsed / len(pages) if pages else 0
    print(f"[bold green]Готово.[/bold green] Кандидаты записаны в {CACHE_PERSONS_LOCAL}")
    print(f"[blue]Скорость:[/blue] {elapsed:.1f}с всего (~{per_note:.1f}с/заметку) при {workers} воркерах")


def scan_people_over_pages(
    overwrite_cache: bool = False,
    workers: int = 8,
    ) -> None:
    """
    Синхронная обёртка поверх async-сканирования для CLI и legacy-кода.
    """
    asyncio.run(
        scan_people_over_pages_async(
            overwrite_cache=overwrite_cache,
            workers=workers,
            )
        )
