import asyncio
from pathlib import Path
from typing import Iterable, List

import orjson
from rich import print, progress

from ..config import settings
from ..models import PersonNormalizationResponse
from .name_normalizer import normalize_people_in_file_async


CACHE_DIR = settings.project_root / "cache"
CACHE_PERSONS_LOCAL = CACHE_DIR / "persons_local_normalized.jsonl"


def iter_markdown_pages() -> Iterable[Path]:
    pages_dir: Path = settings.pages_dir
    yield from sorted(pages_dir.glob("*.md"))


def _serialize_people(norm_resp: PersonNormalizationResponse) -> List[bytes]:
    note_id = norm_resp.note_id
    payloads: List[bytes] = []
    for person in norm_resp.people:
        if not person.is_person:
            continue
        if person.confidence < 0.6:
            continue

        candidate_id = f"{note_id}:{person.local_person_id}"
        snippet = person.snippet_evidence[0].snippet if person.snippet_evidence else None

        record = {
            "candidate_id": candidate_id,
            "note_id": person.note_id,
            "local_person_id": person.local_person_id,
            "normalized_full_name": person.normalized_full_name,
            "canonical_name_in_note": person.canonical_name_in_note,
            "surface_forms": person.surface_forms,
            "name_parts": person.name_parts.model_dump(),
            "note_year_context": person.note_year_context,
            "person_confidence": person.confidence,
            "is_person": person.is_person,
            "snippet_preview": snippet,
            }
        payloads.append(orjson.dumps(record))
    return payloads


async def _process_note(path: Path) -> List[bytes]:
    norm_resp = await normalize_people_in_file_async(path)
    records = _serialize_people(norm_resp)
    return records


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
    queue: asyncio.Queue = asyncio.Queue()

    async def run_with_limit(path: Path):
        async with limiter:
            try:
                records = await _process_note(path)
                await queue.put((path, records, None))
            except Exception as exc:
                await queue.put((path, [], exc))

    tasks = [asyncio.create_task(run_with_limit(path)) for path in pages]

    async def writer():
        with open(CACHE_PERSONS_LOCAL, "wb") as f_out, progress.Progress() as pbar:
            task = pbar.add_task("[green]Сканирование заметок и людей...", total=len(pages))
            processed = 0
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                path, records, error = item
                if error:
                    print(f"[red]Ошибка при обработке {path.name}:[/red] {error}")
                else:
                    for payload in records:
                        f_out.write(payload)
                        f_out.write(b"\n")
                pbar.advance(task)
                processed += 1
                queue.task_done()
            if processed < len(pages):
                remaining = len(pages) - processed
                pbar.advance(task, remaining)

    writer_task = asyncio.create_task(writer())

    await asyncio.gather(*tasks)
    await queue.put(None)
    await writer_task

    print(f"[bold green]Готово.[/bold green] Кандидаты записаны в {CACHE_PERSONS_LOCAL}")


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
