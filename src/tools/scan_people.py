from pathlib import Path
from typing import Iterable

import orjson
from rich import print, progress

from ..config import settings
from ..models import PersonLocalNormalized, PersonNormalizationResponse
from .name_normalizer import normalize_people_in_file


CACHE_DIR = settings.project_root / "cache"
CACHE_PERSONS_LOCAL = CACHE_DIR / "persons_local_normalized.jsonl"


def iter_markdown_pages() -> Iterable[Path]:
    pages_dir: Path = settings.pages_dir
    yield from sorted(pages_dir.glob("*.md"))


def scan_people_over_pages(
    overwrite_cache: bool = False,
    ) -> None:
    """
    Проходит по всем страницам в data/pages, запускает нормализованный извлечения людей
    и пишет результаты в cache/persons_local_normalized.jsonl.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PERSONS_LOCAL.exists() and not overwrite_cache:
        print(f"[yellow]Кэш уже существует:[/yellow] {CACHE_PERSONS_LOCAL} (используй overwrite, если нужно пересчитать)")
        return

    with open(CACHE_PERSONS_LOCAL, "wb") as f_out:
        pages = list(iter_markdown_pages())
        if not pages:
            print(f"[red]В {settings.pages_dir} нет md-файлов[/red]")
            return

        with progress.Progress() as pbar:
            task = pbar.add_task("[green]Сканирование заметок и людей...", total=len(pages))

            for path in pages:
                try:
                    norm_resp: PersonNormalizationResponse = normalize_people_in_file(path)
                except Exception as e:
                    print(f"[red]Ошибка при обработке {path.name}:[/red] {e}")
                    pbar.advance(task)
                    continue

                note_id = path.stem
                for person in norm_resp.people:
                    # только реальные люди с нормальной уверенностью
                    if not person.is_person:
                        continue
                    if person.confidence < 0.6:
                        continue

                    # строим candidate_id
                    candidate_id = f"{note_id}:{person.local_person_id}"

                    # один короткий snippet
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

                    f_out.write(orjson.dumps(record))
                    f_out.write(b"\n")

                pbar.advance(task)

    print(f"[bold green]Готово.[/bold green] Кандидаты записаны в {CACHE_PERSONS_LOCAL}")