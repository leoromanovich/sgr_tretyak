from pathlib import Path
import typer
from rich import print

from .config import settings
from .tools.note_metadata import extract_note_metadata_from_file
from .tools.people_extractor import extract_people_from_file
from .tools.name_normalizer import normalize_people_in_file
from .tools.scan_people import scan_people_over_pages
from .tools.person_candidates import debug_print_candidates
from .tools.cluster_people import cluster_people, load_or_cluster_global_persons
from .tools.person_note_generator import write_person_notes
from .tools.note_linker import link_persons_in_pages
from .utils.logging_utils import setup_file_logging


setup_file_logging("cli")


app = typer.Typer(help="Исторический SGR-пайплайн")


@app.command()
def list_pages():
    """Показать список md-файлов в data/pages"""
    pages_dir: Path = settings.pages_dir
    print("[bold]Файлы в data/pages:[/bold]")
    for p in sorted(pages_dir.glob("*.md")):
        print(" -", p.name)


@app.command()
def note_meta(filename: str = typer.Argument(..., help="Имя файла в data/pages, например '8332.md'")):
    """Извлечь метаданные из одной заметки"""
    pages_dir: Path = settings.pages_dir
    path = pages_dir / filename
    if not path.exists():
        print(f"[red]Файл не найден:[/red] {path}")
        raise typer.Exit(code=1)

    meta_resp = extract_note_metadata_from_file(path)
    print("[bold]Метаданные:[/bold]")
    print(meta_resp.model_dump())


@app.command()
def note_people(filename: str = typer.Argument(..., help="Имя файла в data/pages, например '8332.md'")):
    """Извлечь людей из одной заметки (без нормализации)"""
    pages_dir: Path = settings.pages_dir
    path = pages_dir / filename
    if not path.exists():
        print(f"[red]Файл не найден:[/red] {path}")
        raise typer.Exit(code=1)

    people_resp = extract_people_from_file(path)
    print("[bold]Найденные люди (raw):[/bold]")
    if not people_resp.people:
        print(" (пусто)")
    else:
        for p in people_resp.people:
            print(
                f"- [id={p.local_person_id}] {p.canonical_name_in_note!r}, "
                f"forms={p.surface_forms}, "
                f"is_person={p.is_person}, conf={p.confidence}, "
                f"year={p.note_year_context} ({p.note_year_source})"
                )


@app.command()
def note_people_normalized(filename: str = typer.Argument(..., help="Имя файла в data/pages, например '8332.md'")):
    """Извлечь и нормализовать людей в одной заметке"""
    pages_dir: Path = settings.pages_dir
    path = pages_dir / filename
    if not path.exists():
        print(f"[red]Файл не найден:[/red] {path}")
        raise typer.Exit(code=1)

    norm_resp = normalize_people_in_file(path)
    print("[bold]Нормализованные люди:[/bold]")
    if not norm_resp.people:
        print(" (пусто)")
    else:
        for p in norm_resp.people:
            print(
                f"- [id={p.local_person_id}] canonical={p.canonical_name_in_note!r}, "
                f"normalized={p.normalized_full_name!r}, "
                f"parts=({p.name_parts.last_name!r}, {p.name_parts.first_name!r}, {p.name_parts.patronymic!r}), "
                f"forms={p.surface_forms}"
                )
            if p.abbreviation_links:
                print("   abbreviation_links:")
                for link in p.abbreviation_links:
                    print(
                        f"    * {link.from_form!r} -> {link.to_form!r}, "
                        f"conf={link.confidence}, reason={link.reasoning}"
                        )

@app.command()
def scan_people(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Пересчитать кэш, даже если файл cache/persons_local_normalized.jsonl уже существует",
        ),
    workers: int = typer.Option(
        8,
        "--workers",
        "-w",
        min=1,
        help="Сколько заметок обрабатывать параллельно",
        ),
    ):
    """
    Пройти по всем data/pages и собрать нормализованных персон в кэш.
    """
    scan_people_over_pages(overwrite_cache=overwrite, workers=workers)

@app.command()
def show_candidates(
    limit: int = typer.Option(20, "--limit", help="Сколько кандидатов показать"),
    ):
    """
    Показать первые N глобальных кандидатов (из кэша).
    """
    debug_print_candidates(limit=limit)

@app.command()
def cluster(
    conf_threshold: float = typer.Option(0.8),
    match_workers: int = typer.Option(
        8,
        "--match-workers",
        help="Одновременных запросов к LLM при сравнении пар",
        min=1,
        ),
    use_match_analysis: bool = typer.Option(
        False,
        "--use-match-analysis",
        help="Включить дополнительный reasoning-проход для LLM матчинга",
        ),
    ):
    """
    Запустить кластеризацию персон по всему пулу заметок.
    """
    clusters = cluster_people(
        conf_threshold=conf_threshold,
        match_workers=match_workers,
        use_match_analysis=use_match_analysis,
        )
    for gp in clusters:
        print(f"{gp.global_person_id}: {gp.canonical_full_name!r} ({len(gp.members)} entries)")


@app.command()
def gen_person_notes(
    conf_threshold: float = typer.Option(0.8),
    use_match_analysis: bool = typer.Option(
        False,
        "--use-match-analysis",
        help="Указать, что кэш собран с доп. reasoning-проходом",
        ),
    ):
    """
    Сгенерировать Obsidian-заметки для глобальных персон
    """
    persons = load_or_cluster_global_persons(
        conf_threshold=conf_threshold,
        use_match_analysis=use_match_analysis,
        )
    write_person_notes(persons)

@app.command()
def link_persons(
    conf_threshold: float = typer.Option(
        0.8,
        "--conf-threshold",
        help="Какой порог совпадения ожидается у кэша кластеров",
        ),
    use_match_analysis: bool = typer.Option(
        False,
        "--use-match-analysis",
        help="Указать, что кэш кластеров собран с доп. reasoning-проходом",
        ),
    ):
    """
    Проставить ссылки на людей в заметках и сохранить результат в data/obsidian/items.
    """
    link_persons_in_pages(
        conf_threshold=conf_threshold,
        use_match_analysis=use_match_analysis,
        )


if __name__ == "__main__":
    app()
