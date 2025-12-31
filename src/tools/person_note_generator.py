import re
from pathlib import Path
from typing import Dict, List

import yaml
from rich import print

from ..config import settings
from ..models import GlobalPerson
from .note_naming import build_note_filename

def filename_from_full_name(full_name: str) -> str:
    """
    Собираем имя файла из полного имени:
    - урезаем пробелы по краям;
    - заменяем последовательности пробелов на "_";
    - убираем запрещённые для файлов символы.
    """
    name = full_name.strip()
    # нормализуем пробелы
    name = re.sub(r"\s+", " ", name)
    # заменяем пробелы на "_"
    name = name.replace(" ", "_")

    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '')

    return name + ".md"


def render_global_person_md(person: GlobalPerson, note_link_map: Dict[str, str]) -> str:
    frontmatter = {
        "type": "person",
        "full_name": person.canonical_full_name,
        "global_person_id": person.global_person_id,
        "episodes": [
            {
                "note_id": ep.note_id,
                "year": ep.note_year_context,
                }
            for ep in person.episodes
            ],
        }

    fm = yaml.safe_dump(frontmatter, allow_unicode=True).strip()

    # тело заметки
    body_lines = [
        f"# {person.canonical_full_name}",
        "",
        "## Эпизоды",
        ]

    for ep in person.episodes:
        target = note_link_map.get(ep.note_id, ep.note_id)
        body_lines.append(f"- {ep.note_id} — [[{target}|См. заметку]]")

    body_lines += [
        "",
        "## Краткое резюме",
        "_TODO_",
        "",
        "## Связанные люди",
        "_TODO_",
        "",
        ]

    body = "\n".join(body_lines)

    return f"---\n{fm}\n---\n\n{body}"


def build_note_link_map(persons: List[GlobalPerson]) -> Dict[str, str]:
    pages_dir = settings.pages_dir
    link_map: Dict[str, str] = {}
    needed_ids = {ep.note_id for person in persons for ep in person.episodes}
    for note_id in needed_ids:
        path = pages_dir / f"{note_id}.md"
        if not path.exists():
            link_map[note_id] = note_id
            continue
        text = path.read_text(encoding="utf-8")
        filename = build_note_filename(note_id, text)
        link_map[note_id] = filename[:-3] if filename.endswith(".md") else filename
    return link_map


def write_person_notes(persons: List[GlobalPerson]) -> None:
    out_dir = settings.obsidian_persons_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    note_link_map = build_note_link_map(persons)

    for gp in persons:
        filename = filename_from_full_name(gp.canonical_full_name)
        path = out_dir / filename
        md_text = render_global_person_md(gp, note_link_map)
        path.write_text(md_text, encoding="utf-8")
        print(f"[green]Создано:[/green] {path.name}")
