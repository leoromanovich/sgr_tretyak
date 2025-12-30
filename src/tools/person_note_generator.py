import re
from pathlib import Path
from typing import List
import yaml
from rich import print

from ..config import settings
from ..models import GlobalPerson

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


def render_global_person_md(person: GlobalPerson) -> str:
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
        body_lines.append(f"- {ep.note_id} — [[{ep.note_id}|См. заметку]]")

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


def write_person_notes(persons: List[GlobalPerson]) -> None:
    out_dir = settings.obsidian_persons_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for gp in persons:
        filename = filename_from_full_name(gp.canonical_full_name)
        path = out_dir / filename
        md_text = render_global_person_md(gp)
        path.write_text(md_text, encoding="utf-8")
        print(f"[green]Создано:[/green] {path.name}")