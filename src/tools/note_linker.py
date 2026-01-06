import re
from pathlib import Path
from typing import Dict, List, Tuple

from rich import print

from ..config import settings
from ..models import GlobalPerson, PersonCandidate
from .cluster_people import load_or_cluster_global_persons
from .note_naming import build_note_filename
from .person_candidates import load_person_candidates
from .person_note_generator import filename_from_full_name


def build_candidate_to_file_map(global_persons: List[GlobalPerson]) -> Dict[str, str]:
    """
    candidate_id -> имя файла персоны (как в data/obsidian/persons).
    """
    mapping: Dict[str, str] = {}

    for gp in global_persons:
        fname = filename_from_full_name(gp.canonical_full_name)
        for cid in gp.members:
            mapping[cid] = fname

    return mapping


def group_candidates_by_note(candidates: List[PersonCandidate]) -> Dict[str, List[PersonCandidate]]:
    by_note: Dict[str, List[PersonCandidate]] = {}
    for c in candidates:
        by_note.setdefault(c.note_id, []).append(c)
    return by_note


def build_unique_surface_forms_for_note(
    persons: List[PersonCandidate],
    cid_to_file: Dict[str, str],
    ) -> List[Tuple[str, str]]:
    """
    Возвращает список (surface_form, filename) только для тех форм,
    которые однозначно относятся к одному кандидату в рамках заметки.
    """
    form_to_candidates: Dict[str, List[PersonCandidate]] = {}

    for p in persons:
        if p.candidate_id not in cid_to_file:
            continue
        for form in p.surface_forms:
            form_to_candidates.setdefault(form, []).append(p)

    form_filename_pairs: List[Tuple[str, str]] = []

    for form, plist in form_to_candidates.items():
        if len(plist) != 1:
            # Форму использует несколько кандидатов в этой заметке — лучше не трогать
            continue
        candidate = plist[0]
        fname = cid_to_file.get(candidate.candidate_id)
        if not fname:
            continue
        form_filename_pairs.append((form, fname))

    return form_filename_pairs


import re
from typing import List, Tuple

LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
WORD_CHARS = "0-9A-Za-zА-Яа-яЁё"


def _find_form_spans(form: str, text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []

    literal_pattern = re.escape(form)
    spans = [m.span() for m in re.finditer(literal_pattern, text)]
    if spans:
        return spans

    spans = [m.span() for m in re.finditer(literal_pattern, text, flags=re.IGNORECASE)]
    if spans:
        return spans

    if not LETTER_RE.search(form):
        return []

    def build_flexible_pattern() -> str:
        parts: List[str] = []
        last = 0
        for match in LETTER_RE.finditer(form):
            parts.append(re.escape(form[last:match.start()]))
            word = re.escape(match.group(0))
            parts.append(f"{word}[а-яё]{{0,4}}")
            last = match.end()
        parts.append(re.escape(form[last:]))
        pattern = "".join(parts)

        if re.match(rf"[{WORD_CHARS}]", form):
            pattern = rf"(?<![{WORD_CHARS}])" + pattern
        if re.search(rf"[{WORD_CHARS}]$", form):
            pattern = pattern + rf"(?![{WORD_CHARS}])"
        return pattern

    flex_pattern = build_flexible_pattern()
    return [m.span() for m in re.finditer(flex_pattern, text, flags=re.IGNORECASE)]


def apply_links_to_text(text: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Делаем ОДИН проход по исходному тексту:
    1) для каждой формы находим все вхождения (по исходному text);
    2) выбрасываем пересекающиеся матчи (длинные формы в приоритете);
    3) собираем новый текст, подставляя [[file|original]] только на отмеченные интервалы.

    Так мы не трогаем уже вставленные ссылки и не получаем вложенных [[...]].
    """
    # Сортируем формы по длине (сначала длинные -> выигрывают при пересечениях)
    replacements_sorted = sorted(replacements, key=lambda x: len(x[0]), reverse=True)

    # вырезаем диапазоны, которые нельзя трогать (например, alt-текст изображений)
    excluded_spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"!\[(.*?)\]", text):
        alt_start = match.start(1)
        alt_end = match.end(1)
        excluded_spans.append((alt_start, alt_end))

    def overlaps_excluded(start: int, end: int) -> bool:
        for s, e in excluded_spans:
            if not (end <= s or start >= e):
                return True
        return False

    # 1. Собираем все матчи как интервалы в исходном тексте
    spans: List[Tuple[int, int, str, str]] = []  # (start, end, form, filename)

    for form, fname in replacements_sorted:
        for start, end in _find_form_spans(form, text):

            if overlaps_excluded(start, end):
                continue

            # проверяем, не пересекается ли этот интервал с уже выбранными
            overlap = False
            for s0, e0, _, _ in spans:
                if not (end <= s0 or start >= e0):
                    overlap = True
                    break
            if overlap:
                continue

            spans.append((start, end, form, fname))

    if not spans:
        return text

    # 2. Сортируем интервалы по позиции
    spans.sort(key=lambda x: x[0])

    # 3. Собираем новый текст
    result_parts: List[str] = []
    pos = 0

    for start, end, form, fname in spans:
        # добавляем текст до матчa
        if pos < start:
            result_parts.append(text[pos:start])

        original = text[start:end]
        link_target = fname[:-3] if fname.endswith(".md") else fname
        result_parts.append(f"[[{link_target}|{original}]]")

        pos = end

    # хвост
    if pos < len(text):
        result_parts.append(text[pos:])

    return "".join(result_parts)


def link_persons_in_pages(
    conf_threshold: float = 0.8,
    use_match_analysis: bool = False,
    ) -> None:
    """
    Основной процесс:
    1. Загружаем кандидатов.
    2. Запускаем кластеризацию (cluster_people).
    3. Строим mapping candidate_id -> person_file_name.
    4. По каждой заметке:
       - выбираем кандидатов,
       - считаем уникальные surface_forms,
       - оборачиваем их в [[...]].
    5. Сохраняем новые заметки в data/obsidian/items.
    """
    pages_dir: Path = settings.pages_dir
    out_dir: Path = settings.obsidian_items_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_person_candidates()
    if not candidates:
        raise RuntimeError("Нет кандидатов (сначала запусти scan-people).")

    print(f"[bold]Всего кандидатов:[/bold] {len(candidates)}")

    # 1. Кластеры глобальных персон
    print("[bold]Кластеризация для линковки...[/bold]")
    global_persons = load_or_cluster_global_persons(
        conf_threshold=conf_threshold,
        use_match_analysis=use_match_analysis,
        )

    # 2. Мап candidate_id -> имя файла персоны
    cid_to_file = build_candidate_to_file_map(global_persons)

    # 3. Группировка кандидатов по заметкам
    by_note = group_candidates_by_note(candidates)

    print("[bold]Линкуем заметки...[/bold]")

    for note_id, persons in by_note.items():
        src_path = pages_dir / f"{note_id}.md"
        if not src_path.exists():
            print(f"[yellow]Пропуск: файл не найден для note_id={note_id}: {src_path}[/yellow]")
            continue

        text = src_path.read_text(encoding="utf-8")
        out_filename = build_note_filename(note_id, text)

        # 4. Подбираем однозначные формы
        replacements = build_unique_surface_forms_for_note(persons, cid_to_file)

        if not replacements:
            # ничего линковать
            out_path = out_dir / out_filename
            out_path.write_text(text, encoding="utf-8")
            continue

        new_text = apply_links_to_text(text, replacements)

        out_path = out_dir / out_filename
        out_path.write_text(new_text, encoding="utf-8")
        print(f"[green]Линкована:[/green] {src_path.name} -> {out_path.relative_to(settings.project_root)}")
