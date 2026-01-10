"""
Utilities for validating that surface forms returned by the extractor
actually appear in the source text.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from ..models import PersonExtractionResponse


# Типичные русские падежные окончания для существительных и прилагательных
RUSSIAN_CASE_ENDINGS = [
    # Существительные мужского рода
    "а", "я", "у", "ю", "ом", "ем", "ём", "е",
    # Существительные женского рода
    "ы", "и", "ой", "ей", "ою", "ею",
    # Существительные множественного числа
    "ов", "ев", "ёв", "ей", "ам", "ям", "ами", "ями", "ах", "ях",
    # Прилагательные
    "ого", "его", "ому", "ему", "им", "ым", "ой", "ей", "ую", "юю",
    "ие", "ые", "их", "ых", "ими", "ыми",
]


def normalize_for_comparison(text: str) -> str:
    """Normalize whitespace/case to make fuzzy matching more reliable."""
    text = text.lower()
    # Нормализация ё -> е для единообразия
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_russian_ending(word: str) -> str:
    """
    Попытка удалить русское падежное окончание для получения основы.
    Сортируем по длине (длинные сначала) чтобы "ами" проверялось раньше "и".
    """
    word_lower = word.lower().replace("ё", "е")

    # Сортируем окончания по длине (длинные первые)
    sorted_endings = sorted(RUSSIAN_CASE_ENDINGS, key=len, reverse=True)

    for ending in sorted_endings:
        if word_lower.endswith(ending) and len(word_lower) > len(ending) + 2:
            return word_lower[:-len(ending)]

    return word_lower


def get_word_stems(form: str) -> List[str]:
    """
    Получить основы всех слов в форме имени.
    Возвращает список основ для сравнения.
    """
    words = form.split()
    stems = []
    for word in words:
        # Пропускаем инициалы (типа "И." или "П.Д.")
        if re.match(r'^[А-ЯЁA-Z]\.$', word) or re.match(r'^[А-ЯЁA-Z]\.[А-ЯЁA-Z]\.$', word):
            stems.append(word.lower().replace("ё", "е"))
        else:
            stems.append(strip_russian_ending(word))
    return stems


def stems_match(form_stems: List[str], text: str) -> bool:
    """
    Проверяет, что все основы слов из формы присутствуют в тексте.
    Это позволяет матчить "патриарх Тихон" с "патриарха Тихона".
    """
    norm_text = normalize_for_comparison(text)

    for stem in form_stems:
        if len(stem) < 3:
            # Слишком короткая основа - ищем точное совпадение
            if stem not in norm_text:
                return False
        else:
            # Ищем основу в тексте
            if stem not in norm_text:
                return False

    return True


def find_in_text_fuzzy(form: str, text: str, threshold: float = 0.75) -> bool:
    """
    Returns True if form appears in text (case/whitespace insensitive) or
    has a fuzzy match above threshold.

    Для русского языка также проверяет совпадение основ слов,
    чтобы учесть падежные окончания.
    """
    if not form or not text:
        return False

    norm_form = normalize_for_comparison(form)
    norm_text = normalize_for_comparison(text)

    if not norm_form:
        return False

    # Fast exact match
    if norm_form in norm_text:
        return True

    # Проверка по основам слов (для русских падежей)
    form_stems = get_word_stems(form)
    if len(form_stems) > 0 and stems_match(form_stems, text):
        return True

    # Fuzzy scan for longer forms to catch slight OCR/LLM variations
    if len(norm_form) > 5:
        window_size = len(norm_form) + 5
        max_start = max(len(norm_text) - len(norm_form) + 1, 1)
        for idx in range(max_start):
            window = norm_text[idx : idx + window_size]
            ratio = SequenceMatcher(None, norm_form, window).ratio()
            if ratio >= threshold:
                return True

    return False


def validate_surface_forms(
    forms: List[str],
    text: str,
    threshold: float = 0.85,
) -> Tuple[List[str], List[str]]:
    """Split surface forms into valid/invalid buckets."""
    valid: List[str] = []
    invalid: List[str] = []

    for form in forms:
        if find_in_text_fuzzy(form, text, threshold):
            valid.append(form)
        else:
            invalid.append(form)

    return valid, invalid


def validate_and_fix_extraction(
    response: PersonExtractionResponse,
    text: str,
    threshold: float = 0.85,
) -> Tuple[PersonExtractionResponse, Dict[str, int]]:
    """
    Remove surface forms (and people) that cannot be backed by the text.
    Returns the fixed response and validation statistics.
    """
    stats = {
        "total_forms": 0,
        "valid_forms": 0,
        "removed_forms": 0,
        "removed_persons": 0,
    }

    fixed_people = []

    for person in response.people:
        stats["total_forms"] += len(person.surface_forms)
        valid, invalid = validate_surface_forms(person.surface_forms, text, threshold=threshold)
        stats["valid_forms"] += len(valid)
        stats["removed_forms"] += len(invalid)

        if not valid:
            stats["removed_persons"] += 1
            continue

        person.surface_forms = valid
        if person.canonical_name_in_note not in valid:
            person.canonical_name_in_note = valid[0]

        fixed_people.append(person)

    response.people = fixed_people
    return response, stats
