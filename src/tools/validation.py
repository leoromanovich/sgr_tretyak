"""
Utilities for validating that surface forms returned by the extractor
actually appear in the source text.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from ..models import PersonExtractionResponse


def normalize_for_comparison(text: str) -> str:
    """Normalize whitespace/case to make fuzzy matching more reliable."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_in_text_fuzzy(form: str, text: str, threshold: float = 0.85) -> bool:
    """
    Returns True if form appears in text (case/whitespace insensitive) or
    has a fuzzy match above threshold.
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
