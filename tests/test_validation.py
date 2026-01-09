"""Unit tests for surface form validation utilities."""

from src.models import PersonExtractionResponse, PersonLocal
from src.tools.validation import (
    find_in_text_fuzzy,
    validate_surface_forms,
    validate_and_fix_extraction,
)


def test_find_exact_match():
    """Exact substring matches are detected."""
    assert find_in_text_fuzzy("Иванов", "Пришёл Иванов и сказал") is True


def test_find_case_insensitive_and_spacing():
    """Case and spacing differences are ignored."""
    assert find_in_text_fuzzy("иванов", "Пришёл Иванов") is True
    assert find_in_text_fuzzy("ИВАН   ИВАНОВ", "Иван Иванов пришёл") is True


def test_find_fuzzy_match_handles_suffix():
    """Minor suffix variations pass the fuzzy threshold."""
    assert find_in_text_fuzzy("Ивановым", "говорил с Ивановым") is True


def test_find_fuzzy_missing_name():
    """Missing names are not validated."""
    assert find_in_text_fuzzy("Петров", "Пришёл Иванов") is False


def test_hallucinated_full_name_flagged():
    """Hallucinated longer names should fail when only short form is present."""
    text = "Художник Корин написал картину"
    assert find_in_text_fuzzy("Павел Дмитриевич Корин", text) is False
    assert find_in_text_fuzzy("Корин", text) is True


def test_validate_surface_forms_splits_lists():
    """validate_surface_forms returns valid/invalid partitions."""
    text = "М.И. Кутузов принял командование. Кутузов решил отступать."
    forms = [
        "М.И. Кутузов",
        "Кутузов",
        "Михаил Илларионович Кутузов",
    ]

    valid, invalid = validate_surface_forms(forms, text)

    assert "М.И. Кутузов" in valid
    assert "Кутузов" in valid
    assert "Михаил Илларионович Кутузов" in invalid


def test_validate_and_fix_removes_empty_persons():
    """Persons without any validated surface forms are dropped."""
    text = "Художник Корин написал картину"
    response = PersonExtractionResponse(
        note_id="test",
        people=[
            PersonLocal(
                note_id="test",
                local_person_id="p1",
                surface_forms=["Корин"],
                canonical_name_in_note="Корин",
                is_person=True,
                confidence=0.9,
            ),
            PersonLocal(
                note_id="test",
                local_person_id="p2",
                surface_forms=["Несуществующий Человек"],
                canonical_name_in_note="Несуществующий Человек",
                is_person=True,
                confidence=0.8,
            ),
        ],
    )

    fixed, stats = validate_and_fix_extraction(response, text)

    assert len(fixed.people) == 1
    assert fixed.people[0].local_person_id == "p1"
    assert stats["removed_persons"] == 1
    assert stats["removed_forms"] >= 1
