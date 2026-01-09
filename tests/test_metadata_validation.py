from src.models import NoteMetadata
from src.tools.note_metadata import validate_metadata


def test_validates_year_range():
    meta = NoteMetadata(note_id="test", primary_year=500, reliability=0.9)
    fixed, warnings = validate_metadata(meta, "Текст без годов")

    assert fixed.primary_year is None
    assert any("вне разумного диапазона" in w for w in warnings)
    assert fixed.reliability <= 0.3


def test_warns_when_year_not_in_text():
    meta = NoteMetadata(note_id="test", primary_year=1812, reliability=0.9)
    text = "В тексте есть 1820 и 1830 годы."

    fixed, warnings = validate_metadata(meta, text)

    assert fixed.primary_year == 1812
    assert any("не найден в тексте" in w for w in warnings)
    assert fixed.reliability <= 0.5


def test_swaps_invalid_year_range():
    meta = NoteMetadata(note_id="test", year_start=1900, year_end=1800)
    fixed, warnings = validate_metadata(meta, "какой-то текст")

    assert fixed.year_start == 1800
    assert fixed.year_end == 1900
    assert any("меняем местами" in w for w in warnings)
