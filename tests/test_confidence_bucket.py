"""Tests for confidence bucket logic."""

import orjson
import pytest

from src.models import (
    PersonNormalizationResponse,
    PersonLocalNormalized,
    confidence_to_bucket,
)
from src.tools.scan_people import _serialize_people


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.95, "high"),
        (0.80, "high"),
        (0.79, "medium"),
        (0.60, "medium"),
        (0.59, "low"),
        (0.40, "low"),
        (0.39, "very_low"),
        (0.10, "very_low"),
    ],
)
def test_confidence_to_bucket(confidence, expected):
    assert confidence_to_bucket(confidence) == expected


def test_low_confidence_candidates_saved():
    """Candidates below the old threshold are still serialized with bucket info."""
    person = PersonLocalNormalized(
        note_id="note1",
        local_person_id="p1",
        surface_forms=["Иванов"],
        canonical_name_in_note="Иванов",
        is_person=True,
        confidence=0.5,
    )
    resp = PersonNormalizationResponse(note_id="note1", people=[person])

    payloads = _serialize_people(resp)
    assert len(payloads) == 1

    record = orjson.loads(payloads[0])
    assert record["person_confidence"] == pytest.approx(0.5)
    assert record["confidence_bucket"] == "low"
