from src.models import NameParts, PersonCandidate
from src.tools.cluster_people import cheap_decision


def make_candidate(
    *,
    candidate_id: str,
    canonical: str,
    last_name: str | None = None,
    first_name: str | None = None,
    normalized: str | None = None,
    role: str | None = None,
    year: int | None = None,
):
    return PersonCandidate(
        candidate_id=candidate_id,
        note_id=candidate_id.split(":")[0],
        local_person_id=candidate_id.split(":")[1],
        normalized_full_name=normalized,
        canonical_name_in_note=canonical,
        surface_forms=[canonical],
        name_parts=NameParts(
            last_name=last_name,
            first_name=first_name,
        ),
        note_year_context=year,
        person_confidence=0.9,
        is_person=True,
        snippet_preview=None,
        role=role,
    )


def test_same_role_with_matching_names_prefers_same_person():
    c1 = make_candidate(
        candidate_id="1:a",
        canonical="Иван Иванов",
        last_name="Иванов",
        first_name="Иван",
        normalized="Иван Иванов",
        role="архитектор",
    )
    c2 = make_candidate(
        candidate_id="2:b",
        canonical="Иванов",
        last_name="Иванов",
        first_name="Иван",
        normalized="Иван Иванов",
        role="архитектор",
    )

    assert cheap_decision(c1, c2) == "same_person"


def test_different_roles_split_when_names_not_full():
    c1 = make_candidate(
        candidate_id="1:a",
        canonical="Архитектор Иванов",
        last_name="Иванов",
        role="архитектор",
    )
    c2 = make_candidate(
        candidate_id="2:b",
        canonical="Художник Иванов",
        last_name="Иванов",
        role="художник",
    )

    assert cheap_decision(c1, c2) == "different_person"
