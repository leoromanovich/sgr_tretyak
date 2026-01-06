from src.models import NameParts, PersonCandidate
from src.tools.cluster_people import cheap_decision


def make_candidate(
    *,
    candidate_id: str,
    last_name: str | None = None,
    first_name: str | None = None,
    patronymic: str | None = None,
    canonical: str,
    normalized: str | None = None,
    forms: list[str] | None = None,
    year: int | None = None,
):
    return PersonCandidate(
        candidate_id=candidate_id,
        note_id=candidate_id.split(":")[0],
        local_person_id=candidate_id.split(":")[1],
        normalized_full_name=normalized,
        canonical_name_in_note=canonical,
        surface_forms=forms or [],
        name_parts=NameParts(
            last_name=last_name,
            first_name=first_name,
            patronymic=patronymic,
        ),
        note_year_context=year,
        person_confidence=1.0,
        is_person=True,
        snippet_preview=None,
    )


def test_same_full_name_year_close():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        first_name="Павел",
        patronymic="Дмитриевич",
        canonical="Павел Дмитриевич Корин",
        normalized="Павел Дмитриевич Корин",
        forms=["Павел Корин"],
        year=1935,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Павел",
        patronymic="Дмитриевич",
        canonical="П.Д. Корин",
        normalized="Павел Дмитриевич Корин",
        forms=["П.Д. Корин"],
        year=1936,
    )
    assert cheap_decision(c1, c2) == "same_person"


def test_different_years_far_apart():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        first_name="Павел",
        patronymic="Дмитриевич",
        canonical="Павел Корин",
        normalized="Павел Корин",
        forms=["Корин"],
        year=1800,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Павел",
        patronymic="Дмитриевич",
        canonical="Павел Корин",
        normalized="Павел Корин",
        forms=["Корин"],
        year=1900,
    )
    assert cheap_decision(c1, c2) == "different_person"


def test_surface_forms_overlap_same_year():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        first_name="Павел",
        patronymic="Дмитриевич",
        canonical="Павел Корин",
        forms=["Павел Корин"],
        year=1935,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Павел",
        canonical="П.Д. Корин",
        forms=["Павел Корин", "Корин"],
        year=1936,
    )
    assert cheap_decision(c1, c2) == "same_person"


def test_same_surface_forms_far_years():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        first_name="Павел",
        canonical="Павел Корин",
        forms=["Павел Корин"],
        year=1935,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Павел",
        canonical="П.Д. Корин",
        forms=["Павел Корин"],
        year=1980,
    )
    assert cheap_decision(c1, c2) == "different_person"


def test_same_last_name_different_first_and_no_overlap_forms():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        first_name="Павел",
        canonical="Павел Корин",
        forms=["Павел"],
        year=1935,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Дмитрий",
        canonical="Дмитрий Корин",
        forms=["Дмитрий"],
        year=1935,
    )
    assert cheap_decision(c1, c2) == "different_person"


def test_missing_first_name_but_shared_forms_and_year():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        canonical="Корин",
        forms=["Корин", "П. Корин"],
        year=1935,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Павел",
        patronymic="Дмитриевич",
        canonical="Павел Корин",
        forms=["П. Корин"],
        year=1936,
    )
    assert cheap_decision(c1, c2) == "same_person"


def test_identical_canonical_names():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Ленин",
        first_name="Владимир",
        patronymic="Ильич",
        canonical="Владимир Ильич Ленин",
        forms=["В. Ленин"],
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Ленин",
        first_name=None,
        patronymic=None,
        canonical="Владимир Ильич Ленин",
        forms=["Ленин"],
    )
    assert cheap_decision(c1, c2) == "same_person"


def test_far_years_same_surface_forms_returns_different():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Корин",
        first_name="Павел",
        canonical="Павел Корин",
        forms=["Корин"],
        year=1850,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Корин",
        first_name="Павел",
        canonical="Павел Корин",
        forms=["Корин"],
        year=1950,
    )
    assert cheap_decision(c1, c2) == "different_person"


def test_same_first_name_surface_form_without_last_name():
    c1 = make_candidate(
        candidate_id="1:a",
        last_name="Шишкин",
        first_name="Иван",
        canonical="Иван Шишкин",
        forms=["Иван"],
        year=1883,
    )
    c2 = make_candidate(
        candidate_id="2:b",
        last_name="Достоевский",
        first_name="Фёдор",
        canonical="Фёдор Достоевский",
        forms=["Иван"],
        year=1883,
    )
    assert cheap_decision(c1, c2) is None
