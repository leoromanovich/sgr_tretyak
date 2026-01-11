import asyncio
import pytest

from src.models import (
    CandidateWithType,
    MentionExtractionResponse,
    MentionGroup,
    MentionGroupingResponse,
    NoteMetadata,
    PersonMention,
)
from src.tools import people_extractor
from src.tools import mention_extractor


def test_two_pass_pipeline_converts_groups(monkeypatch):
    """Тест старого подхода (chunked=False) — проверяет конвертацию групп."""
    mentions_resp = MentionExtractionResponse(
        note_id="test",
        mentions=[
            PersonMention(
                mention_id="m1",
                text_span="Михаил Илларионович Кутузов",
                context_snippet="В 1812 году Михаил Илларионович Кутузов принял командование.",
                inline_year=1812,
            ),
            PersonMention(
                mention_id="m2",
                text_span="Кутузов",
                context_snippet="Кутузов решил отступать.",
            ),
        ],
    )
    groups_resp = MentionGroupingResponse(
        note_id="test",
        groups=[
            MentionGroup(
                group_id="g1",
                mention_ids=["m1", "m2"],
                canonical_name="Михаил Илларионович Кутузов",
                is_person=True,
                confidence=0.9,
            )
        ],
    )

    async def fake_extract_mentions(note_id, text, metadata):
        return mentions_resp

    async def fake_group_mentions(note_id, mentions, text):
        assert mentions == mentions_resp.mentions
        return groups_resp

    monkeypatch.setattr(people_extractor, "extract_mentions_async", fake_extract_mentions)
    monkeypatch.setattr(people_extractor, "group_mentions_async", fake_group_mentions)

    metadata = NoteMetadata(note_id="test")
    result = asyncio.run(
        people_extractor.extract_people_two_pass_async(
            note_id="test",
            text="sample text",
            metadata=metadata,
            validate=False,
            chunked=False,  # Используем старый подход для этого теста
        )
    )

    assert len(result.people) == 1
    person = result.people[0]
    assert person.canonical_name_in_note == "Михаил Илларионович Кутузов"
    assert set(person.surface_forms) == {"Михаил Илларионович Кутузов", "Кутузов"}
    assert person.note_year_context == 1812
    assert person.note_year_source == "inline"


def test_two_pass_chunked_pipeline(monkeypatch):
    """Тест нового chunked подхода (по умолчанию)."""
    # Мокаем extract_mentions_chunked_async
    mentions_resp = MentionExtractionResponse(
        note_id="test",
        mentions=[
            PersonMention(
                mention_id="m1",
                text_span="Павел Корин",
                context_snippet="Извлечено из chunked extraction",
                likely_person=True,
            ),
            PersonMention(
                mention_id="m2",
                text_span="Е.В. Вучетича",
                context_snippet="Извлечено из chunked extraction",
                likely_person=True,
            ),
        ],
    )
    groups_resp = MentionGroupingResponse(
        note_id="test",
        groups=[
            MentionGroup(
                group_id="g1",
                mention_ids=["m1"],
                canonical_name="Павел Корин",
                is_person=True,
                confidence=0.95,
            ),
            MentionGroup(
                group_id="g2",
                mention_ids=["m2"],
                canonical_name="Е.В. Вучетич",
                is_person=True,
                confidence=0.9,
            ),
        ],
    )

    async def fake_extract_mentions_chunked(note_id, text, metadata, ner_predictions=None):
        return mentions_resp

    async def fake_group_mentions(note_id, mentions, text):
        return groups_resp

    monkeypatch.setattr(
        people_extractor, "extract_mentions_chunked_async", fake_extract_mentions_chunked
    )
    monkeypatch.setattr(people_extractor, "group_mentions_async", fake_group_mentions)

    metadata = NoteMetadata(note_id="test")
    result = asyncio.run(
        people_extractor.extract_people_two_pass_async(
            note_id="test",
            text="## Метаинформация\n\nПавел Корин\n\n## Описание\n\nЕ.В. Вучетича",
            metadata=metadata,
            validate=False,
            # chunked=True по умолчанию
        )
    )

    assert len(result.people) == 2
    names = {p.canonical_name_in_note for p in result.people}
    assert "Павел Корин" in names
    assert "Е.В. Вучетич" in names


def test_split_into_chunks():
    """Тест функции разбиения текста на chunks."""
    text = """## Метаинформация

**Автор:** Корин Павел
**Год:** 1935

## Описание

Первый параграф описания.

Второй параграф описания.

Третий параграф."""

    chunks = mention_extractor.split_into_chunks(text)

    assert len(chunks) == 4  # meta + 3 параграфа
    assert "Корин Павел" in chunks[0]  # meta
    assert "Первый параграф" in chunks[1]
    assert "Второй параграф" in chunks[2]
    assert "Третий параграф" in chunks[3]


def test_split_into_chunks_no_headers():
    """Тест разбиения текста без заголовков — весь текст как один chunk."""
    text = "Просто текст без заголовков."
    chunks = mention_extractor.split_into_chunks(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_candidates_to_mentions():
    """Тест конвертации кандидатов в mentions."""
    candidates = {
        "Павел Корин": CandidateWithType(name="Павел Корин", is_person=True),
        "Москва": CandidateWithType(name="Москва", is_person=False),
    }

    mentions = mention_extractor._candidates_to_mentions(candidates, "test")

    assert len(mentions) == 2
    names = {m.text_span for m in mentions}
    assert "Павел Корин" in names
    assert "Москва" in names

    # Проверяем likely_person
    for m in mentions:
        if m.text_span == "Павел Корин":
            assert m.likely_person is True
        elif m.text_span == "Москва":
            assert m.likely_person is False


def test_merge_ner_candidates():
    """Тест мёржа NER кандидатов с LLM кандидатами."""
    from src.tools.ner_provider import NERPrediction

    llm_candidates = {
        "Павел Корин": CandidateWithType(name="Павел Корин", is_person=True),
    }

    ner_predictions = [
        NERPrediction(word="Павел", entity="FIRST_NAME", score=0.99),
        NERPrediction(word="Корин", entity="LAST_NAME", score=0.99),
        NERPrediction(word="Тихон", entity="FIRST_NAME", score=0.98),  # Новый
        NERPrediction(word="Россия", entity="COUNTRY", score=0.99),  # Не человек
    ]

    merged = mention_extractor._merge_ner_candidates(llm_candidates, ner_predictions)

    # Павел и Корин добавлены из NER (не было в LLM)
    # Тихон добавлен из NER
    # Россия НЕ добавлена (COUNTRY, не person entity)
    assert "Павел Корин" in merged  # Был в LLM
    assert "Павел" in merged  # Добавлен из NER
    assert "Корин" in merged  # Добавлен из NER
    assert "Тихон" in merged  # Добавлен из NER
    assert "Россия" not in merged  # COUNTRY не добавляется
    assert merged["Тихон"].is_person is True
