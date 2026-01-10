import asyncio
import pytest

from src.models import MentionExtractionResponse, NoteMetadata, PersonMention
from src.tools import mention_extractor


TEXT_WITH_MULTIPLE_FORMS = """
В 1812 году Михаил Илларионович Кутузов принял командование армией.
М.И. Кутузов был опытным полководцем. Кутузов решил отступать.
"""


def test_extracts_all_mention_forms(monkeypatch):
    async def fake_chat(messages, model_cls, **kwargs):
        return MentionExtractionResponse(
            note_id="test",
            mentions=[
                PersonMention(
                    mention_id="m1",
                    text_span="Михаил Илларионович Кутузов",
                    context_snippet="В 1812 году Михаил Илларионович Кутузов принял командование армией.",
                    inline_year=1812,
                ),
                PersonMention(
                    mention_id="m2",
                    text_span="М.И. Кутузов",
                    context_snippet="М.И. Кутузов был опытным полководцем.",
                ),
                PersonMention(
                    mention_id="m3",
                    text_span="Кутузов",
                    context_snippet="Кутузов решил отступать.",
                ),
            ],
        )

    monkeypatch.setattr(mention_extractor, "chat_sgr_parse_async", fake_chat)

    metadata = NoteMetadata(note_id="test", primary_year=1812)
    result = asyncio.run(
        mention_extractor.extract_mentions_async("test", TEXT_WITH_MULTIPLE_FORMS, metadata)
    )

    spans = [m.text_span for m in result.mentions]
    assert "Михаил Илларионович Кутузов" in spans
    assert "М.И. Кутузов" in spans
    assert "Кутузов" in spans
    assert len(result.mentions) == 3


def test_extracts_inline_year(monkeypatch):
    async def fake_chat(messages, model_cls, **kwargs):
        return MentionExtractionResponse(
            note_id="test",
            mentions=[
                PersonMention(
                    mention_id="m1",
                    text_span="Пестель",
                    context_snippet="В 1825 году декабрист Пестель был арестован.",
                    inline_year=1825,
                ),
            ],
        )

    monkeypatch.setattr(mention_extractor, "chat_sgr_parse_async", fake_chat)

    metadata = NoteMetadata(note_id="test")
    text = "В 1825 году декабрист Пестель был арестован."

    result = asyncio.run(mention_extractor.extract_mentions_async("test", text, metadata))
    assert result.mentions[0].inline_year == 1825


TEXT_WITH_AMBIGUOUS_NAMES = """
Архитектор Иванов спроектировал здание. Художник Иванов расписал потолок.
"""


def test_does_not_merge_different_people(monkeypatch):
    async def fake_chat(messages, model_cls, **kwargs):
        return MentionExtractionResponse(
            note_id="test",
            mentions=[
                PersonMention(
                    mention_id="m1",
                    text_span="Архитектор Иванов",
                    context_snippet="Архитектор Иванов спроектировал здание.",
                ),
                PersonMention(
                    mention_id="m2",
                    text_span="Художник Иванов",
                    context_snippet="Художник Иванов расписал потолок.",
                ),
            ],
        )

    monkeypatch.setattr(mention_extractor, "chat_sgr_parse_async", fake_chat)

    metadata = NoteMetadata(note_id="test")
    result = asyncio.run(
        mention_extractor.extract_mentions_async("test", TEXT_WITH_AMBIGUOUS_NAMES, metadata)
    )

    ivanovs = [m for m in result.mentions if "Иванов" in m.text_span]
    assert len(ivanovs) == 2
