import asyncio
import pytest

from src.models import MentionGroup, MentionGroupingResponse, PersonMention
from src.tools import mention_grouper


def test_groups_same_person_forms(monkeypatch):
    mentions = [
        PersonMention(
            mention_id="m1",
            text_span="Михаил Илларионович Кутузов",
            context_snippet="...принял командование...",
        ),
        PersonMention(
            mention_id="m2",
            text_span="М.И. Кутузов",
            context_snippet="...был опытным...",
        ),
        PersonMention(
            mention_id="m3",
            text_span="Кутузов",
            context_snippet="...решил отступать...",
        ),
    ]

    async def fake_chat(messages, model_cls, **kwargs):
        return MentionGroupingResponse(
            note_id="test",
            groups=[
                MentionGroup(
                    group_id="g1",
                    mention_ids=["m1", "m2", "m3"],
                    canonical_name="Михаил Илларионович Кутузов",
                    is_person=True,
                    confidence=0.95,
                )
            ],
        )

    monkeypatch.setattr(mention_grouper, "chat_sgr_parse_async", fake_chat)

    result = asyncio.run(mention_grouper.group_mentions_async("test", mentions, "..."))
    assert len(result.groups) == 1
    assert set(result.groups[0].mention_ids) == {"m1", "m2", "m3"}
    assert "Кутузов" in result.groups[0].canonical_name


def test_separates_different_people_same_surname(monkeypatch):
    mentions = [
        PersonMention(
            mention_id="m1",
            text_span="Архитектор Иванов",
            context_snippet="спроектировал здание",
        ),
        PersonMention(
            mention_id="m2",
            text_span="Художник Иванов",
            context_snippet="расписал потолок",
        ),
    ]

    async def fake_chat(messages, model_cls, **kwargs):
        return MentionGroupingResponse(
            note_id="test",
            groups=[
                MentionGroup(
                    group_id="g1",
                    mention_ids=["m1"],
                    canonical_name="Архитектор Иванов",
                    is_person=True,
                    confidence=0.7,
                ),
                MentionGroup(
                    group_id="g2",
                    mention_ids=["m2"],
                    canonical_name="Художник Иванов",
                    is_person=True,
                    confidence=0.7,
                ),
            ],
        )

    monkeypatch.setattr(mention_grouper, "chat_sgr_parse_async", fake_chat)

    result = asyncio.run(mention_grouper.group_mentions_async("test", mentions, "..."))
    assert len(result.groups) == 2
    assert all(len(group.mention_ids) == 1 for group in result.groups)
