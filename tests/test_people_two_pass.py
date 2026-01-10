import asyncio
import pytest

from src.models import (
    MentionExtractionResponse,
    MentionGroup,
    MentionGroupingResponse,
    NoteMetadata,
    PersonMention,
)
from src.tools import people_extractor


def test_two_pass_pipeline_converts_groups(monkeypatch):
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
        )
    )

    assert len(result.people) == 1
    person = result.people[0]
    assert person.canonical_name_in_note == "Михаил Илларионович Кутузов"
    assert set(person.surface_forms) == {"Михаил Илларионович Кутузов", "Кутузов"}
    assert person.note_year_context == 1812
    assert person.note_year_source == "inline"
