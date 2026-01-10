from typing import Any, Dict, List
from pathlib import Path
import asyncio
import json
import logging

import frontmatter

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..models import (
    MentionExtractionResponse,
    MentionGroupingResponse,
    NoteMetadata,
    NoteMetadataResponse,
    PersonExtractionResponse,
    PersonLocal,
    PersonSnippetEvidence,
)
from .note_metadata import extract_note_metadata_from_file, extract_note_metadata_from_file_async
from ..llm_prompts import add_no_think
from .validation import validate_and_fix_extraction
from .mention_extractor import extract_mentions_async, extract_mentions
from .mention_grouper import group_mentions_async, group_mentions
from .pipeline_trace import log_pipeline_stage


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Ты помощник-историк, который извлекает упоминания людей из исторической заметки в формате Markdown.

Твои задачи:

1. Найти все упоминания КОНКРЕТНЫХ людей:
   - исторические личности,
   - персонажи, явно описанные как отдельные люди.
2. НЕ считать людьми:
   - абстрактные группы ("крестьяне", "монахи", "чиновники"),
   - организации ("комиссия", "правительство", "армия"),
   - храмы, города, страны.
3. Объединять очевидные варианты одного и того же человека:
   - "И.И. Иванов", "Иван Иванович", "Иван Иванович Иванов", "Иванов" (если в контексте ясно, что это один человек).
4. Для каждого человека:
   - surface_forms — перечисли ТОЛЬКО разные формы имени, встречающиеся в тексте (как минимум 1, максимум 10). Если форм больше, выбери самые разные и представительные. Одинаковые строки нельзя повторять.
   - canonical_name_in_note — выбери наиболее полную и однозначную форму (обычно полное ФИО, если оно есть).
   - is_person — true только если ты уверен, что это человек.
   - confidence — насколько уверенно, что это отдельная историческая персона (0–1).
5. note_year_context:
   1) Сначала ищи год прямо рядом с упоминанием (в том же предложении/абзаце) — используй его и поставь note_year_source = "inline".
   2) Если inline-год не найден, но в метаданных указан основной год/диапазон, и персона явно относится к главному эпизоду — возьми основной год заметки и поставь note_year_source = "note_metadata".
   3) Если даже метаданные не помогают — оставь note_year_context = null и note_year_source = "unknown".
   - inline_year — это явное число в тексте: например, "В 1812 году Кутузов..." (inline_year=1812) или "Корин (1892–1967)..." (inline_year=1892).
   - Если указан диапазон (1812–1814), выбирай наиболее релевантный год (обычно начало или середина эпизода).
6. Роль персоны (role):
   - Если в тексте явно указана профессия, должность или роль — запиши её.
   - Примеры: "художник", "архитектор", "генерал", "купец", "меценат", "император".
   - Если роль не указана — оставь role = null и role_confidence = 0.
   - role_confidence — насколько уверенно роль следует из текста.
7. snippet_evidence:
   - для каждого человека дай несколько (1–3) коротких фрагментов текста, где он упоминается;
   - для каждого фрагмента поясни, почему ты считаешь, что это человек, а не организация/место.

ОЧЕНЬ ВАЖНО:
1) Для КАЖДОГО человека:
   - surface_forms: только УНИКАЛЬНЫЕ формы, НЕ БОЛЕЕ 10 штук.
     Даже если одно написание встретилось десятки раз, в список его
     нужно добавить строго один раз. Если форм много — выбери 5–10
     наиболее различных, остальные опусти.
   - snippet_evidence: не более 3 фрагментов текста. Каждый фрагмент
     короче 300 символов. Не нужно дублировать одинаковый текст.

2) Для ВСЕЙ заметки:
   - people: не более 20 персон. Если людей больше, выбери наиболее
     значимых (по количеству и важности упоминаний).

3) НЕЛЬЗЯ перечислять одно и то же имя десятки раз.
   Список должен быть компактным: 1–3 формы достаточно.

4) НЕЛЬЗЯ повторять один и тот же фрагмент текста в snippet_evidence.


ОЧЕНЬ ВАЖНО:
- НЕЛЬЗЯ придумывать людей, которых нет в тексте.
- НЕЛЬЗЯ расширять имена по внешним знаниям (если в тексте только "Иван", нельзя превращать его в "Иван Иванович Иванов").
- Используй только информацию из заметки и переданных метаданных.
"""


def _build_people_messages(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    ) -> List[Dict[str, Any]]:
    metadata_json = metadata.model_dump()
    metadata_str = json.dumps(metadata_json, ensure_ascii=False, indent=2)

    user_content = (
        "Ниже приведён текст заметки в формате Markdown и её метаданные.\n"
        f"ИД заметки: {note_id}\n\n"
        "Метаданные заметки (JSON):\n"
        f"{metadata_str}\n\n"
        "Извлеки людей согласно инструкции system-промпта и верни результат строго в заданной JSON-схеме.\n\n"
        "Текст заметки:\n"
        f"{text}"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
            },
        {
            "role": "user",
            # "content": add_no_think(user_content),
            "content": user_content,
            },
        ]


async def extract_people_from_text_async(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    validate: bool = True,
    two_pass: bool = True,
) -> PersonExtractionResponse:
    """
    Async-вызов people extractor. По умолчанию использует двухпроходную схему.
    """
    if two_pass:
        return await extract_people_two_pass_async(
            note_id=note_id,
            text=text,
            metadata=metadata,
            validate=validate,
        )

    messages = _build_people_messages(note_id, text, metadata)

    response = await chat_sgr_parse_async(
        messages=messages,
        model_cls=PersonExtractionResponse,
        schema_name="people_extraction",
        temperature=0.0,
        max_tokens=8192,
    )

    if validate:
        response, stats = validate_and_fix_extraction(response, text)
        if stats["removed_forms"] > 0:
            logger.warning(
                "Note %s: removed %s invalid surface forms, %s persons dropped",
                note_id,
                stats["removed_forms"],
                stats["removed_persons"],
            )

    return response


def extract_people_from_text(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    validate: bool = True,
    two_pass: bool = True,
) -> PersonExtractionResponse:
    """
    Синхронная версия people extractor с двухпроходной схемой по умолчанию.
    """
    if two_pass:
        return extract_people_two_pass(
            note_id=note_id,
            text=text,
            metadata=metadata,
            validate=validate,
        )

    messages = _build_people_messages(note_id, text, metadata)

    response = chat_sgr_parse(
        messages=messages,
        model_cls=PersonExtractionResponse,
        schema_name="people_extraction",
        temperature=0.0,
        max_tokens=8192,
    )

    if validate:
        response, stats = validate_and_fix_extraction(response, text)
        if stats["removed_forms"] > 0:
            logger.warning(
                "Note %s: removed %s invalid surface forms, %s persons dropped",
                note_id,
                stats["removed_forms"],
                stats["removed_persons"],
            )

    return response


def extract_people_from_file(path: Path, two_pass: bool = True) -> PersonExtractionResponse:
    """
    Удобный вход: по пути к файлу:
    1) извлекает метаданные заметки,
    2) загружает markdown,
    3) вызывает LLM для извлечения людей.
    """
    # 1. метаданные (наш предыдущий тул)
    meta_resp: NoteMetadataResponse = extract_note_metadata_from_file(path)
    metadata = meta_resp.metadata

    # 2. тело заметки
    post = frontmatter.load(path)
    body = post.content
    note_id = path.stem

    # 3. people extractor
    return extract_people_from_text(
        note_id=note_id,
        text=body,
        metadata=metadata,
        two_pass=two_pass,
    )


async def extract_people_from_file_async(
    path: Path,
    two_pass: bool = True,
) -> PersonExtractionResponse:
    """
    Async-вариант пайплайна для одной заметки.
    """
    meta_resp: NoteMetadataResponse = await extract_note_metadata_from_file_async(path)
    metadata = meta_resp.metadata

    post = await asyncio.to_thread(frontmatter.load, path)
    body = post.content
    note_id = path.stem

    return await extract_people_from_text_async(
        note_id=note_id,
        text=body,
        metadata=metadata,
        two_pass=two_pass,
    )


def _build_people_from_group(
    note_id: str,
    group,
    mention_objs,
    local_person_id: str,
) -> PersonLocal:
    surface_forms: List[str] = []
    for mention in mention_objs:
        if mention.text_span not in surface_forms:
            surface_forms.append(mention.text_span)
        if len(surface_forms) >= 10:
            break

    inline_year = next((m.inline_year for m in mention_objs if m.inline_year is not None), None)
    note_year_source = "inline" if inline_year is not None else "unknown"

    snippet_evidence: List[PersonSnippetEvidence] = []
    for mention in mention_objs[:3]:
        snippet_evidence.append(
            PersonSnippetEvidence(
                snippet=mention.context_snippet,
                reasoning=f"Упоминание {mention.mention_id} выявлено на этапе mention extraction",
            )
        )

    if not surface_forms:
        surface_forms = [group.canonical_name]

    return PersonLocal(
        note_id=note_id,
        local_person_id=local_person_id,
        surface_forms=surface_forms,
        canonical_name_in_note=group.canonical_name,
        is_person=group.is_person,
        confidence=group.confidence,
        note_year_context=inline_year,
        note_year_source=note_year_source,
        snippet_evidence=snippet_evidence,
        role=None,
        role_confidence=0.0,
    )


def _convert_groups_to_extraction_response(
    note_id: str,
    groups_resp: MentionGroupingResponse,
    mentions_resp: MentionExtractionResponse,
) -> PersonExtractionResponse:
    """
    Конвертирует результат группировки в старый формат PersonExtractionResponse.
    """
    mentions_map = {m.mention_id: m for m in mentions_resp.mentions}
    people: List[PersonLocal] = []

    for group in groups_resp.groups:
        mention_objs = [mentions_map[mid] for mid in group.mention_ids if mid in mentions_map]
        if not mention_objs:
            continue
        local_person_id = f"p{len(people) + 1}"
        person = _build_people_from_group(note_id, group, mention_objs, local_person_id)
        people.append(person)

    return PersonExtractionResponse(note_id=note_id, people=people)


def extract_people_two_pass(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    validate: bool = True,
) -> PersonExtractionResponse:
    """
    Синхронный двухпроходный people extractor.
    """
    mentions_resp = extract_mentions(note_id=note_id, text=text, metadata=metadata)
    groups_resp = group_mentions(note_id, mentions_resp.mentions, text)
    response = _convert_groups_to_extraction_response(note_id, groups_resp, mentions_resp)
    log_pipeline_stage(note_id, "people_raw", response.model_dump())

    if validate:
        response, stats = validate_and_fix_extraction(response, text)
        if stats["removed_forms"] > 0:
            logger.warning(
                "Note %s (two-pass sync): removed %s invalid surface forms, %s persons dropped",
                note_id,
                stats["removed_forms"],
                stats["removed_persons"],
            )
        log_pipeline_stage(note_id, "people_final", response.model_dump())
    else:
        log_pipeline_stage(note_id, "people_final", response.model_dump())

    return response


async def extract_people_two_pass_async(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    validate: bool = True,
) -> PersonExtractionResponse:
    """
    Двухпроходное извлечение: mentions -> grouping -> PersonExtractionResponse.
    """
    mentions_resp = await extract_mentions_async(note_id=note_id, text=text, metadata=metadata)
    groups_resp = await group_mentions_async(note_id, mentions_resp.mentions, text)
    response = _convert_groups_to_extraction_response(note_id, groups_resp, mentions_resp)
    log_pipeline_stage(note_id, "people_raw", response.model_dump())

    if validate:
        response, stats = validate_and_fix_extraction(response, text)
        if stats["removed_forms"] > 0:
            logger.warning(
                "Note %s (two-pass): removed %s invalid surface forms, %s persons dropped",
                note_id,
                stats["removed_forms"],
                stats["removed_persons"],
            )
        log_pipeline_stage(note_id, "people_final", response.model_dump())
    else:
        log_pipeline_stage(note_id, "people_final", response.model_dump())

    return response
