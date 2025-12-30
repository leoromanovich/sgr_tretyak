from typing import Dict, Any
from pathlib import Path
import asyncio
import json

import frontmatter

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..llm_prompts import add_no_think
from ..models import (
    NoteMetadata,
    NoteMetadataResponse,
    PersonExtractionResponse,
    PersonNormalizationResponse,
    )
from .note_metadata import extract_note_metadata_from_file, extract_note_metadata_from_file_async
from .people_extractor import (
    extract_people_from_text,
    extract_people_from_text_async,
    )


# JSON Schema для нормализации имён
NAME_NORMALIZER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "note_id": {"type": "string"},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # поля из PersonLocal
                    "note_id": {"type": "string"},
                    "local_person_id": {"type": "string"},
                    "surface_forms": {"type": "array", "items": {"type": "string"}},
                    "canonical_name_in_note": {"type": "string"},
                    "is_person": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        },
                    "note_year_context": {"type": ["integer", "null"]},
                    "note_year_source": {
                        "type": "string",
                        "enum": ["inline", "note_metadata", "unknown"],
                        },
                    "snippet_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "snippet": {"type": "string"},
                                "reasoning": {"type": "string"},
                                },
                            "required": ["snippet", "reasoning"],
                            "additionalProperties": False,
                            },
                        },
                    # новые поля
                    "normalized_full_name": {"type": ["string", "null"]},
                    "name_parts": {
                        "type": "object",
                        "properties": {
                            "last_name": {"type": ["string", "null"]},
                            "first_name": {"type": ["string", "null"]},
                            "patronymic": {"type": ["string", "null"]},
                            },
                        "required": ["last_name", "first_name", "patronymic"],
                        "additionalProperties": False,
                        },
                    "abbreviation_links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from_form": {"type": "string"},
                                "to_form": {"type": "string"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    },
                                "reasoning": {"type": "string"},
                                },
                            "required": ["from_form", "to_form", "confidence", "reasoning"],
                            "additionalProperties": False,
                            },
                        },
                    },
                "required": [
                    "note_id",
                    "local_person_id",
                    "surface_forms",
                    "canonical_name_in_note",
                    "is_person",
                    "confidence",
                    "note_year_context",
                    "note_year_source",
                    "snippet_evidence",
                    "normalized_full_name",
                    "name_parts",
                    "abbreviation_links",
                    ],
                "additionalProperties": False,
                },
            },
        },
    "required": ["note_id", "people"],
    "additionalProperties": False,
    }


SYSTEM_PROMPT = """
Ты помощник, который нормализует имена людей в историческом тексте на русском языке.

На вход ты получаешь:
- текст заметки (Markdown),
- список уже извлечённых людей с их surface_forms и canonical_name_in_note.

Твои задачи:

1. Для каждого человека попытаться восстановить максимально полное ФИО по контексту этой заметки.
   - Например, из комбинации "П.Д. Корин", "Павел Корин", "Корин Павел" сделать "Павел Дмитриевич Корин", если все части явно встречаются в тексте.
   - ЕСЛИ отчество или имя нигде не встречаются, НЕ придумывай их.

2. Заполнить normalized_full_name:
   - Если есть полное ФИО в явном виде — используй его.
   - Если есть только фамилия и имя — используй "Имя Фамилия".
   - Если есть только фамилия или только имя — используй то, что есть.
   - Если информации недостаточно даже для этого — оставь null.

3. Заполнить name_parts:
   - last_name, first_name, patronymic — только если можно уверенно выделить из текста.
   - Если чего-то нет в тексте — оставь null.

4. Сопоставить сокращения и полные формы:
   - Если "П.Д. Корин" и "Павел Дмитриевич Корин" явно относятся к одному человеку, добавь запись в abbreviation_links:
     - from_form: "П.Д. Корин"
     - to_form: "Павел Дмитриевич Корин"
     - confidence: от 0 до 1
     - reasoning: краткое объяснение (совпадение фамилии, инициалов и отсутствие альтернатив).

5. Важно:
   - НЕ использовать внешние знания (никаких "известных художников").
   - Работать только с тем, что реально есть в тексте.
   - Если есть сомнения — лучше оставить части имени null и повысить прозрачность reasoning в abbreviation_links.
"""


def _build_normalizer_messages(
    note_id: str,
    text: str,
    people_extraction: PersonExtractionResponse,
    metadata: NoteMetadata,
    ) -> list[Dict[str, Any]]:
    metadata_str = json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2)
    people_str = json.dumps(people_extraction.model_dump(), ensure_ascii=False, indent=2)

    user_content = (
        "Ниже приведён текст заметки в формате Markdown, её метаданные "
        "и результат первичного извлечения людей (people_extraction).\n"
        f"ИД заметки: {note_id}\n\n"
        "Метаданные заметки (JSON):\n"
        f"{metadata_str}\n\n"
        "Первичное извлечение людей (JSON):\n"
        f"{people_str}\n\n"
        "Нормализуй имена людей согласно инструкции system-промпта и верни результат строго в заданной JSON-схеме.\n\n"
        "Текст заметки:\n"
        f"{text}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": add_no_think(user_content)},
        ]

async def normalize_people_in_text_async(
    note_id: str,
    text: str,
    people_extraction: PersonExtractionResponse,
    metadata: NoteMetadata,
    ) -> PersonNormalizationResponse:
    messages = _build_normalizer_messages(note_id, text, people_extraction, metadata)
    return await chat_sgr_parse_async(
        messages=messages,
        schema_name="name_normalizer",
        schema=NAME_NORMALIZER_SCHEMA,
        model_cls=PersonNormalizationResponse,
        temperature=0.0,
        max_tokens=None,
        )


def normalize_people_in_text(
    note_id: str,
    text: str,
    people_extraction: PersonExtractionResponse,
    metadata: NoteMetadata,
    ) -> PersonNormalizationResponse:
    messages = _build_normalizer_messages(note_id, text, people_extraction, metadata)
    return chat_sgr_parse(
        messages=messages,
        schema_name="name_normalizer",
        schema=NAME_NORMALIZER_SCHEMA,
        model_cls=PersonNormalizationResponse,
        temperature=0.0,
        max_tokens=None,
        )


def normalize_people_in_file(path: Path) -> PersonNormalizationResponse:
    """
    Высокоуровневая функция:
    1) метаданные заметки,
    2) первичный people_extractor,
    3) нормализация имён.
    """
    # 1. метаданные
    meta_resp: NoteMetadataResponse = extract_note_metadata_from_file(path)
    metadata = meta_resp.metadata

    # 2. тело заметки
    post = frontmatter.load(path)
    body = post.content
    note_id = path.stem

    # 3. первичное извлечение людей
    people_extraction = extract_people_from_text(note_id=note_id, text=body, metadata=metadata)

    # 4. нормализация
    return normalize_people_in_text(
        note_id=note_id,
        text=body,
        people_extraction=people_extraction,
        metadata=metadata,
        )


async def normalize_people_in_file_async(path: Path) -> PersonNormalizationResponse:
    """
    Async версия пайплайна для одной заметки.
    """
    meta_resp: NoteMetadataResponse = await extract_note_metadata_from_file_async(path)
    metadata = meta_resp.metadata

    post = await asyncio.to_thread(frontmatter.load, path)
    body = post.content
    note_id = path.stem

    people_extraction = await extract_people_from_text_async(
        note_id=note_id,
        text=body,
        metadata=metadata,
        )

    return await normalize_people_in_text_async(
        note_id=note_id,
        text=body,
        people_extraction=people_extraction,
        metadata=metadata,
        )
