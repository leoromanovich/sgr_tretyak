from typing import Dict, Any
from pathlib import Path
import json

import frontmatter

from ..llm_client import chat_sgr_parse
from ..models import (
    NoteMetadata,
    NoteMetadataResponse,
    PersonExtractionResponse,
    )
from .note_metadata import extract_note_metadata_from_file
from ..llm_prompts import add_no_think


# JSON Schema для structured output (должна соответствовать PersonExtractionResponse)
PEOPLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "note_id": {"type": "string"},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "local_person_id": {"type": "string"},
                    "surface_forms": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 100},
                        },
                    "canonical_name_in_note": {"type": "string"},
                    "is_person": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        },
                    "note_year_context": {
                        "type": ["integer", "null"],
                        },
                    "note_year_source": {
                        "type": "string",
                        "enum": ["inline", "note_metadata", "unknown"],
                        },
                    "snippet_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "snippet": {"type": "string", "maxLength": 300},
                                "reasoning": {"type": "string", "maxLength": 300},
                                },
                            "required": ["snippet", "reasoning"],
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
                    ],
                "additionalProperties": False,
                },
            },
        },
    "required": ["note_id", "people"],
    "additionalProperties": False,
    }


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
   - surface_forms — перечисли все разные формы имени, встречающиеся в тексте (как минимум 1).
   - canonical_name_in_note — выбери наиболее полную и однозначную форму (обычно полное ФИО, если оно есть).
   - is_person — true только если ты уверен, что это человек.
   - confidence — насколько уверенно, что это отдельная историческая персона (0–1).
5. note_year_context:
   - если рядом с упоминанием есть конкретный год — используй его и поставь note_year_source = "inline";
   - иначе, если из метаданных заметки дан период или primary_year и СМЫСЛОВО этот человек относится к основному эпизоду — используй основной год заметки (например, середину диапазона) и note_year_source = "note_metadata";
   - если нельзя уверенно привязать к году — оставь note_year_context = null и note_year_source = "unknown".
6. snippet_evidence:
   - для каждого человека дай несколько (1–3) коротких фрагментов текста, где он упоминается;
   - для каждого фрагмента поясни, почему ты считаешь, что это человек, а не организация/место.

ОЧЕНЬ ВАЖНО:
1) Для КАЖДОГО человека:
   - surface_forms: только УНИКАЛЬНЫЕ формы, НЕ БОЛЕЕ 5 штук.
     Если в тексте одно и то же написание встречается много раз,
     ты добавляешь его В СПИСОК ТОЛЬКО ОДИН РАЗ.
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


def extract_people_from_text(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    ) -> PersonExtractionResponse:
    """
    Вызывает LLM для извлечения людей из текста заметки с учётом NoteMetadata.
    """
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

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
            },
        {
            "role": "user",
            "content": add_no_think(user_content),
            },
        ]

    return chat_sgr_parse(
        messages=messages,
        schema_name="people_extraction",
        schema=PEOPLE_SCHEMA,
        model_cls=PersonExtractionResponse,
        temperature=0.0,
        max_tokens=8192,
        )


def extract_people_from_file(path: Path) -> PersonExtractionResponse:
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
    return extract_people_from_text(note_id=note_id, text=body, metadata=metadata)