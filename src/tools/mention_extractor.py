from __future__ import annotations

from typing import Any, Dict, List
import json
import logging

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..models import MentionExtractionResponse, NoteMetadata
from .pipeline_trace import log_pipeline_stage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_MENTIONS = """
Ты извлекаешь упоминания людей из исторического текста.

КРИТИЧЕСКИ ВАЖНО:
- Максимум 50 упоминаний на весь текст
- КАЖДОЕ упоминание должно иметь УНИКАЛЬНЫЙ context_snippet
- Если одно имя встречается много раз, выбери 2-3 РАЗНЫХ контекста

Для каждого упоминания:
1. text_span — ТОЧНАЯ строка из текста, БУКВА В БУКВУ как написано (включая падеж!)
2. context_snippet — 1-2 предложения вокруг (УНИКАЛЬНЫЙ для каждого mention!)
3. likely_person — false только если это ТОЧНО организация/место
4. inline_year — если рядом с именем есть год, укажи его

ПРАВИЛА ИЗВЛЕЧЕНИЯ text_span:
- КОПИРУЙ text_span ТОЧНО как в тексте, включая падежные окончания!
- Если в тексте "патриарха Тихона" — пиши "патриарха Тихона", НЕ "патриарх Тихон"
- Если в тексте "Павла Корина" — пиши "Павла Корина", НЕ "Павел Корин"
- Если в тексте "М.Ф. Ларионова" — пиши "М.Ф. Ларионова", НЕ "М.Ф. Ларионов"
- НЕ НОРМАЛИЗУЙ падежи! Сохраняй ТОЧНО как в оригинале!

ПРАВИЛА ПО УПОМИНАНИЯМ:
- Разные формы одного имени ("И.И. Иванов" и "Иван Иванович") = РАЗНЫЕ mentions
- Одна и та же форма в РАЗНЫХ контекстах = РАЗНЫЕ mentions (но максимум 3 на форму)
- Одна и та же форма в ОДНОМ контексте = ОДИН mention (НЕ дублировать!)

Включай:
- Полные ФИО
- Сокращённые формы (И.И. Иванов)
- Только фамилии, если из контекста ясно, что это человек
- Прозвища и псевдонимы
- Имена людей в названиях организаций ("комбинат имени Е.В. Вучетича" → извлеки "Е.В. Вучетича")
- Имена людей, в честь которых что-то названо
- Королей, царей, императоров ("Альфонсо XIII", "Николай II", "Людовик XIV")
- Людей с титулами ("патриарха Тихона", "князя Голицына")
- Людей из метаданных (поле "Происхождение": "от А.К. Томилиной" → извлеки "А.К. Томилиной")
- Людей упомянутых как режиссёры, антрепренёры, артисты и т.д. ("Григорьев, режиссер")

НЕ включай:
- Абстрактные группы ("крестьяне", "монахи")
- Сами названия организаций (но людей из них — включай!)
- Географические названия

ПОМНИ: максимум 50 упоминаний, каждое с уникальным контекстом!
"""

def _build_mention_messages(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
) -> List[Dict[str, Any]]:
    metadata_json = metadata.model_dump()
    metadata_str = json.dumps(metadata_json, ensure_ascii=False, indent=2)

    user_content = (
        "Тебе передана историческая заметка в формате Markdown вместе с метаданными.\n"
        f"note_id: {note_id}\n\n"
        "Метаданные заметки (JSON):\n"
        f"{metadata_str}\n\n"
        "Извлеки ВСЕ возможные упоминания людей согласно правилам system-промпта "
        "и верни результат строго в JSON-схеме модели MentionExtractionResponse.\n\n"
        "Текст заметки:\n"
        f"{text}"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_MENTIONS,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


async def extract_mentions_async(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
) -> MentionExtractionResponse:
    """
    Первый проход пайплайна: извлечение всех потенциальных упоминаний людей.
    """
    messages = _build_mention_messages(note_id, text, metadata)
    logger.debug("Extracting mentions for note_id=%s", note_id)

    response = await chat_sgr_parse_async(
        messages=messages,
        model_cls=MentionExtractionResponse,
        schema_name="mention_extraction",
        temperature=0.0,
        max_tokens=8192,
    )

    # Гарантируем, что note_id всегда указан
    if not response.note_id:
        response.note_id = note_id

    log_pipeline_stage(note_id, "mentions", response.model_dump())
    return response


def extract_mentions(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
) -> MentionExtractionResponse:
    """
    Синхронный вариант извлечения упоминаний.
    """
    messages = _build_mention_messages(note_id, text, metadata)
    response = chat_sgr_parse(
        messages=messages,
        model_cls=MentionExtractionResponse,
        schema_name="mention_extraction",
        temperature=0.0,
        max_tokens=8192,
    )
    if not response.note_id:
        response.note_id = note_id
    log_pipeline_stage(note_id, "mentions", response.model_dump())
    return response
