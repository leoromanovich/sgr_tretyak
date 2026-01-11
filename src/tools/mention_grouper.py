from __future__ import annotations

from typing import Any, Dict, List
import json
import logging

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..models import MentionGroupingResponse, PersonMention
from .pipeline_trace import log_pipeline_stage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_GROUPING = """
Ты группируешь упоминания людей, которые относятся к одному человеку.

На входе — список mentions из одной заметки.
На выходе — группы, где каждая группа = один УНИКАЛЬНЫЙ человек.

КРИТИЧЕСКИ ВАЖНО:
- Каждая группа = ОДИН конкретный человек!
- РАЗНЫЕ люди = РАЗНЫЕ группы, даже если они упоминаются рядом!
- НЕ объединяй людей только потому, что они упоминаются в одном контексте!
- НЕ ПОВТОРЯЙ одни и те же группы с разными group_id!
- Каждый mention_id должен встречаться ТОЛЬКО В ОДНОЙ группе!

ПРАВИЛА ГРУППИРОВКИ:
1. ОБЪЕДИНЯЙ только если это ТОЧНО один и тот же человек:
   - "И.И. Иванов" и "Иван Иванович Иванов" → одна группа (совпадают инициалы)
   - "Нижинский" и "Вацлав Нижинский" → одна группа (одна фамилия)
   - "Чаплин" и "Чарли Чаплин" → одна группа (одна фамилия)

2. НЕ ОБЪЕДИНЯЙ разных людей:
   - "Нижинский" и "Чаплин" → РАЗНЫЕ группы (разные фамилии!)
   - "М.Ф. Ларионова" и "Э. Пёрвиенс" → РАЗНЫЕ группы (разные люди!)
   - Люди в перечислении "А, Б, В и др." → каждый в СВОЕЙ группе!

3. Признаки РАЗНЫХ людей:
   - Разные фамилии
   - Разные инициалы при одинаковой фамилии
   - Разные имена

ПРИМЕРЫ:
- Mentions: ["Вацлав Нижинский", "Нижинского", "Чарли Чаплин", "Чаплина"]
  → Группа 1: Вацлав Нижинский (mentions: Вацлав Нижинский, Нижинского)
  → Группа 2: Чарли Чаплин (mentions: Чарли Чаплин, Чаплина)

- Mentions: ["Э. Пёрвиенс", "Э. Кэмпбелла", "Ш. Мино", "Л. Лопуховой"]
  → 4 РАЗНЫХ группы (разные фамилии = разные люди)!

canonical_name — самая полная форма, ПРИВЕДЁННАЯ К ИМЕНИТЕЛЬНОМУ ПАДЕЖУ.
confidence — уверенность, что все mentions в группе = один человек (1.0 для очевидных случаев).
"""


def _build_grouping_messages(
    note_id: str,
    mentions: List[PersonMention],
    text: str,
) -> List[Dict[str, Any]]:
    mentions_payload = [m.model_dump() for m in mentions]
    mentions_str = json.dumps(mentions_payload, ensure_ascii=False, indent=2)

    user_content = (
        "Ниже приведён текст заметки и список упоминаний (mentions) в формате JSON.\n"
        f"note_id: {note_id}\n\n"
        "mentions:\n"
        f"{mentions_str}\n\n"
        "Сгруппируй mentions в людей согласно правилам system-промпта. "
        "Верни JSON в формате MentionGroupingResponse. "
        "Если mention явно не про человека, всё равно оставь его в отдельной группе "
        "с is_person=false.\n\n"
        "Текст заметки:\n"
        f"{text}"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_GROUPING,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


async def group_mentions_async(
    note_id: str,
    mentions: List[PersonMention],
    text: str,
) -> MentionGroupingResponse:
    """
    Второй проход пайплайна: группировка mentions в людей.
    """
    if not mentions:
        empty_resp = MentionGroupingResponse(note_id=note_id, groups=[])
        log_pipeline_stage(note_id, "grouping", empty_resp.model_dump())
        return empty_resp

    messages = _build_grouping_messages(note_id, mentions, text)
    logger.debug(
        "Grouping mentions for note_id=%s (mentions=%d)", note_id, len(mentions)
    )

    response = await chat_sgr_parse_async(
        messages=messages,
        model_cls=MentionGroupingResponse,
        schema_name="mention_grouping",
        temperature=0.2,  # Выше 0 для избежания петель
        max_tokens=8192,
        frequency_penalty=0.5,  # Штрафуем повторения
    )

    if not response.note_id:
        response.note_id = note_id

    log_pipeline_stage(note_id, "grouping", response.model_dump())
    return response


def group_mentions(
    note_id: str,
    mentions: List[PersonMention],
    text: str,
) -> MentionGroupingResponse:
    """
    Синхронный вариант группировки mentions в людей.
    """
    if not mentions:
        empty_resp = MentionGroupingResponse(note_id=note_id, groups=[])
        log_pipeline_stage(note_id, "grouping", empty_resp.model_dump())
        return empty_resp

    messages = _build_grouping_messages(note_id, mentions, text)
    logger.debug(
        "Grouping mentions for note_id=%s (mentions=%d) [sync]",
        note_id,
        len(mentions),
    )

    response = chat_sgr_parse(
        messages=messages,
        model_cls=MentionGroupingResponse,
        schema_name="mention_grouping",
        temperature=0.2,  # Выше 0 для избежания петель
        max_tokens=8192,
        frequency_penalty=0.5,  # Штрафуем повторения
    )

    if not response.note_id:
        response.note_id = note_id

    log_pipeline_stage(note_id, "grouping", response.model_dump())
    return response
