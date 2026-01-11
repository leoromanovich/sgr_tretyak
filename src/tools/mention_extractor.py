from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..models import (
    CandidateList,
    CandidateWithType,
    MentionExtractionResponse,
    NoteMetadata,
    PersonMention,
)
from .pipeline_trace import log_pipeline_stage
from .ner_provider import (
    NERPrediction,
    NERCacheNotFoundError,
    load_ner_from_cache,
    extract_person_names_from_ner,
)

logger = logging.getLogger(__name__)


# --- Утилиты для разбиения текста на chunks ---

def extract_section(text: str, header: str, next_header: Optional[str]) -> str:
    """Извлекает секцию между заголовками ## header и ## next_header."""
    marker = f"## {header}"
    start = text.find(marker)
    if start == -1:
        return ""
    start = start + len(marker)
    end = text.find(f"## {next_header}", start) if next_header else -1
    return text[start:end].strip() if end != -1 else text[start:].strip()


def split_paragraphs(text: str) -> List[str]:
    """Разбивает текст на параграфы по двойным переносам строк."""
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def split_into_chunks(text: str) -> List[str]:
    """
    Разбивает текст заметки на chunks для обработки.
    Возвращает: [meta_section, paragraph1, paragraph2, ...]
    """
    meta_section = extract_section(text, "Метаинформация", "Описание")
    description_section = extract_section(text, "Описание", None)
    description_paragraphs = split_paragraphs(description_section)

    chunks = []
    if meta_section:
        chunks.append(meta_section)
    chunks.extend(description_paragraphs)

    # Если нет структуры с заголовками, обрабатываем весь текст как один chunk
    if not chunks:
        chunks = [text] if text.strip() else []

    return chunks


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


# --- Новый подход: chunked extraction ---

SYSTEM_PROMPT_CANDIDATES = """
Ты извлекаешь ВСЕ имена собственные из текста и определяешь, являются ли они ЛЮДЬМИ.

ВАЖНО:
- Извлекай имена в той форме (падеже), в которой они встречаются в тексте
- Для организаций, в которых есть упоминание человека, извлекай его имя отдельно
- Отмечай is_person=true ТОЛЬКО для реальных людей (имя, фамилия, инициалы)
- Отмечай is_person=false для:
  * названий мест (города, страны, улицы)
  * названий организаций, трупп, компаний
  * названий произведений искусства (фильмы, балеты, картины)
  * географических объектов

Примеры:
- 'Вацлав Нижинский' → is_person=true
- 'Нью-Йорк' → is_person=false (город)
- 'Русского балета' → is_person=false (название труппы)
- 'Тихая улица' → is_person=false (название фильма)
- 'комбината имени Е.В. Вучетича' даёт два имени:
  1) 'Е.В. Вучетича' → is_person=true
  2) 'комбината имени Е.В. Вучетича' → is_person=false (организация)
"""


def _build_candidate_messages(chunk: str) -> List[Dict[str, Any]]:
    """Строит сообщения для извлечения кандидатов из одного chunk."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_CANDIDATES,
        },
        {
            "role": "user",
            "content": chunk,
        },
    ]


async def extract_candidates_from_chunk_async(chunk: str) -> List[CandidateWithType]:
    """Извлекает все имена собственные из одного chunk (async)."""
    if not chunk.strip():
        return []

    messages = _build_candidate_messages(chunk)
    response = await chat_sgr_parse_async(
        messages=messages,
        model_cls=CandidateList,
        schema_name="candidate_list",
        temperature=0.0,
        max_tokens=4096,
    )
    return response.candidates


def extract_candidates_from_chunk(chunk: str) -> List[CandidateWithType]:
    """Извлекает все имена собственные из одного chunk (sync)."""
    if not chunk.strip():
        return []

    messages = _build_candidate_messages(chunk)
    response = chat_sgr_parse(
        messages=messages,
        model_cls=CandidateList,
        schema_name="candidate_list",
        temperature=0.0,
        max_tokens=4096,
    )
    return response.candidates


def _candidates_to_mentions(
    candidates: Dict[str, CandidateWithType],
    note_id: str,
) -> List[PersonMention]:
    """
    Конвертирует дедуплицированных кандидатов в PersonMention.
    """
    mentions: List[PersonMention] = []
    for idx, candidate in enumerate(candidates.values()):
        mention = PersonMention(
            mention_id=f"m{idx + 1}",
            text_span=candidate.name,
            context_snippet=f"Извлечено из chunked extraction",
            likely_person=candidate.is_person,
            inline_year=None,
        )
        mentions.append(mention)
    return mentions


def _merge_ner_candidates(
    llm_candidates: Dict[str, CandidateWithType],
    ner_predictions: List[NERPrediction],
) -> Dict[str, CandidateWithType]:
    """
    Добавляет NER-кандидатов к LLM-кандидатам.

    NER-кандидаты добавляются только если их ещё нет в списке LLM.
    NER даёт is_person=True для FIRST_NAME, LAST_NAME, MIDDLE_NAME.

    Args:
        llm_candidates: Кандидаты от LLM
        ner_predictions: Предсказания NER модели

    Returns:
        Объединённый словарь кандидатов
    """
    merged = dict(llm_candidates)
    ner_names = extract_person_names_from_ner(ner_predictions)

    added_from_ner = 0
    for name in ner_names:
        if name not in merged:
            merged[name] = CandidateWithType(name=name, is_person=True)
            added_from_ner += 1

    if added_from_ner > 0:
        logger.debug("Added %d candidates from NER (total: %d)", added_from_ner, len(merged))

    return merged


async def extract_mentions_chunked_async(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    ner_predictions: Optional[List[NERPrediction]] = None,
) -> MentionExtractionResponse:
    """
    Извлечение упоминаний по chunks (новый подход).
    Разбивает текст на meta_section + параграфы и обрабатывает каждый отдельно.

    Args:
        note_id: ID заметки
        text: Текст заметки
        metadata: Метаданные заметки
        ner_predictions: Предсказания NER модели (если None, загружаются из кэша)

    Raises:
        NERCacheNotFoundError: Если NER кэш не найден
    """
    # Загружаем NER из кэша, если не переданы
    if ner_predictions is None:
        ner_predictions = load_ner_from_cache(note_id)

    chunks = split_into_chunks(text)
    logger.debug(
        "Extracting mentions chunked for note_id=%s (chunks=%d, ner=%d)",
        note_id,
        len(chunks),
        len(ner_predictions),
    )

    # Собираем кандидатов из всех chunks с дедупликацией по имени
    all_candidates: Dict[str, CandidateWithType] = {}
    for chunk in chunks:
        chunk_candidates = await extract_candidates_from_chunk_async(chunk)
        for candidate in chunk_candidates:
            if candidate.name not in all_candidates:
                all_candidates[candidate.name] = candidate

    # Мёржим с NER кандидатами
    all_candidates = _merge_ner_candidates(all_candidates, ner_predictions)

    # Конвертируем в mentions
    mentions = _candidates_to_mentions(all_candidates, note_id)

    response = MentionExtractionResponse(note_id=note_id, mentions=mentions)
    log_pipeline_stage(note_id, "mentions_chunked", response.model_dump())
    return response


def extract_mentions_chunked(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    ner_predictions: Optional[List[NERPrediction]] = None,
) -> MentionExtractionResponse:
    """
    Синхронная версия chunked extraction.

    Args:
        note_id: ID заметки
        text: Текст заметки
        metadata: Метаданные заметки
        ner_predictions: Предсказания NER модели (если None, загружаются из кэша)

    Raises:
        NERCacheNotFoundError: Если NER кэш не найден
    """
    # Загружаем NER из кэша, если не переданы
    if ner_predictions is None:
        ner_predictions = load_ner_from_cache(note_id)

    chunks = split_into_chunks(text)
    logger.debug(
        "Extracting mentions chunked for note_id=%s (chunks=%d, ner=%d) [sync]",
        note_id,
        len(chunks),
        len(ner_predictions),
    )

    all_candidates: Dict[str, CandidateWithType] = {}
    for chunk in chunks:
        chunk_candidates = extract_candidates_from_chunk(chunk)
        for candidate in chunk_candidates:
            if candidate.name not in all_candidates:
                all_candidates[candidate.name] = candidate

    # Мёржим с NER кандидатами
    all_candidates = _merge_ner_candidates(all_candidates, ner_predictions)

    mentions = _candidates_to_mentions(all_candidates, note_id)

    response = MentionExtractionResponse(note_id=note_id, mentions=mentions)
    log_pipeline_stage(note_id, "mentions_chunked", response.model_dump())
    return response
