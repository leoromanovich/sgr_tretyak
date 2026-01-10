from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
import json
import re

import frontmatter

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..llm_prompts import add_no_think
from ..models import (
    NoteMetadata,
    NoteMetadataResponse,
    PersonExtractionResponse,
    PersonLocalNormalized,
    PersonNormalizationResponse,
    )
from .note_metadata import (
    extract_note_metadata_from_file,
    extract_note_metadata_from_file_async,
    validate_metadata,
)
from .people_extractor import (
    extract_people_from_text,
    extract_people_from_text_async,
    )


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
   - ВАЖНО: Все имена (normalized_full_name, name_parts) должны быть в ИМЕНИТЕЛЬНОМ падеже.
   - Если в тексте встречается только "Тихона" (родительный) — нормализуй до "Тихон".
"""

TITLE_TOKENS = {
    "граф",
    "князь",
    "княгиня",
    "княжна",
    "барон",
    "баронесса",
    "господин",
    "госпожа",
    "г-н",
    "г-жа",
    "гн",
    "гжа",
    "генерал",
    "полковник",
    "подполковник",
    "майор",
    "капитан",
    "поручик",
    "штабс",
    "штабскапитан",
    "адъютант",
    "профессор",
    "доктор",
    "академик",
    "священник",
    "протоиерей",
    "архимандрит",
    "диакон",
    "епископ",
    "архиепископ",
    "митрополит",
    "патриарх",
    }

REFORM_TRANSLATION = str.maketrans({
    "ё": "е",
    "ѣ": "е",
    "і": "и",
    "ѳ": "ф",
    "ѵ": "и",
    "ѫ": "у",
    "ѧ": "я",
    })

PUNCTUATION_RE = re.compile(r"[^\w\s-]+", re.UNICODE)
NON_ALNUM_RE = re.compile(r"[^a-zа-я0-9]+", re.IGNORECASE)


def normalize_name_tokens(value: Optional[str], remove_titles: bool = True) -> List[str]:
    if not value:
        return []
    lowered = value.strip().lower().translate(REFORM_TRANSLATION)
    lowered = PUNCTUATION_RE.sub(" ", lowered)
    raw_tokens = re.split(r"\s+", lowered)
    tokens: List[str] = []
    for raw in raw_tokens:
        cleaned = NON_ALNUM_RE.sub("", raw)
        if not cleaned:
            continue
        if remove_titles and cleaned in TITLE_TOKENS:
            continue
        tokens.append(cleaned)
    return tokens


def normalize_name_for_blocking(value: Optional[str], remove_titles: bool = True) -> Optional[str]:
    tokens = normalize_name_tokens(value, remove_titles=remove_titles)
    if not tokens:
        return None
    return " ".join(tokens)


def infer_name_parts_from_full_name(full_name: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    tokens = normalize_name_tokens(full_name)
    if not tokens:
        return None, None, None
    if len(tokens) >= 3:
        first_name = tokens[0]
        patronymic = tokens[1]
        last_name = tokens[-1]
    elif len(tokens) == 2:
        first_name = tokens[0]
        patronymic = None
        last_name = tokens[1]
    else:
        first_name = None
        patronymic = None
        last_name = tokens[0]
    return last_name, first_name, patronymic


def initial_from_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value[0]


def enrich_person_for_blocking(person: PersonLocalNormalized) -> None:
    inferred_last = inferred_first = inferred_patronymic = None
    if person.normalized_full_name:
        inferred_last, inferred_first, inferred_patronymic = infer_name_parts_from_full_name(
            person.normalized_full_name
            )
    elif person.canonical_name_in_note:
        inferred_last, inferred_first, inferred_patronymic = infer_name_parts_from_full_name(
            person.canonical_name_in_note
            )

    last_source = person.name_parts.last_name or inferred_last
    first_source = person.name_parts.first_name or inferred_first
    patronymic_source = person.name_parts.patronymic or inferred_patronymic

    normalized_last = normalize_name_for_blocking(last_source)
    normalized_first = normalize_name_for_blocking(first_source, remove_titles=False)
    normalized_patronymic = normalize_name_for_blocking(patronymic_source, remove_titles=False)

    person.normalized_last_name = normalized_last
    person.first_initial = initial_from_name(normalized_first)
    person.patronymic_initial = initial_from_name(normalized_patronymic)


def _build_normalizer_messages(
    note_id: str,
    text: str,
    people_extraction: PersonExtractionResponse,
    metadata: NoteMetadata,
    ) -> List[Dict[str, Any]]:
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
    response = await chat_sgr_parse_async(
        messages=messages,
        model_cls=PersonNormalizationResponse,
        schema_name="name_normalizer",
        temperature=0.0,
        max_tokens=None,
        )
    for person in response.people:
        enrich_person_for_blocking(person)
    return response


def normalize_people_in_text(
    note_id: str,
    text: str,
    people_extraction: PersonExtractionResponse,
    metadata: NoteMetadata,
    ) -> PersonNormalizationResponse:
    messages = _build_normalizer_messages(note_id, text, people_extraction, metadata)
    response = chat_sgr_parse(
        messages=messages,
        model_cls=PersonNormalizationResponse,
        schema_name="name_normalizer",
        temperature=0.0,
        max_tokens=None,
        )
    for person in response.people:
        enrich_person_for_blocking(person)
    return response


def _normalize_pipeline(
    note_id: str,
    body: str,
    metadata: NoteMetadata,
    ) -> Tuple[PersonNormalizationResponse, List[str]]:
    metadata, warnings = validate_metadata(metadata, body)
    if metadata.reliability and metadata.reliability < 0.5:
        warnings.append(f"Low metadata reliability: {metadata.reliability}")

    people_extraction = extract_people_from_text(note_id=note_id, text=body, metadata=metadata)
    response = normalize_people_in_text(
        note_id=note_id,
        text=body,
        people_extraction=people_extraction,
        metadata=metadata,
        )
    return response, warnings


async def _normalize_pipeline_async(
    note_id: str,
    body: str,
    metadata: NoteMetadata,
    ) -> Tuple[PersonNormalizationResponse, List[str]]:
    metadata, warnings = validate_metadata(metadata, body)
    if metadata.reliability and metadata.reliability < 0.5:
        warnings.append(f"Low metadata reliability: {metadata.reliability}")

    people_extraction = await extract_people_from_text_async(
        note_id=note_id,
        text=body,
        metadata=metadata,
        )
    response = await normalize_people_in_text_async(
        note_id=note_id,
        text=body,
        people_extraction=people_extraction,
        metadata=metadata,
        )
    return response, warnings


def normalize_people_in_file(path: Path) -> PersonNormalizationResponse:
    """
    Высокоуровневая функция:
    1) метаданные заметки,
    2) первичный people_extractor,
    3) нормализация имён.
    """
    meta_resp: NoteMetadataResponse = extract_note_metadata_from_file(path)
    metadata = meta_resp.metadata

    post = frontmatter.load(path)
    body = post.content
    note_id = path.stem

    response, _ = _normalize_pipeline(note_id=note_id, body=body, metadata=metadata)
    return response


def normalize_people_in_file_with_warnings(path: Path) -> Tuple[PersonNormalizationResponse, List[str]]:
    meta_resp: NoteMetadataResponse = extract_note_metadata_from_file(path)
    metadata = meta_resp.metadata

    post = frontmatter.load(path)
    body = post.content
    note_id = path.stem

    return _normalize_pipeline(note_id=note_id, body=body, metadata=metadata)


async def normalize_people_in_file_async(path: Path) -> PersonNormalizationResponse:
    """
    Async версия пайплайна для одной заметки.
    """
    meta_resp: NoteMetadataResponse = await extract_note_metadata_from_file_async(path)
    metadata = meta_resp.metadata

    post = await asyncio.to_thread(frontmatter.load, path)
    body = post.content
    note_id = path.stem

    response, _ = await _normalize_pipeline_async(note_id=note_id, body=body, metadata=metadata)
    return response


async def normalize_people_in_file_with_warnings_async(
    path: Path,
    ) -> Tuple[PersonNormalizationResponse, List[str]]:
    meta_resp: NoteMetadataResponse = await extract_note_metadata_from_file_async(path)
    metadata = meta_resp.metadata

    post = await asyncio.to_thread(frontmatter.load, path)
    body = post.content
    note_id = path.stem

    return await _normalize_pipeline_async(note_id=note_id, body=body, metadata=metadata)
