from typing import Any, Dict, List
from pathlib import Path
import asyncio

import frontmatter

from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..models import NoteMetadataResponse


SYSTEM_PROMPT = """
Ты помощник, который извлекает метаданные из исторической текстовой заметки в формате Markdown.

Твои задачи:
1. Определить, о каком годе или периоде в основном идёт речь.
2. Если явно указан период (например, 1812–1814), заполни year_start и year_end.
3. Если заметка фокусируется на одном году (например, 1812), укажи primary_year.
4. Если в заметке упоминаются многие годы, но один явно доминирует по смыслу, выбери его как primary_year.
5. Если место действия (город, регион, страна) явно указано и важно для эпизода — заполни поле location.
6. Сформулируй краткий topic (одно–два предложения), строго опираясь только на текст заметки.
7. Если информации мало или она противоречива, оставь соответствующие поля null и понизь reliability.

НЕЛЬЗЯ использовать внешние знания. Все выводы только из данного текста.
"""  # noqa: E501


def _build_metadata_messages(note_id: str, text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Ниже приведён текст заметки в формате Markdown.\n"
                f"ИД заметки: {note_id}\n\n"
                "Определи метаданные согласно инструкции.\n\n"
                f"Текст заметки:\n{text}"
                "/no_think"
            ),
        },
    ]


async def extract_note_metadata_from_text_async(note_id: str, text: str) -> NoteMetadataResponse:
    """
    Вызывает LLM с SGR-схемой и возвращает NoteMetadataResponse.
    """
    messages = _build_metadata_messages(note_id, text)
    return await chat_sgr_parse_async(
        messages=messages,
        model_cls=NoteMetadataResponse,
        schema_name="note_metadata",
        temperature=0.0,
        max_tokens=800,
    )


def extract_note_metadata_from_text(note_id: str, text: str) -> NoteMetadataResponse:
    return chat_sgr_parse(
        messages=_build_metadata_messages(note_id, text),
        model_cls=NoteMetadataResponse,
        schema_name="note_metadata",
        temperature=0.0,
        max_tokens=800,
        )


async def extract_note_metadata_from_file_async(path: Path) -> NoteMetadataResponse:
    """
    Читает MD-файл, вытаскивает body (игнорируя существующий фронтматтер)
    и передаёт в LLM.
    """
    post = await asyncio.to_thread(frontmatter.load, path)
    body = post.content
    note_id = path.stem
    return await extract_note_metadata_from_text_async(note_id=note_id, text=body)


def extract_note_metadata_from_file(path: Path) -> NoteMetadataResponse:
    post = frontmatter.load(path)
    body = post.content
    note_id = path.stem
    return extract_note_metadata_from_text(note_id=note_id, text=body)
