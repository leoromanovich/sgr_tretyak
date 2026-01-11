"""
NER Provider: загрузка предсказаний из кэша и запуск модели.

Модель: Gherman/bert-base-NER-Russian
"""

from __future__ import annotations

import json
import logging
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..config import settings

logger = logging.getLogger(__name__)

# Типы сущностей, относящиеся к людям
PERSON_ENTITY_TYPES = frozenset({"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"})


@dataclass
class NERPrediction:
    """Одно предсказание NER модели."""
    word: str
    entity: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None

    @property
    def is_person_entity(self) -> bool:
        """Является ли сущность частью имени человека."""
        return self.entity in PERSON_ENTITY_TYPES


class NERCacheNotFoundError(Exception):
    """NER кэш не найден для заметки."""
    pass


def load_ner_from_cache(
    note_id: str,
    cache_dir: Optional[Path] = None,
) -> List[NERPrediction]:
    """
    Загружает NER предсказания из кэша.

    Args:
        note_id: ID заметки (имя файла без расширения)
        cache_dir: Директория с кэшем (по умолчанию settings.ner_cache_dir)

    Returns:
        Список NERPrediction

    Raises:
        NERCacheNotFoundError: Если кэш не найден
    """
    if cache_dir is None:
        cache_dir = settings.ner_cache_dir

    cache_path = cache_dir / f"{note_id}.json"
    if not cache_path.exists():
        raise NERCacheNotFoundError(
            f"NER cache not found for note_id={note_id} at {cache_path}. "
            f"Run 'ner-extract' command first."
        )

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        predictions = []
        for entry in data.values():
            predictions.append(NERPrediction(
                word=entry["word"],
                entity=entry["entity"],
                score=entry.get("score", 0.0) or 0.0,
                start=entry.get("start"),
                end=entry.get("end"),
            ))
        logger.debug("Loaded %d NER predictions from cache for note_id=%s", len(predictions), note_id)
        return predictions
    except (json.JSONDecodeError, KeyError) as e:
        raise NERCacheNotFoundError(
            f"Failed to parse NER cache for note_id={note_id}: {e}"
        ) from e


def extract_person_names_from_ner(predictions: List[NERPrediction]) -> List[str]:
    """
    Извлекает уникальные имена людей из NER предсказаний.

    Возвращает только сущности типа FIRST_NAME, LAST_NAME, MIDDLE_NAME.

    Args:
        predictions: Список NER предсказаний

    Returns:
        Список уникальных имён (дедуплицированный)
    """
    seen = set()
    names = []
    for pred in predictions:
        if pred.is_person_entity and pred.word not in seen:
            seen.add(pred.word)
            names.append(pred.word)
    return names


# --- Функции для запуска NER модели (используются в CLI команде) ---

_ner_pipeline = None


def _get_ner_pipeline():
    """Ленивая инициализация NER pipeline."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        logger.info("Loading NER model Gherman/bert-base-NER-Russian...")
        _ner_pipeline = pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            aggregation_strategy="simple",
        )
        logger.info("NER model loaded successfully")
    return _ner_pipeline


def run_ner_on_text(text: str) -> List[NERPrediction]:
    """
    Запускает NER модель на тексте.

    Args:
        text: Текст для анализа

    Returns:
        Список NERPrediction
    """
    pipe = _get_ner_pipeline()
    results = pipe(text)
    predictions = []
    for result in results:
        word = text[result["start"]:result["end"]] if result.get("start") is not None else result.get("word", "")
        predictions.append(NERPrediction(
            word=word,
            entity=result.get("entity_group") or result.get("entity", ""),
            score=result.get("score", 0.0),
            start=result.get("start"),
            end=result.get("end"),
        ))
    return predictions


def save_ner_to_cache(
    note_id: str,
    predictions: List[NERPrediction],
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Сохраняет NER предсказания в кэш.

    Args:
        note_id: ID заметки
        predictions: Список предсказаний
        cache_dir: Директория для кэша

    Returns:
        Путь к созданному файлу
    """
    if cache_dir is None:
        cache_dir = settings.ner_cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{note_id}.json"

    payload = {
        str(idx): {
            "word": pred.word,
            "entity": pred.entity,
            "score": float(pred.score) if isinstance(pred.score, numbers.Real) else None,
            "start": int(pred.start) if isinstance(pred.start, numbers.Integral) else pred.start,
            "end": int(pred.end) if isinstance(pred.end, numbers.Integral) else pred.end,
        }
        for idx, pred in enumerate(predictions)
    }

    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_path


def run_ner_for_pages(
    pages_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> int:
    """
    Прогоняет NER модель по всем md-файлам и сохраняет в кэш.

    Args:
        pages_dir: Директория с md-файлами
        cache_dir: Директория для кэша
        overwrite: Перезаписывать существующий кэш

    Returns:
        Количество обработанных файлов
    """
    if pages_dir is None:
        pages_dir = settings.pages_dir
    if cache_dir is None:
        cache_dir = settings.ner_cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    md_files = sorted(pages_dir.glob("*.md"))

    processed = 0
    for md_path in md_files:
        note_id = md_path.stem
        cache_path = cache_dir / f"{note_id}.json"

        if cache_path.exists() and not overwrite:
            logger.debug("Skipping %s (cache exists)", note_id)
            continue

        text = md_path.read_text(encoding="utf-8")
        predictions = run_ner_on_text(text)
        save_ner_to_cache(note_id, predictions, cache_dir)
        logger.info("Processed %s: %d entities", note_id, len(predictions))
        processed += 1

    return processed
