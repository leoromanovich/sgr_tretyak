from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict


ConfidenceBucket = Literal["high", "medium", "low", "very_low"]


class NoteMetadata(BaseModel):
    note_id: str = Field(
        ...,
        description="Уникальный идентификатор заметки (обычно имя файла без расширения)",
    )
    primary_year: Optional[int] = Field(
        None,
        description="Ключевой год эпизода, если можно определить (например, 1812)",
    )
    year_start: Optional[int] = Field(
        None,
        description="Начало диапазона лет, если в заметке описан период",
    )
    year_end: Optional[int] = Field(
        None,
        description="Конец диапазона лет, если в заметке описан период",
    )
    location: Optional[str] = Field(
        None,
        description="Основное место действия (город, регион), если явно указано",
    )
    topic: Optional[str] = Field(
        None,
        description="Краткое текстовое описание, о чём эта заметка",
    )
    reliability: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Насколько уверенно определены годы и контекст (0–1)",
    )

class NoteMetadataResponse(BaseModel):
    metadata: NoteMetadata


class PersonSnippetEvidence(BaseModel):
    snippet: str = Field(
        ...,
        max_length=300,
        description="Короткий фрагмент текста, где упоминается персона",
    )
    reasoning: str = Field(
        ...,
        max_length=300,
        description="Краткое объяснение, почему это упоминание интерпретируется как человек",
    )


class PersonMention(BaseModel):
    """Одно упоминание потенциального человека в тексте."""

    mention_id: str = Field(..., description="Уникальный ID упоминания (m1, m2, ...)")
    text_span: str = Field(
        ...,
        max_length=200,
        description="Точное упоминание как оно встречается в тексте",
    )
    start_char: Optional[int] = Field(
        None,
        description="Позиция начала упоминания в тексте, если известна",
    )
    context_snippet: str = Field(
        ...,
        max_length=300,
        description="Короткий контекст вокруг упоминания",
    )
    likely_person: bool = Field(
        True,
        description="true, если упоминание похоже на человека (а не организацию/место)",
    )
    inline_year: Optional[int] = Field(
        None,
        description="Год рядом с упоминанием, если можно определить",
    )


class MentionExtractionResponse(BaseModel):
    note_id: str
    mentions: List[PersonMention] = Field(
        default_factory=list, 
        max_length=50,  # Было 100
        description=(
            "Список упоминаний людей. МАКСИМУМ 50 упоминаний! "
            "Каждое упоминание должно иметь уникальный context_snippet."
        )
    )

class MentionGroup(BaseModel):
    """Группа упоминаний, относящихся к одному человеку."""

    group_id: str = Field(..., description="ID группы (g1, g2, ...)")
    mention_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Список mention_id, входящих в группу",
    )
    canonical_name: str = Field(
        ...,
        description="Наиболее полная форма имени внутри группы",
    )
    is_person: bool = Field(
        True,
        description="true, если группа соответствует реальному человеку",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность, что группа = один человек",
    )


class MentionGroupingResponse(BaseModel):
    note_id: str
    groups: List[MentionGroup] = Field(default_factory=list)


class PersonLocal(BaseModel):
    note_id: str = Field(..., description="ID заметки, где встречается персона")
    local_person_id: str = Field(
        ..., description="Локальный ID персоны внутри этой заметки (например, p1, p2)"
    )
    surface_forms: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description='Разные формы имени, встречающиеся в тексте ("И.И. Иванов", "Иван Иванович", "Иванов")',
    )
    canonical_name_in_note: str = Field(
        ...,
        description='Наиболее полная/удобная форма имени для этой заметки (например, "Иван Иванович Иванов")',
    )
    is_person: bool = Field(
        ...,
        description="true, если модель уверена, что это человек, а не организация/место",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность, что это отдельная историческая персона (0–1)",
    )
    note_year_context: Optional[int] = Field(
        None,
        description="Основной год, с которым связан этот человек в контексте данной заметки (если можно вывести)",
    )
    note_year_source: Literal["inline", "note_metadata", "unknown"] = Field(
        "unknown",
        description=(
            "Источник года: inline — год явно рядом с упоминанием; "
            "note_metadata — взят из метаданных заметки; unknown — определить нельзя"
        ),
    )
    snippet_evidence: List[PersonSnippetEvidence] = Field(
        default_factory=list,
        description="Набор фрагментов текста с объяснением, почему это человек",
    )
    role: Optional[str] = Field(
        None,
        max_length=100,
        description="Роль/профессия персоны из контекста (художник, архитектор, генерал и т.д.)",
    )
    role_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Уверенность модели в том, что роль указана корректно",
    )


class PersonExtractionResponse(BaseModel):
    note_id: str
    people: List[PersonLocal]


class NameParts(BaseModel):
    last_name: Optional[str] = Field(None, description="Фамилия, если можно выделить")
    first_name: Optional[str] = Field(None, description="Имя, если можно выделить")
    patronymic: Optional[str] = Field(None, description="Отчество, если есть")


class AbbreviationLink(BaseModel):
    from_form: str = Field(..., description="Короткая/сокращённая форма имени (например, 'П.Д. Корин')")
    to_form: str = Field(..., description="Более полная форма имени (например, 'Павел Дмитриевич Корин')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность, что это одна и та же персона (0–1)")
    reasoning: str = Field(..., description="Краткое объяснение, почему это соответствие принято")


class PersonLocalNormalized(PersonLocal):
    normalized_full_name: Optional[str] = Field(
        None,
        description=(
            "Максимально полное ФИО, которое можно собрать по этой заметке "
            "(например, 'Павел Дмитриевич Корин'). Если недостаточно данных — null."
        ),
        )
    normalized_last_name: Optional[str] = Field(
        None,
        description="Нормализованная фамилия для блокинга (ё→е, дореформенные буквы и т.п.)",
        )
    first_initial: Optional[str] = Field(
        None,
        description="Инициал имени для блокинга (нормализованный)",
        )
    patronymic_initial: Optional[str] = Field(
        None,
        description="Инициал отчества для блокинга (нормализованный)",
        )
    name_parts: NameParts = Field(
        default_factory=NameParts,
        description="Разложение имени на части, если возможно"
        )
    abbreviation_links: List[AbbreviationLink] = Field(
        default_factory=list,
        description=(
            "Список связей между сокращёнными и полными формами имени, "
            "актуальных для этой персоны в этой заметке."
        ),
        )


class PersonNormalizationResponse(BaseModel):
    note_id: str
    people: List[PersonLocalNormalized]


class PersonCandidate(BaseModel):
    """
    Кандидат-персона в глобальном пуле, одна запись = один человек в одной заметке.
    """
    candidate_id: str = Field(..., description="Глобальный ID кандидата, например '8332:p1'")
    note_id: str = Field(..., description="ID заметки (имя файла без расширения)")
    local_person_id: str = Field(..., description="Локальный ID из PersonLocal/PersonLocalNormalized")

    normalized_full_name: Optional[str] = Field(
        None,
        description="Максимально полное имя из нормализованной персоны (может быть None)",
        )
    canonical_name_in_note: str = Field(
        ...,
        description="canonical_name_in_note из нормализованной персоны",
        )
    surface_forms: List[str] = Field(
        default_factory=list,
        description="Все формы имени из нормализованной персоны",
        )
    name_parts: "NameParts" = Field(
        default_factory=lambda: NameParts(),
        description="Фамилия/имя/отчество, если выделены",
        )
    normalized_last_name: Optional[str] = Field(
        None,
        description="Нормализованная фамилия для блокинга",
        )
    first_initial: Optional[str] = Field(
        None,
        description="Инициал имени для блокинга",
        )
    patronymic_initial: Optional[str] = Field(
        None,
        description="Инициал отчества для блокинга",
        )
    note_year_context: Optional[int] = Field(
        None,
        description="Год из контекста заметки для этого человека (может быть None)",
        )
    person_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность из PersonLocal/PersonLocalNormalized",
        )
    confidence_bucket: ConfidenceBucket = Field(
        "medium",
        description=(
            "Категория уверенности: high (≥0.8), medium (0.6–0.8), "
            "low (0.4–0.6), very_low (<0.4)"
        ),
        )
    role: Optional[str] = Field(
        None,
        description="Роль/профессия персоны из контекста заметки (если есть)",
        )
    role_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Уверенность в корректности роли",
        )
    is_person: bool = Field(
        ...,
        description="Копия is_person для фильтрации",
        )
    snippet_preview: Optional[str] = Field(
        None,
        description="Один короткий snippet для понимания контекста (можно взять первый)",
        )


class EpisodeRef(BaseModel):
    note_id: str
    local_person_id: str
    note_year_context: Optional[int] = None
    snippet_preview: Optional[str] = None


class GlobalPerson(BaseModel):
    """
    Результат кластеризации: одна сущность = один человек (по нашему мнению).
    """
    global_person_id: str = Field(..., description="Глобальный ID персоны, например 'gp_0001'")
    canonical_full_name: str = Field(..., description="Выбранное каноническое имя (ФИО)")
    year_key: Optional[str] = Field(
        None,
        description="Ключ для года/периода, например '1812' или '1812–1815'",
        )
    disambiguation_key: Optional[str] = Field(
        None,
        description="Краткая пометка для различения тёзок (роль, место и т.п.)",
        )
    members: List[str] = Field(
        ...,
        description="Список candidate_id, входящих в этот кластер",
        )
    episodes: List[EpisodeRef] = Field(
        default_factory=list,
        description="Эпизоды, откуда мы знаем про этого человека",
        )

class PersonMatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_candidate_id: str
    right_candidate_id: str
    relation: Literal["same_person", "different_person", "unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class PersonMatchBinaryChoice(BaseModel):
    choice: Literal["yes", "no"] = Field(
        ...,
        description="yes — один и тот же человек; no — разные или недостаточно данных",
    )


def confidence_to_bucket(confidence: float) -> ConfidenceBucket:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    if confidence >= 0.4:
        return "low"
    return "very_low"
