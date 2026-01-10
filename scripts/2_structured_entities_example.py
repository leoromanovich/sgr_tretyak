from __future__ import annotations

from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field


class CandidateWithType(BaseModel):
    name: str = Field(description="Имя собственное в точной форме из текста")
    is_person: bool = Field(description="True если это человек, False если нет")


class CandidateList(BaseModel):
    candidates: list[CandidateWithType] = Field(
        description="Список всех имён собственных с указанием, является ли сущность человеком"
    )


class PersonCluster(BaseModel):
    person: str = Field(description="Каноничное полное имя человека (в именительном падеже)")
    mentions: list[int] = Field(
        description="Список индексов (0-based) из списка candidates, которые относятся к этому человеку"
    )
    explanation: str = Field(
        description="Краткое объяснение, почему эти упоминания относятся к одному человеку"
    )


class PersonClusters(BaseModel):
    clusters: list[PersonCluster] = Field(
        description="Группы кандидатов, которые являются одним и тем же человеком"
    )


NOTE_PATH = Path("tests/samples/pages/157163.md")

if NOTE_PATH.exists():
    note_text = NOTE_PATH.read_text(encoding="utf-8")
else:
    print("NOT FOUND NOTE")
    note_text = (
        "Вчера я встретил Ивана Петрова в офисе Acme Corp. "
        "Позже Иван говорил с Петром, а Петров зашёл в кафе на Невском проспекте."
    )

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)


def extract_section(text: str, header: str, next_header: str | None) -> str:
    marker = f"## {header}"
    start = text.find(marker)
    if start == -1:
        return ""
    start = start + len(marker)
    end = text.find(f"## {next_header}", start) if next_header else -1
    return text[start:end].strip() if end != -1 else text[start:].strip()


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def run_stage1(chunk: str) -> list[CandidateWithType]:
    """Извлекает все имена собственные и определяет, являются ли они людьми."""
    stage1_messages = [
        {
            "role": "system",
            "content": (
                "Ты извлекаешь ВСЕ имена собственные из текста и определяешь, являются ли они ЛЮДЬМИ.\n\n"
                "ВАЖНО:\n"
                "- Извлекай имена в той форме (падеже), в которой они встречаются в тексте\n"
                "- Отмечай is_person=true ТОЛЬКО для реальных людей (имя, фамилия, инициалы)\n"
                "- Отмечай is_person=false для:\n"
                "  * названий мест (города, страны, улицы)\n"
                "  * названий организаций, трупп, компаний\n"
                "  * названий произведений искусства (фильмы, балеты, картины)\n"
                "  * географических объектов\n\n"
                "Примеры:\n"
                "- 'Вацлав Нижинский' → is_person=true\n"
                "- 'Нью-Йорк' → is_person=false (город)\n"
                "- 'Русского балета' → is_person=false (название труппы)\n"
                "- 'Тихая улица' → is_person=false (название фильма)\n\n"
                f"Схема ответа: {CandidateList.model_json_schema()}"
            ),
        },
        {"role": "user", "content": chunk},
    ]
    
    stage1_response = client.chat.completions.create(
        model="Qwen/Qwen3-4B-FP8",
        messages=stage1_messages,
        temperature=0.1,
        max_tokens=2048,
        extra_body={
            "structured_outputs": {
                "json": CandidateList.model_json_schema(),
            }
        },
    )
    
    stage1_raw = stage1_response.choices[0].message.content
    stage1_data = CandidateList.model_validate_json(stage1_raw)
    return stage1_data.candidates


# Извлекаем разделы
meta_section = extract_section(note_text, "Метаинформация", "Описание")
description_section = extract_section(note_text, "Описание", None)
description_paragraphs = split_paragraphs(description_section)
chunks = [chunk for chunk in [meta_section, *description_paragraphs] if chunk]

# Stage 1: Извлечение кандидатов
all_candidates: dict[str, CandidateWithType] = {}
for chunk in chunks:
    for candidate in run_stage1(chunk):
        # Дедупликация по имени
        if candidate.name not in all_candidates:
            all_candidates[candidate.name] = candidate

# Фильтруем только людей
person_candidates = [c for c in all_candidates.values() if c.is_person]
person_names = [c.name for c in person_candidates]

print("STAGE 1 CANDIDATES (только люди):")
for idx, name in enumerate(person_names):
    print(f"{idx}. {name}")

if not person_names:
    print("Не найдено ни одного человека!")
    exit(0)

# Stage 2: Кластеризация
indexed_candidates = "\n".join(
    f"{idx}: {name}" for idx, name in enumerate(person_names)
)

stage2_messages = [
    {
        "role": "system",
        "content": (
            "Ты группируешь упоминания одного и того же человека.\n\n"
            "ВАЖНО:\n"
            "- Разные падежи одного имени (Дягилев, Дягилева) = ОДИН человек\n"
            "- Полное имя и фамилия (Вацлав Нижинский, Нижинский, В. Нижинский) = ОДИН человек\n"
            "- Инициалы и полное имя (М.Ф. Ларионов) = ОДИН человек\n"
            "- Если имя встречается один раз - всё равно создай кластер из одного индекса\n"
            "- В поле 'person' указывай ПОЛНОЕ имя в ИМЕНИТЕЛЬНОМ падеже\n\n"
            "Примеры группировки:\n"
            "- 'Дягилев' (индекс 0) и 'Дягилева' (индекс 5) → person='Сергей Дягилев', mentions=[0, 5]\n"
            "- 'Вацлав Нижинский', 'Нижинский', 'В. Нижинский' → person='Вацлав Нижинский', mentions=[...]\n\n"
            "Выводи только JSON по схеме."
        ),
    },
    {
        "role": "user",
        "content": (
            "Контекст из заметки (для понимания, кто есть кто):\n"
            f"{note_text}\n\n"
            "---\n\n"
            "Кандидаты для группировки (индекс: имя):\n"
            f"{indexed_candidates}\n\n"
            "Сгруппируй эти имена, указав для каждого человека:\n"
            "- каноничное полное имя\n"
            "- все индексы упоминаний\n"
            "- краткое объяснение группировки"
        ),
    },
]

stage2_response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-FP8",
    messages=stage2_messages,
    temperature=0,
    max_tokens=1024,
    extra_body={
        "structured_outputs": {
            "json": PersonClusters.model_json_schema(),
        }
    },
)

stage2_raw = stage2_response.choices[0].message.content
stage2_data = PersonClusters.model_validate_json(stage2_raw)

print("\nSTAGE 2 CLUSTERS:")
for cluster in stage2_data.clusters:
    mentions_str = ", ".join(str(idx) for idx in cluster.mentions)
    mentioned_names = [person_names[i] for i in cluster.mentions]
    print(f"- {cluster.person}: [{mentions_str}]")
    print(f"  Упоминания: {', '.join(mentioned_names)}")
    print(f"  Объяснение: {cluster.explanation}")
    print()

# Итоговая статистика
print(f"Всего найдено упоминаний: {len(person_names)}")
print(f"Уникальных персон: {len(stage2_data.clusters)}")
