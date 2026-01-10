from __future__ import annotations

from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field


class CandidateList(BaseModel):
    candidates: list[str] = Field(
        description=(
            "Список всех имён собственных в порядке появления в тексте. "
            "Сохраняй точную форму написания (как в тексте)."
        )
    )


class PersonCluster(BaseModel):
    person: str = Field(description="Каноничное имя человека для кластера.")
    mentions: list[int] = Field(
        description=(
            "Список индексов (0-based) из списка candidates, которые относятся "
            "к одному и тому же человеку."
        )
    )


class PersonClusters(BaseModel):
    clusters: list[PersonCluster] = Field(
        description="Группы кандидатов, которые являются одним и тем же человеком."
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


def run_stage1(chunk: str) -> list[str]:
    stage1_messages = [
        {
            "role": "system",
            "content": (
                "Ты извлекаешь все имена людей из заметки в той форме, что указана в тексте. Требуются все вхождения имени в заметке."
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


meta_section = extract_section(note_text, "Метаинформация", "Описание")
description_section = extract_section(note_text, "Описание", None)
description_paragraphs = split_paragraphs(description_section)
chunks = [chunk for chunk in [meta_section, *description_paragraphs] if chunk]

deduped_candidates: dict[str, None] = {}
for chunk in chunks:
    for candidate in run_stage1(chunk):
        deduped_candidates.setdefault(candidate, None)

stage1_data = CandidateList(candidates=list(deduped_candidates.keys()))

print("STAGE 1 CANDIDATES:")
for idx, name in enumerate(stage1_data.candidates):
    print(f"{idx}. {name}")

indexed_candidates = "\n".join(
    f"{idx}: {name}" for idx, name in enumerate(stage1_data.candidates)
)

stage2_messages = [
    {
        "role": "system",
        "content": (
            "Ты группируешь кандидатов, которые относятся к одному и тому же человеку. "
            "Используй индексы списка candidates. "
            "Включай только тех, кто является человеком. "
            "Если имя человека встречается один раз, всё равно можешь сделать кластер из одного индекса. "
            "Выводи только JSON по схеме."
        ),
    },
    {
        "role": "user",
        "content": (
            "Заметка:\n"
            f"{note_text}\n\n"
            "Кандидаты (индекс: строка):\n"
            f"{indexed_candidates}"
        ),
    },
]

stage2_response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-FP8",
    messages=stage2_messages,
    temperature=0,
    max_tokens=512,
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
    mentions = ", ".join(str(idx) for idx in cluster.mentions)
    print(f"- {cluster.person}: [{mentions}]")
