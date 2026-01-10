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


NOTE_PATH = Path("../tests/samples/pages/157163.md")

if NOTE_PATH.exists():
    note_text = NOTE_PATH.read_text(encoding="utf-8")
else:
    note_text = (
        "Вчера я встретил Ивана Петрова в офисе Acme Corp. "
        "Позже Иван говорил с Петром, а Петров зашёл в кафе на Невском проспекте."
    )

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

stage1_messages = [
    {
        "role": "system",
        "content": (
            "Ты извлекаешь все имена собственные из заметки. "
            "Включай любые имена собственные (люди, организации, места, бренды) "
            "и сохраняй точную форму написания и порядок появления. "
            "Инвентарный номер не является именем собственным. "
            "Выводи только JSON по схеме."
        ),
    },
    {"role": "user", "content": note_text},
]

stage1_response = client.chat.completions.create(
    model="models/Qwen/Qwen3-14B-FP8",
    messages=stage1_messages,
    temperature=0,
    max_tokens=512,
    extra_body={
        "structured_outputs": {
            "json": CandidateList.model_json_schema(),
        }
    },
)

stage1_raw = stage1_response.choices[0].message.content
stage1_data = CandidateList.model_validate_json(stage1_raw)

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
    model="models/Qwen/Qwen3-14B-FP8",
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
