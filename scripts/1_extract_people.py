from pathlib import Path
from typing import List

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError


# ====== Pydantic-схема результата ======

class Person(BaseModel):
    name: str
    role: str


class People(BaseModel):
    people: List[Person]


# ====== Чтение заметки ======

NOTE_PATH = Path("../example.md")

if not NOTE_PATH.exists():
    raise FileNotFoundError(f"Не найден файл {NOTE_PATH.resolve()}")

note_md = NOTE_PATH.read_text(encoding="utf-8")


# ====== JSON Schema для structured output ======

json_schema = People.model_json_schema()


# ====== Клиент OpenAI (vLLM OpenAI server) ======

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM игнорирует, но параметр обязателен
    )


# ====== Вызов модели с structured output ======

messages = [
    {
        "role": "system",
        "content": (
            "Ты извлекаешь людей (персон) из русскоязычной заметки "
            "о произведении искусства и заполняешь JSON по заданной схеме.\n\n"
            "Схема (словами):\n"
            "- Корневой объект с ключом \"people\".\n"
            "- \"people\" — массив объектов с полями:\n"
            "  - \"name\" — строка, полное нормализованное имя человека;\n"
            "  - \"role\" — строка, краткое описание роли человека в контексте заметки.\n\n"
            "Требования:\n"
            "- Ищи только конкретных людей (персон), без абстрактных групп.\n"
            "- Разные варианты имени одного человека объединяй в одну запись.\n"
            "- Выводи ТОЛЬКО JSON, без ```json, без комментариев, без лишнего текста."
        ),
        },
    {
        "role": "user",
        "content": note_md,
        },
    ]

response = client.chat.completions.create(
    # model="models/openai/gpt-oss-20b",
    model="models/Qwen/Qwen3-14B-FP8",
    messages=messages,
    temperature=0,
    max_tokens=512,
    extra_body={
        # Ключевой параметр: включаем structured output с JSON Schema
        "structured_outputs": {
            "json": json_schema
            }
        },
    )

raw = response.choices[0].message.content
print("RAW RESPONSE:")
print(raw)
print("-" * 80)

# ====== Валидация ответа Pydantic'ом ======

try:
    people = People.model_validate_json(raw)
except ValidationError as e:
    print("Pydantic validation error:")
    print(e)
    raise

print("EXTRACTED PEOPLE:")
for person in people.people:
    print(f"- {person.name} → {person.role}")