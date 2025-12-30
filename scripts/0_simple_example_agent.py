from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM на это не смотрит
    )

# Самая простая JSON Schema
json_schema = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Короткий ответ модели"
            }
        },
    "required": ["answer"],
    "additionalProperties": False,
    }

messages = [
    {
        "role": "system",
        "content": (
            "Отвечай строго в формате JSON по схеме. "
            "Не добавляй ```json, не добавляй комментариев."
        ),
        },
    {
        "role": "user",
        "content": "Скажи привет по-русски в поле answer.",
        },
    ]

response = client.chat.completions.create(
    model="models/openai/gpt-oss-20b",
    messages=messages,
    temperature=0,
    max_tokens=128,
    # КЛЮЧЕВОЕ: vLLM structured outputs
    extra_body={
        "structured_outputs": {
            "json": json_schema
            }
        },
    )

raw = response.choices[0].message.content
print("RAW:", raw)

# Если structured output реально работает, json.loads не должен падать
data = json.loads(raw)
print("PARSED:", data)

print("answer:", data["answer"])