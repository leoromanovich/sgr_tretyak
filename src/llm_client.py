from typing import Any, Dict, List, Optional, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config import settings

client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    )

T = TypeVar("T", bound=BaseModel)


def chat_raw(
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": temperature,
        }

    if response_format is not None:
        kwargs["response_format"] = response_format

    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    message = resp.choices[0].message

    return {
        "message": message,
        "raw": resp,
        }


def chat_sgr_parse(
    messages: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
    model_cls: Type[T],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    ) -> T:
    """
    Делает вызов LLM с response_format=json_schema и парсит
    результат через Pydantic-модель `model_cls`.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,  # включи, если твой vLLM это поддерживает
            },
        }

    resp = chat_raw(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        )

    content = resp["message"].content
    if not content:
        raise RuntimeError("LLM вернул пустой content при SGR-вызове")

    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Невалидный JSON от LLM: {e}\nRaw: {content}")

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(f"JSON не прошёл валидацию Pydantic: {e}\nData: {data}")