import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from src.config import settings

LLM_CONCURRENCY_LIMIT = 16

client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    )
async_client = AsyncOpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    )
_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)

T = TypeVar("T", bound=BaseModel)


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Нельзя вызывать синхронный LLM-клиент внутри активного event loop; "
        "используй chat_raw_async/chat_sgr_parse_async"
        )


async def chat_raw_async(
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

    async with _llm_semaphore:
        resp = await async_client.chat.completions.create(**kwargs)
    message = resp.choices[0].message

    return {
        "message": message,
        "raw": resp,
        }


def chat_raw(
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
    """
    Синхронная обёртка поверх chat_raw_async для обратной совместимости.
    """
    return _run_sync(
        chat_raw_async(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            )
        )


async def chat_sgr_parse_async(
    messages: List[Dict[str, Any]],
    model_cls: Type[T],
    schema_name: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    ) -> T:
    """
    Делает вызов LLM с response_format=json_schema и парсит
    результат через Pydantic-модель `model_cls`.
    """
    if schema is None:
        schema = model_cls.model_json_schema()
    if schema_name is None:
        schema_name = schema.get("title") or model_cls.__name__

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,  # включи, если твой vLLM это поддерживает
            },
        }

    resp = await chat_raw_async(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        )

    content = resp["message"].content
    if not content:
        raise RuntimeError("LLM вернул пустой content при SGR-вызове")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Невалидный JSON от LLM: {e}\nRaw: {content}")

    if _should_trace():
        trace_dir = settings.project_root / "log_tracing"
        trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_path = trace_dir / f"{schema_name}_{timestamp}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_name": schema_name,
                    "messages": messages,
                    "response": resp["raw"].model_dump(),
                    "parsed_json": data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(f"JSON не прошёл валидацию Pydantic: {e}\nData: {data}")


def chat_sgr_parse(
    messages: List[Dict[str, Any]],
    model_cls: Type[T],
    schema_name: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    ) -> T:
    """
    Синхронная обёртка поверх chat_sgr_parse_async.
    """
    return _run_sync(
        chat_sgr_parse_async(
            messages=messages,
            model_cls=model_cls,
            schema_name=schema_name,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            )
        )
def _should_trace() -> bool:
    return os.getenv("SGR_LOGGING") == "DEBUG"
