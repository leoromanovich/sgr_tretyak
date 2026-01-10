import asyncio
import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import APITimeoutError, AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError
from rich import print as rich_print

from src.config import settings
import logging

logger = logging.getLogger(__name__)

LLM_CONCURRENCY_LIMIT = 32 
LLM_TIMEOUT_RETRY_DELAY_SECONDS = 2.0

client = OpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    timeout=settings.llm_request_timeout,
    max_retries=settings.llm_client_max_retries,
    )
async_client = AsyncOpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key,
    timeout=settings.llm_request_timeout,
    max_retries=settings.llm_client_max_retries,
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


def detect_truncation(content: str) -> bool:
    """Определяет, был ли JSON обрезан."""
    if not content:
        return True
    content = content.strip()
    if not content.endswith('}') and not content.endswith(']'):
        return True
    if content.count('{') != content.count('}'):
        return True
    if content.count('[') != content.count(']'):
        return True
    return False


def _deduplicate_mentions_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Удаляет дублирующиеся mentions, у которых совпадают text_span и context_snippet.
    """
    mentions = data.get("mentions")
    if not isinstance(mentions, list):
        return data

    seen_keys = set()
    deduped: List[Dict[str, Any]] = []
    removed = 0

    for item in mentions:
        if not isinstance(item, dict):
            deduped.append(item)
            continue
        key = (item.get("text_span"), item.get("context_snippet"))
        if key in seen_keys:
            removed += 1
            continue
        seen_keys.add(key)
        deduped.append(item)

    if removed:
        logger.debug("Deduplicated %s repeated mentions", removed)
        data["mentions"] = deduped

    return data


def _postprocess_response_data(schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Дополнительная нормализация json-ответа от LLM перед валидацией.
    """
    if schema_name == "mention_extraction":
        return _deduplicate_mentions_payload(data)
    return data


def try_repair_truncated_json(content: str) -> Optional[Dict[str, Any]]:
    """Пытается восстановить обрезанный JSON."""
    if not content:
        return None
    
    # Находим последний полный объект
    last_complete_obj = content.rfind('},')
    if last_complete_obj == -1:
        last_complete_obj = content.rfind('}')
    if last_complete_obj == -1:
        return None
    
    truncated = content[:last_complete_obj + 1]
    
    # Закрываем незакрытые скобки
    open_brackets = truncated.count('[') - truncated.count(']')
    open_braces = truncated.count('{') - truncated.count('}')
    
    repaired = truncated + ']' * open_brackets + '}' * open_braces
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None

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
        "extra_body": {"reasoning_effort": "low"}
        }

    if response_format is not None:
        kwargs["response_format"] = response_format

    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    effective_max_tokens = max_tokens
    if effective_max_tokens is None:
        effective_max_tokens = settings.llm_max_output_tokens

    if effective_max_tokens is not None:
        kwargs["max_tokens"] = effective_max_tokens

    timeout_retries = max(0, settings.llm_timeout_retries)
    attempt = 0
    while True:
        try:
            async with _llm_semaphore:
                resp = await async_client.chat.completions.create(**kwargs)
            break
        except APITimeoutError as exc:
            _log_timeout_request(kwargs, attempt, timeout_retries)
            if attempt >= timeout_retries:
                raise exc
            delay = LLM_TIMEOUT_RETRY_DELAY_SECONDS
            rich_print(
                f"[yellow]LLM timeout (attempt {attempt + 1}/{timeout_retries + 1}), "
                f"retrying in {delay:.1f}s...[/yellow]"
                )
            await asyncio.sleep(delay)
            attempt += 1
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
            # "strict": True,  # включи, если твой vLLM это поддерживает
            },
        }

    parse_attempts = 2
    last_error: Optional[str] = None

    for attempt in range(parse_attempts):
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
            if detect_truncation(content):
                logger.warning(
                    "JSON обрезан (превышен max_tokens?), пытаемся восстановить..."
                )
                data = try_repair_truncated_json(content)
                if data is None:
                    last_error = (
                        f"Невалидный JSON от LLM (обрезан, восстановить не удалось): {e}\n"
                        f"Raw (последние 500 символов): ...{content[-500:]}"
                    )
                    if attempt + 1 < parse_attempts:
                        logger.warning(
                            "Повторный запрос после обрезанного JSON (attempt %s/%s)",
                            attempt + 1,
                            parse_attempts,
                        )
                        continue
                    raise RuntimeError(last_error)
                logger.warning(
                    "JSON успешно восстановлен, но данные могут быть неполными"
                )
            else:
                last_error = f"Невалидный JSON от LLM: {e}\nRaw: {content}"
                if attempt + 1 < parse_attempts:
                    logger.warning(
                        "LLM вернул невалидный JSON (attempt %s/%s), повторяем запрос",
                        attempt + 1,
                        parse_attempts,
                    )
                    continue
                raise RuntimeError(last_error)

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
            data = _postprocess_response_data(schema_name, data)
            return model_cls.model_validate(data)
        except ValidationError as e:
            last_error = f"JSON не прошёл валидацию Pydantic: {e}\nData: {data}"
            if attempt + 1 < parse_attempts:
                logger.warning(
                    "LLM ответ не прошёл валидацию (attempt %s/%s), повторяем запрос",
                    attempt + 1,
                    parse_attempts,
                )
                continue
            raise RuntimeError(last_error)

    # по идее не должно сюда доходить
    raise RuntimeError(last_error or "Не удалось распарсить ответ LLM")


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


def _log_timeout_request(payload: Dict[str, Any], attempt: int, timeout_retries: int) -> None:
    """
    Записывает полный payload LLM-запроса, если хотя бы одна попытка завершилась таймаутом.
    """
    trace_dir = settings.project_root / "log_tracing" / "timeouts"
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_payload = copy.deepcopy(payload)
    record = {
        "timestamp": timestamp,
        "attempt": attempt + 1,
        "max_attempts": timeout_retries + 1,
        "request": safe_payload,
    }
    trace_path = trace_dir / f"timeout_{timestamp}_attempt{attempt + 1}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
