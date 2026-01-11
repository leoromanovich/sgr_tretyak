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
from src.llm_prompts import add_no_think
import logging

logger = logging.getLogger(__name__)


def _apply_no_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Добавляет /no_think к system prompt для Qwen3 моделей.
    Работает только для локального провайдера (vLLM с Qwen).
    """
    if settings.llm_provider != "local":
        return messages

    result = []
    for msg in messages:
        if msg.get("role") == "system":
            new_msg = msg.copy()
            new_msg["content"] = add_no_think(msg["content"])
            result.append(new_msg)
        else:
            result.append(msg)
    return result

LLM_CONCURRENCY_LIMIT = 32 
LLM_TIMEOUT_RETRY_DELAY_SECONDS = 2.0

def _get_extra_headers() -> dict[str, str]:
    """Возвращает дополнительные заголовки для OpenRouter."""
    if settings.llm_provider == "openrouter":
        return {
            "HTTP-Referer": "https://github.com/sgr-tretyak",
            "X-Title": "SGR Tretyak Pipeline",
        }
    return {}


client = OpenAI(
    base_url=settings.effective_base_url,
    api_key=settings.effective_api_key,
    timeout=settings.llm_request_timeout,
    max_retries=settings.llm_client_max_retries,
    default_headers=_get_extra_headers(),
)
async_client = AsyncOpenAI(
    base_url=settings.effective_base_url,
    api_key=settings.effective_api_key,
    timeout=settings.llm_request_timeout,
    max_retries=settings.llm_client_max_retries,
    default_headers=_get_extra_headers(),
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


def detect_repetition_loop(content: str, threshold: int = 3) -> bool:
    """
    Определяет, зациклилась ли модель на повторении одного паттерна.

    Проверяет последние N объектов в массиве на идентичность.
    """
    if not content or len(content) < 100:
        return False

    # Ищем последние несколько объектов в массиве
    # Паттерн: {"mention_id": "m123", ... }, {"mention_id": "m124", ... }
    import re
    # Находим все объекты вида {"..."}
    objects = re.findall(r'\{[^{}]*\}', content[-5000:])  # Смотрим последние 5000 символов

    if len(objects) < threshold * 2:
        return False

    # Проверяем, что последние N объектов почти идентичны (отличаются только ID)
    last_objects = objects[-threshold:]

    # Нормализуем объекты, удаляя ID для сравнения
    normalized = []
    for obj in last_objects:
        # Убираем mention_id/group_id и их значения
        normalized_obj = re.sub(r'"(?:mention_id|group_id)":\s*"[^"]*"', '', obj)
        normalized.append(normalized_obj)

    # Если все нормализованные объекты идентичны, это петля
    if len(set(normalized)) == 1 and len(normalized) >= threshold:
        logger.warning("Обнаружена петля повторений в JSON (последние %d объектов идентичны)", threshold)
        return True

    return False


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

    # Также проверяем на петли повторений
    if detect_repetition_loop(content):
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
    frequency_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
    # Применяем /no_think для Qwen3 моделей
    processed_messages = _apply_no_think(messages)

    kwargs: Dict[str, Any] = {
        "model": settings.effective_model,
        "messages": processed_messages,
        "temperature": temperature,
    }

    # Добавляем frequency_penalty для борьбы с повторениями
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty

    # reasoning_effort только для локальных моделей (vLLM)
    if settings.llm_provider == "local":
        kwargs["extra_body"] = {"reasoning_effort": "low"}

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
    frequency_penalty: Optional[float] = None,
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
            frequency_penalty=frequency_penalty,
            )
        )


async def chat_sgr_parse_async(
    messages: List[Dict[str, Any]],
    model_cls: Type[T],
    schema_name: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
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
            frequency_penalty=frequency_penalty,
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
                    if _should_log_failures():
                        _log_failed_response(
                            schema_name=schema_name,
                            messages=messages,
                            resp=resp,
                            content=content,
                            error=last_error,
                            attempt=attempt,
                            parse_attempts=parse_attempts,
                        )
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
                if _should_log_failures():
                    _log_failed_response(
                        schema_name=schema_name,
                        messages=messages,
                        resp=resp,
                        content=content,
                        error=last_error,
                        attempt=attempt,
                        parse_attempts=parse_attempts,
                    )
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
            if _should_log_failures():
                _log_failed_response(
                    schema_name=schema_name,
                    messages=messages,
                    resp=resp,
                    content=content,
                    error=last_error,
                    attempt=attempt,
                    parse_attempts=parse_attempts,
                )
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
    frequency_penalty: Optional[float] = None,
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
            frequency_penalty=frequency_penalty,
            )
        )
def _should_trace() -> bool:
    return os.getenv("SGR_LOGGING") == "DEBUG"


def _should_log_failures() -> bool:
    return os.getenv("SGR_LOG_FAILURES") == "1"


def _log_failed_response(
    *,
    schema_name: str,
    messages: List[Dict[str, Any]],
    resp: Dict[str, Any],
    content: str,
    error: str,
    attempt: int,
    parse_attempts: int,
) -> None:
    trace_dir = settings.project_root / "log_tracing" / "failures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = trace_dir / f"{schema_name}_{timestamp}.json"
    record = {
        "timestamp": timestamp,
        "schema_name": schema_name,
        "attempt": attempt + 1,
        "max_attempts": parse_attempts,
        "error": error,
        "messages": messages,
        "content": content,
        "response": resp["raw"].model_dump(),
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)


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
