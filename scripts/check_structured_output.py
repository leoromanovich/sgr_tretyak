"""
Utility to stress-test the structured-output (SGR) pipeline with many
concurrent requests. Each request still asks за огромное стихотворение,
но схема ограничивает ответ значением "да/нет".
"""

import argparse
import asyncio
from typing import Literal

from pydantic import BaseModel
from rich import print
from tqdm.auto import tqdm

from src.llm_client import chat_sgr_parse_async


TOTAL_REQUESTS = 1000


class YesNoResponse(BaseModel):
    answer: Literal["да", "нет"]


MESSAGES = [
    {
        "role": "system",
        "content": "Ты помощник, который должен следовать указаниям пользователя.",
    },
    {
        "role": "user",
        "content": "Пожалуйста, напиши огромное стихотворение с множеством строф и деталей. Хорошенько подумай перед ответом.",
    },
]


async def _probe_once(semaphore: asyncio.Semaphore, idx: int) -> tuple[bool, str | None]:
    async with semaphore:
        try:
            resp = await chat_sgr_parse_async(
                messages=MESSAGES,
                model_cls=YesNoResponse,
                schema_name="YesNoStructuredOutputProbe",
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure
            return False, f"request #{idx}: {exc}"
        return True, None


async def main(concurrency: int) -> None:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [_probe_once(semaphore, idx) for idx in range(1, TOTAL_REQUESTS + 1)]
    successes = 0
    failures: list[str] = []

    with tqdm(total=TOTAL_REQUESTS, desc="Structured output probes", unit="req") as pbar:
        for coro in asyncio.as_completed(tasks):
            ok, error = await coro
            if ok:
                successes += 1
            elif error:
                failures.append(error)
            pbar.update(1)

    print(
        f"[green]Готово:[/green] успешных запросов {successes}/{TOTAL_REQUESTS}, "
        f"ошибок {len(failures)}."
    )
    if failures:
        print("[red]Первые ошибки:[/red]")
        for entry in failures[:10]:
            print(" -", entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGR structured-output stress probe.")
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=32,
        help="Максимум одновременных LLM-запросов (default: 32).",
    )
    args = parser.parse_args()
    asyncio.run(main(concurrency=args.concurrency))
