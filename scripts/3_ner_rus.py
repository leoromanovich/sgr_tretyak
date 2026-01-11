import argparse
import json
import os
from pathlib import Path

import numbers

from transformers import pipeline


def run_ner_on_text(ner_pipe, text: str) -> list[dict]:
    return ner_pipe(text)

def format_result(result: dict, source_text: str, show_offsets: bool) -> str:
    if isinstance(result.get("start"), int) and isinstance(result.get("end"), int):
        word = source_text[result["start"] : result["end"]]
        offsets = f" [{result['start']}:{result['end']}]"
    else:
        word = result.get("word") or result.get("text") or "<unknown>"
        word = word.replace("##", "")
        offsets = ""
    entity = result.get("entity") or result.get("entity_group") or "<unknown>"
    score = result.get("score")
    if isinstance(score, numbers.Real):
        score_str = f"{float(score):.4f}"
    else:
        score_str = "n/a"
    if show_offsets and offsets:
        return f"Word: {word}{offsets}, Entity: {entity}, Score: {score_str}"
    return f"Word: {word}, Entity: {entity}, Score: {score_str}"


def result_to_cache_entry(result: dict, source_text: str) -> dict:
    if isinstance(result.get("start"), int) and isinstance(result.get("end"), int):
        word = source_text[result["start"] : result["end"]]
        start = result["start"]
        end = result["end"]
    else:
        word = result.get("word") or result.get("text") or "<unknown>"
        word = word.replace("##", "")
        start = None
        end = None

    entity = result.get("entity") or result.get("entity_group") or "<unknown>"
    score = result.get("score")
    score_value = float(score) if isinstance(score, numbers.Real) else None

    return {
        "word": word,
        "entity": entity,
        "score": score_value,
        "start": start,
        "end": end,
    }


def write_cache(cache_dir: Path, md_path: Path, results: list[dict], text: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{md_path.stem}.json"
    payload = {
        str(index): result_to_cache_entry(result, text)
        for index, result in enumerate(results)
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER on sample notes")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("tests/samples/pages"),
        help="Directory with sample .md and _expected.json files",
    )
    parser.add_argument(
        "--show-offsets",
        action="store_true",
        help="Show start/end offsets for matched entities",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache_sample/ner_predicts"),
        help="Directory for cached NER predictions",
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir
    if not samples_dir.exists():
        raise SystemExit(f"Samples directory not found: {samples_dir}")

    ner_pipe = pipeline(
        "ner",
        model="Gherman/bert-base-NER-Russian",
        aggregation_strategy="simple",
    )

    for md_path in sorted(samples_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")

        print(f"\n=== {md_path.stem} ===")
        print(f"Loaded markdown: {md_path}")
        results = run_ner_on_text(ner_pipe, text)
        write_cache(args.cache_dir, md_path, results, text)
        for result in results:
            print(format_result(result, text, args.show_offsets))


if __name__ == "__main__":
    main()
