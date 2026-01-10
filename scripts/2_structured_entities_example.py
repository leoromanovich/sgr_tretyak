from __future__ import annotations

import json
import re
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

model_name = "Qwen/Qwen3-8B-FP8"
# model_name = "ai-sage/GigaChat3-10B-A1.8B"
# model_name = "models/openai/gpt-oss-20b"

class CandidateWithType(BaseModel):
    name: str = Field(description="Имя собственное в точной форме из текста")
    is_person: bool = Field(description="True если это человек, False если нет")


class CandidateList(BaseModel):
    candidates: list[CandidateWithType] = Field(
        description="Список всех имён собственных с указанием, является ли сущность человеком"
    )


class PersonCluster(BaseModel):
    person: str = Field(description="Каноничное полное имя человека (в именительном падеже)")
    mentions: list[int] = Field(
        description="Список индексов (0-based) из списка candidates, которые относятся к этому человеку"
    )
    explanation: str = Field(
        description="Краткое объяснение, почему эти упоминания относятся к одному человеку"
    )


class PersonClusters(BaseModel):
    clusters: list[PersonCluster] = Field(
        description="Группы кандидатов, которые являются одним и тем же человеком"
    )


SAMPLES_DIR = Path("tests/samples/pages")
FALLBACK_TEXT = (
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


def run_stage1(chunk: str) -> list[CandidateWithType]:
    """Извлекает все имена собственные и определяет, являются ли они людьми."""
    stage1_messages = [
        {
            "role": "system",
            "content": (
                "Ты извлекаешь ВСЕ имена собственные из текста и определяешь, являются ли они ЛЮДЬМИ.\n\n"
                "ВАЖНО:\n"
                "- Извлекай имена в той форме (падеже), в которой они встречаются в тексте\n"
                "- Для организаций, в которых есть упоминание человека, извлекай его имя."
                "- Отмечай is_person=true ТОЛЬКО для реальных людей (имя, фамилия, инициалы)\n"
                "- Отмечай is_person=false для:\n"
                "  * названий мест (города, страны, улицы)\n"
                "  * названий организаций, трупп, компаний\n"
                "  * названий произведений искусства (фильмы, балеты, картины)\n"
                "  * географических объектов\n\n"
                "Примеры:\n"
                "- 'Вацлав Нижинский' → is_person=true\n"
                "- 'Нью-Йорк' → is_person=false (город)\n"
                "- 'Русского балета' → is_person=false (название труппы)\n"
                "- 'Тихая улица' → is_person=false (название фильма)\n\n"
                "- 'комбината имени Е.В. Вучетича' даёт два имени-> 1) Е.В. Вучетича, is_person=True, 2) 'комбината имени Е.В. Вучетича' -> is_person=false (организация)\n\n"
                f"Схема ответа: {CandidateList.model_json_schema()}"
            ),
        },
        {"role": "user", "content": chunk},
    ]
    
    stage1_response = client.chat.completions.create(
        # model="Qwen/Qwen3-4B-FP8",
        # model="Qwen/Qwen3-8B-FP8",
        model=model_name,
        messages=stage1_messages,
        temperature=0,
        max_tokens=8192,
        extra_body={
            "structured_outputs": {
                "json": CandidateList.model_json_schema(),
            }
        },
    )
    
    stage1_raw = stage1_response.choices[0].message.content
    stage1_data = CandidateList.model_validate_json(stage1_raw)
    return stage1_data.candidates


def evaluate_note(note_path: Path, note_text: str) -> dict[str, object]:
    # Извлекаем разделы
    meta_section = extract_section(note_text, "Метаинформация", "Описание")
    description_section = extract_section(note_text, "Описание", None)
    description_paragraphs = split_paragraphs(description_section)
    chunks = [chunk for chunk in [meta_section, *description_paragraphs] if chunk]

    print(f"NOTE: {note_path.name}")
    print("NOTE STRUCTURE:")
    print(f"- meta_section: {'yes' if meta_section else 'no'}")
    print(f"- description_paragraphs: {len(description_paragraphs)}")
    for idx, chunk in enumerate(chunks):
        preview = chunk.replace("\n", " ")
        preview = preview[:140] + ("…" if len(preview) > 140 else "")
        print(f"  chunk {idx + 1}: {preview}")
    print()

    # Stage 1: Извлечение кандидатов
    all_candidates: dict[str, CandidateWithType] = {}
    for chunk in chunks:
        for candidate in run_stage1(chunk):
            # Дедупликация по имени
            if candidate.name not in all_candidates:
                all_candidates[candidate.name] = candidate

    # Фильтруем только людей
    person_candidates = [c for c in all_candidates.values() if c.is_person]
    person_names = [c.name for c in person_candidates]

    print("STAGE 1 CANDIDATES (только люди):")
    for idx, name in enumerate(person_names):
        print(f"{idx}. {name}")

    if not person_names:
        print("Не найдено ни одного человека!")
        return {
            "note": note_path.name,
            "expected_patterns": [],
            "stage1_people": [],
            "stage2_people": [],
            "stage1_hits": [],
            "stage1_misses": [],
            "stage2_hits": [],
            "stage2_misses": [],
            "unexpected_stage1": [],
            "unexpected_stage2": [],
        }

    # Stage 2: Кластеризация
    indexed_candidates = "\n".join(
        f"{idx}: {name}" for idx, name in enumerate(person_names)
    )

    stage2_messages = [
        {
            "role": "system",
            "content": (
                "Ты группируешь упоминания одного и того же человека.\n\n"
                "ВАЖНО:\n"
                "- Разные падежи одного имени (Дягилев, Дягилева) = ОДИН человек\n"
                "- Полное имя и фамилия (Вацлав Нижинский, Нижинский, В. Нижинский) = ОДИН человек\n"
                "- Инициалы и полное имя (М.Ф. Ларионов) = ОДИН человек\n"
                "- Если имя встречается один раз - всё равно создай кластер из одного индекса\n"
                "- В поле 'person' указывай ПОЛНОЕ имя в ИМЕНИТЕЛЬНОМ падеже\n\n"
                "- Не придумывай информацию. Если расшифровки инициалов в примерах нет, то оставляем инициалы\n\n"
                "Примеры группировки:\n"
                "- 'Дягилев' (индекс 0) и 'Дягилева' (индекс 5) → person='Сергей Дягилев', mentions=[0, 5]\n"
                "- 'Вацлав Нижинский', 'Нижинский', 'В. Нижинский' → person='Вацлав Нижинский', mentions=[...]\n\n"
                "Выводи только JSON по схеме."
            ),
        },
        {
            "role": "user",
            "content": (
                "Контекст из заметки (для понимания, кто есть кто):\n"
                f"{note_text}\n\n"
                "---\n\n"
                "Кандидаты для группировки (индекс: имя):\n"
                f"{indexed_candidates}\n\n"
                "Сгруппируй эти имена, указав для каждого человека:\n"
                "- каноничное полное имя\n"
                "- все индексы упоминаний\n"
                "- краткое объяснение группировки"
            ),
        },
    ]

    stage2_response = client.chat.completions.create(
        # model="Qwen/Qwen3-8B-FP8",
        model=model_name,
        messages=stage2_messages,
        temperature=0,
        max_tokens=8192,
        extra_body={
            "structured_outputs": {
                "json": PersonClusters.model_json_schema(),
            }
        },
    )

    stage2_raw = stage2_response.choices[0].message.content
    stage2_data = PersonClusters.model_validate_json(stage2_raw)
    stage2_people = [cluster.person for cluster in stage2_data.clusters]

    print("\nSTAGE 2 CLUSTERS:")
    for cluster in stage2_data.clusters:
        mentions_str = ", ".join(str(idx) for idx in cluster.mentions)
        mentioned_names = [person_names[i] for i in cluster.mentions]
        print(f"- {cluster.person}: [{mentions_str}]")
        print(f"  Упоминания: {', '.join(mentioned_names)}")
        print(f"  Объяснение: {cluster.explanation}")
        print()

    # Итоговая статистика
    print(f"Всего найдено упоминаний: {len(person_names)}")
    print(f"Уникальных персон: {len(stage2_data.clusters)}")

    # Оценка качества на заметке (если есть ожидаемый файл)
    expected_path = note_path.with_name(f"{note_path.stem}_expected.json")
    expected_patterns: list[tuple[str, float | None]] = []
    if expected_path.exists():
        expected_data = json.loads(expected_path.read_text(encoding="utf-8"))
        expected_people = expected_data.get("expected_people", [])
        expected_patterns = [
            (item.get("name_pattern", ""), item.get("min_confidence", None))
            for item in expected_people
            if item.get("is_person") is True
        ]
        compiled_patterns = [
            (pattern, re.compile(pattern, re.IGNORECASE), min_conf)
            for pattern, min_conf in expected_patterns
            if pattern
        ]

        stage1_hits: list[str] = []
        stage1_misses: list[str] = []
        stage1_wrong_type: list[str] = []
        stage2_hits: list[str] = []
        stage2_misses: list[str] = []
        for pattern, regex, _min_conf in compiled_patterns:
            matched_candidates = [
                c for c in all_candidates.values() if regex.search(c.name)
            ]
            matched_people = [
                c for c in person_candidates if regex.search(c.name)
            ]
            matched_stage2 = [name for name in stage2_people if regex.search(name)]
            if matched_people:
                stage1_hits.append(pattern)
            elif matched_candidates:
                stage1_wrong_type.append(pattern)
            else:
                stage1_misses.append(pattern)
            if matched_stage2:
                stage2_hits.append(pattern)
            else:
                stage2_misses.append(pattern)

        unexpected_stage1 = [
            name
            for name in person_names
            if not any(regex.search(name) for _, regex, _ in compiled_patterns)
        ]
        unexpected_stage2 = [
            name
            for name in stage2_people
            if not any(regex.search(name) for _, regex, _ in compiled_patterns)
        ]

        print("\nEVALUATION (expected_people):")
        print(f"- stage1_hits: {len(stage1_hits)}")
        print(f"- stage1_misses: {len(stage1_misses)}")
        print(f"- stage1_wrong_type: {len(stage1_wrong_type)}")
        print(f"- stage2_hits: {len(stage2_hits)}")
        print(f"- stage2_misses: {len(stage2_misses)}")
        print(f"- unexpected_stage1: {len(unexpected_stage1)}")
        print(f"- unexpected_stage2: {len(unexpected_stage2)}")
        if stage1_misses:
            print("  Stage1 missing patterns:")
            for pattern in stage1_misses:
                print(f"    - {pattern}")
        if stage1_wrong_type:
            print("  Stage1 present but not marked as person:")
            for pattern in stage1_wrong_type:
                print(f"    - {pattern}")
        if stage2_misses:
            print("  Stage2 missing patterns:")
            for pattern in stage2_misses:
                print(f"    - {pattern}")
        if unexpected_stage1:
            print("  Stage1 unexpected people:")
            for name in unexpected_stage1:
                print(f"    - {name}")
        if unexpected_stage2:
            print("  Stage2 unexpected people:")
            for name in unexpected_stage2:
                print(f"    - {name}")
    else:
        print("\nEVALUATION: ожидаемый файл не найден.")
        stage1_hits = []
        stage1_misses = []
        stage1_wrong_type = []
        stage2_hits = []
        stage2_misses = []
        unexpected_stage1 = []
        unexpected_stage2 = []

    return {
        "note": note_path.name,
        "expected_patterns": [pattern for pattern, _ in expected_patterns],
        "stage1_people": person_names,
        "stage2_people": stage2_people,
        "stage1_hits": stage1_hits,
        "stage1_misses": stage1_misses,
        "stage1_wrong_type": stage1_wrong_type,
        "stage2_hits": stage2_hits,
        "stage2_misses": stage2_misses,
        "unexpected_stage1": unexpected_stage1,
        "unexpected_stage2": unexpected_stage2,
    }


if SAMPLES_DIR.exists():
    note_paths = sorted(SAMPLES_DIR.glob("*.md"))
else:
    note_paths = []

all_reports: list[dict[str, object]] = []
if not note_paths:
    print("Не найдено заметок в tests/samples/pages, используется FALLBACK_TEXT.")
    all_reports.append(evaluate_note(Path("fallback.md"), FALLBACK_TEXT))
else:
    for idx, note_path in enumerate(note_paths):
        note_text = note_path.read_text(encoding="utf-8")
        all_reports.append(evaluate_note(note_path, note_text))
        if idx < len(note_paths) - 1:
            print("\n" + "=" * 80 + "\n")

def format_md_list(items: list[str]) -> str:
    if not items:
        return "—"
    return "<br>".join(items)


report_lines: list[str] = ["# Structured Entities Report", ""]

for report in all_reports:
    note_name = report["note"]
    expected_patterns = report["expected_patterns"]
    stage1_people = report["stage1_people"]
    stage2_people = report["stage2_people"]
    stage1_hits = report["stage1_hits"]
    stage1_misses = report["stage1_misses"]
    stage1_wrong_type = report["stage1_wrong_type"]
    stage2_hits = report["stage2_hits"]
    stage2_misses = report["stage2_misses"]
    unexpected_stage1 = report["unexpected_stage1"]
    unexpected_stage2 = report["unexpected_stage2"]

    report_lines.append(f"## {note_name}")
    report_lines.append("")
    report_lines.append("| Section | Items |")
    report_lines.append("| --- | --- |")
    report_lines.append(
        f"| Target (expected patterns) | {format_md_list(expected_patterns)} |"
    )
    report_lines.append(
        f"| Stage 1 output (people) | {format_md_list(stage1_people)} |"
    )
    report_lines.append(
        f"| Stage 2 output (clusters.person) | {format_md_list(stage2_people)} |"
    )
    report_lines.append("")
    report_lines.append("| Errors | Items |")
    report_lines.append("| --- | --- |")
    report_lines.append(
        f"| Stage 1 missing patterns | {format_md_list(stage1_misses)} |"
    )
    report_lines.append(
        f"| Stage 1 wrong type | {format_md_list(stage1_wrong_type)} |"
    )
    report_lines.append(
        f"| Stage 1 unexpected people | {format_md_list(unexpected_stage1)} |"
    )
    report_lines.append(
        f"| Stage 2 missing patterns | {format_md_list(stage2_misses)} |"
    )
    report_lines.append(
        f"| Stage 2 unexpected people | {format_md_list(unexpected_stage2)} |"
    )
    report_lines.append("")

report_path = Path("logs/structured_entities_report.md")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\nWrote report: {report_path}")
