#!/usr/bin/env python
"""
Валидация пайплайна извлечения людей по ground truth.

Проверяет два этапа:
1. Stage 1 (Candidates): извлечение имён собственных из chunks
2. Final (People): финальный результат после группировки

Использование:
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --samples-dir tests/samples/pages
    python scripts/validate_pipeline.py --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.models import NoteMetadata
from src.tools.mention_extractor import (
    extract_mentions_chunked_async,
    split_into_chunks,
)
from src.tools.people_extractor import extract_people_from_text_async
from src.tools.note_metadata import extract_note_metadata_from_file_async

console = Console()


@dataclass
class Stage1Result:
    """Результат валидации Stage 1."""
    note_id: str
    expected_found: list[str] = field(default_factory=list)
    expected_missing: list[str] = field(default_factory=list)
    forbidden_found: list[tuple[str, str]] = field(default_factory=list)
    extracted_candidates: list[dict] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.expected_missing and not self.forbidden_found


@dataclass
class FinalResult:
    """Результат валидации финального этапа."""
    note_id: str
    expected_found: list[str] = field(default_factory=list)
    expected_missing: list[str] = field(default_factory=list)
    forbidden_found: list[str] = field(default_factory=list)
    low_confidence: list[tuple[str, float, float]] = field(default_factory=list)
    extracted_people: list[dict] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.expected_missing and not self.forbidden_found


@dataclass
class ValidationReport:
    """Полный отчёт валидации."""
    note_id: str
    stage1: Stage1Result
    final: FinalResult
    chunks_count: int = 0
    total_duration_sec: float = 0.0


def load_expected(json_path: Path) -> dict[str, Any]:
    """Загружает ground truth из JSON."""
    return json.loads(json_path.read_text(encoding="utf-8"))


def normalize_name(name: str) -> str:
    """Нормализует имя для сравнения."""
    return name.lower().strip()


def check_name_match(actual: str, expected: str) -> bool:
    """Проверяет совпадение имени (fuzzy)."""
    actual_norm = normalize_name(actual)
    expected_norm = normalize_name(expected)

    if actual_norm == expected_norm:
        return True
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True
    if len(expected_norm) > 3 and expected_norm[:-2] in actual_norm:
        return True

    return False


def check_pattern_match(text: str, pattern: str) -> bool:
    """Проверяет совпадение по regex-паттерну."""
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except re.error:
        return pattern.lower() in text.lower()


async def validate_stage1(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    expected: dict[str, Any],
) -> Stage1Result:
    """Валидирует Stage 1 (извлечение кандидатов)."""
    result = Stage1Result(note_id=note_id)

    start = time.perf_counter()
    mentions_resp = await extract_mentions_chunked_async(note_id, text, metadata)
    result.duration_sec = time.perf_counter() - start

    extracted_names = [m.text_span for m in mentions_resp.mentions]
    result.extracted_candidates = [
        {"name": m.text_span, "is_person": m.likely_person}
        for m in mentions_resp.mentions
    ]

    stage1_gt = expected.get("stage1_candidates", {})
    expected_candidates = stage1_gt.get("expected", [])
    forbidden_patterns = stage1_gt.get("forbidden", [])

    for exp in expected_candidates:
        exp_name = exp["name"]
        found = any(check_name_match(actual, exp_name) for actual in extracted_names)
        if found:
            result.expected_found.append(exp_name)
        else:
            result.expected_missing.append(exp_name)

    for forb in forbidden_patterns:
        pattern = forb["name_pattern"]
        reason = forb.get("reason", "unknown")
        for actual in extracted_names:
            if check_pattern_match(actual, pattern):
                mention = next(
                    (m for m in mentions_resp.mentions if m.text_span == actual),
                    None
                )
                if mention and mention.likely_person:
                    result.forbidden_found.append((actual, reason))

    return result


async def validate_final(
    note_id: str,
    text: str,
    metadata: NoteMetadata,
    expected: dict[str, Any],
) -> FinalResult:
    """Валидирует финальный этап (люди после группировки)."""
    result = FinalResult(note_id=note_id)

    start = time.perf_counter()
    people_resp = await extract_people_from_text_async(
        note_id=note_id,
        text=text,
        metadata=metadata,
        validate=True,
    )
    result.duration_sec = time.perf_counter() - start

    result.extracted_people = [
        {
            "canonical_name": p.canonical_name_in_note,
            "surface_forms": p.surface_forms,
            "is_person": p.is_person,
            "confidence": p.confidence,
        }
        for p in people_resp.people
    ]

    final_gt = expected.get("final_people", {})
    expected_people = final_gt.get("expected", [])
    forbidden_patterns = final_gt.get("forbidden_patterns", [])

    for exp in expected_people:
        pattern = exp["name_pattern"]
        min_conf = exp.get("min_confidence", 0.0)

        found = False
        for person in people_resp.people:
            all_forms = [person.canonical_name_in_note] + person.surface_forms
            if any(check_pattern_match(form, pattern) for form in all_forms):
                found = True
                if person.confidence < min_conf:
                    result.low_confidence.append((pattern, person.confidence, min_conf))
                break

        if found:
            result.expected_found.append(pattern)
        else:
            result.expected_missing.append(pattern)

    for pattern in forbidden_patterns:
        for person in people_resp.people:
            all_forms = [person.canonical_name_in_note] + person.surface_forms
            if any(check_pattern_match(form, pattern) for form in all_forms):
                result.forbidden_found.append(pattern)
                break

    return result


async def validate_note(
    md_path: Path,
    expected_path: Path,
) -> ValidationReport:
    """Валидирует одну заметку."""
    note_id = md_path.stem
    text = md_path.read_text(encoding="utf-8")
    expected = load_expected(expected_path)

    meta_resp = await extract_note_metadata_from_file_async(md_path)
    metadata = meta_resp.metadata

    chunks = split_into_chunks(text)

    start_total = time.perf_counter()
    stage1_result = await validate_stage1(note_id, text, metadata, expected)
    final_result = await validate_final(note_id, text, metadata, expected)
    total_duration = time.perf_counter() - start_total

    return ValidationReport(
        note_id=note_id,
        stage1=stage1_result,
        final=final_result,
        chunks_count=len(chunks),
        total_duration_sec=total_duration,
    )


def print_note_details(report: ValidationReport, expected: dict[str, Any]):
    """Выводит детальную информацию по заметке."""
    console.print(f"\n{'='*60}")
    console.print(f"[bold cyan]{report.note_id}[/bold cyan]  "
                  f"(chunks: {report.chunks_count}, "
                  f"time: {report.total_duration_sec:.1f}s)")
    console.print('='*60)

    # --- Stage 1 ---
    s1 = report.stage1
    status1 = "[green]PASS[/green]" if s1.passed else "[red]FAIL[/red]"
    console.print(f"\n[bold]Stage 1: Candidates[/bold]  {status1}  ({s1.duration_sec:.1f}s)")

    # Что извлекла модель
    console.print("\n  [dim]Model extracted:[/dim]")
    persons = [c for c in s1.extracted_candidates if c["is_person"]]
    non_persons = [c for c in s1.extracted_candidates if not c["is_person"]]

    if persons:
        names_str = ", ".join(c["name"] for c in persons[:15])
        if len(persons) > 15:
            names_str += f" ... (+{len(persons)-15})"
        console.print(f"    [green]is_person=True:[/green] {names_str}")

    if non_persons:
        names_str = ", ".join(c["name"] for c in non_persons[:10])
        if len(non_persons) > 10:
            names_str += f" ... (+{len(non_persons)-10})"
        console.print(f"    [yellow]is_person=False:[/yellow] {names_str}")

    if s1.expected_missing:
        console.print(f"\n  [red]Missing ({len(s1.expected_missing)}):[/red]")
        for name in s1.expected_missing:
            console.print(f"    - {name}")

    if s1.forbidden_found:
        console.print(f"\n  [yellow]Forbidden found ({len(s1.forbidden_found)}):[/yellow]")
        for name, reason in s1.forbidden_found:
            console.print(f"    - {name} ({reason})")

    if s1.passed:
        console.print(f"  [green]All {len(s1.expected_found)} expected candidates found[/green]")

    # --- Final ---
    f = report.final
    status2 = "[green]PASS[/green]" if f.passed else "[red]FAIL[/red]"
    console.print(f"\n[bold]Final: People[/bold]  {status2}  ({f.duration_sec:.1f}s)")

    # Что извлекла модель
    console.print("\n  [dim]Model extracted:[/dim]")
    if f.extracted_people:
        for p in f.extracted_people:
            conf_color = "green" if p["confidence"] >= 0.7 else "yellow" if p["confidence"] >= 0.5 else "red"
            forms = ", ".join(p["surface_forms"][:3])
            if len(p["surface_forms"]) > 3:
                forms += "..."
            console.print(f"    [{conf_color}]{p['confidence']:.2f}[/{conf_color}] "
                          f"{p['canonical_name']}  [dim]forms: {forms}[/dim]")
    else:
        console.print("    [dim](no people extracted)[/dim]")

    # Что ожидалось
    final_gt = expected.get("final_people", {})
    expected_patterns = final_gt.get("expected", [])

    if f.expected_missing:
        console.print(f"\n  [red]Missing patterns ({len(f.expected_missing)}):[/red]")
        for pattern in f.expected_missing:
            # Найдём comment если есть
            exp_item = next((e for e in expected_patterns if e["name_pattern"] == pattern), {})
            comment = exp_item.get("comment", "")
            if comment:
                console.print(f"    - {pattern}  [dim]({comment})[/dim]")
            else:
                console.print(f"    - {pattern}")

    if f.forbidden_found:
        console.print(f"\n  [yellow]Forbidden found:[/yellow]")
        for pattern in f.forbidden_found:
            console.print(f"    - {pattern}")

    if f.low_confidence:
        console.print(f"\n  [magenta]Low confidence:[/magenta]")
        for pattern, actual, minimum in f.low_confidence:
            console.print(f"    - {pattern}: {actual:.2f} < {minimum:.2f}")

    if f.passed:
        console.print(f"  [green]All {len(f.expected_found)} expected people found[/green]")


def print_summary_table(reports: list[ValidationReport]) -> bool:
    """Выводит итоговую таблицу. Возвращает True если все тесты прошли."""
    all_passed = True

    # Только упавшие тесты в таблице
    failed_reports = [r for r in reports if not r.stage1.passed or not r.final.passed]

    if failed_reports:
        table = Table(title="[red]Failed Tests[/red]", show_lines=True)
        table.add_column("Note", style="cyan", width=12)
        table.add_column("Stage", width=8)
        table.add_column("Issue", style="red")
        table.add_column("Details")

        for r in failed_reports:
            all_passed = False

            if not r.stage1.passed:
                issues = []
                if r.stage1.expected_missing:
                    issues.append(f"missing: {len(r.stage1.expected_missing)}")
                if r.stage1.forbidden_found:
                    issues.append(f"forbidden: {len(r.stage1.forbidden_found)}")

                details = ", ".join(r.stage1.expected_missing[:2])
                if len(r.stage1.expected_missing) > 2:
                    details += "..."

                table.add_row(r.note_id, "Stage1", " + ".join(issues), details)

            if not r.final.passed:
                issues = []
                if r.final.expected_missing:
                    issues.append(f"missing: {len(r.final.expected_missing)}")
                if r.final.forbidden_found:
                    issues.append(f"forbidden: {len(r.final.forbidden_found)}")

                details = ", ".join(r.final.expected_missing[:2])
                if len(r.final.expected_missing) > 2:
                    details += "..."

                table.add_row(r.note_id, "Final", " + ".join(issues), details)

        console.print(table)
    else:
        console.print("[green]All tests passed![/green]")

    # Статистика
    total = len(reports)
    stage1_passed = sum(1 for r in reports if r.stage1.passed)
    final_passed = sum(1 for r in reports if r.final.passed)
    total_time = sum(r.total_duration_sec for r in reports)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(f"  Model: [cyan]{settings.effective_model}[/cyan]")
    console.print(f"  Stage 1: {stage1_passed}/{total} passed")
    console.print(f"  Final:   {final_passed}/{total} passed")
    console.print(f"  Total time: {total_time:.1f}s")

    return all_passed


def save_report(reports: list[ValidationReport]):
    """Сохраняет детальный отчёт в reports/."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"validation_{timestamp}.json"

    total_time = sum(r.total_duration_sec for r in reports)

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "provider": settings.llm_provider,
        "model": settings.effective_model,
        "base_url": settings.effective_base_url,
        "total_duration_sec": total_time,
        "summary": {
            "total_notes": len(reports),
            "stage1_passed": sum(1 for r in reports if r.stage1.passed),
            "final_passed": sum(1 for r in reports if r.final.passed),
        },
        "notes": [],
    }

    for r in reports:
        note_data = {
            "note_id": r.note_id,
            "chunks_count": r.chunks_count,
            "total_duration_sec": r.total_duration_sec,
            "stage1": {
                "passed": r.stage1.passed,
                "duration_sec": r.stage1.duration_sec,
                "expected_found": r.stage1.expected_found,
                "expected_missing": r.stage1.expected_missing,
                "forbidden_found": [{"name": n, "reason": rs} for n, rs in r.stage1.forbidden_found],
                "extracted_candidates": r.stage1.extracted_candidates,
            },
            "final": {
                "passed": r.final.passed,
                "duration_sec": r.final.duration_sec,
                "expected_found": r.final.expected_found,
                "expected_missing": r.final.expected_missing,
                "forbidden_found": r.final.forbidden_found,
                "low_confidence": [
                    {"pattern": p, "actual": a, "min": m}
                    for p, a, m in r.final.low_confidence
                ],
                "extracted_people": r.final.extracted_people,
            },
        }
        report_data["notes"].append(note_data)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    console.print(f"\n[dim]Report saved: {report_path}[/dim]")


async def main():
    parser = argparse.ArgumentParser(description="Validate extraction pipeline")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("tests/samples/pages"),
        help="Directory with sample .md and _expected.json files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed extraction results and save report",
    )
    args = parser.parse_args()

    if args.verbose:
        os.environ["SGR_LOG_FAILURES"] = "1"

    samples_dir = args.samples_dir
    if not samples_dir.exists():
        console.print(f"[red]Samples directory not found: {samples_dir}[/red]")
        sys.exit(1)

    # Находим пары (md, expected.json)
    test_cases = []
    expected_data = {}
    for md_file in sorted(samples_dir.glob("*.md")):
        expected_file = md_file.with_name(f"{md_file.stem}_expected.json")
        if expected_file.exists():
            test_cases.append((md_file, expected_file))
            expected_data[md_file.stem] = load_expected(expected_file)

    if not test_cases:
        console.print(f"[yellow]No test cases found in {samples_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[bold]Validating {len(test_cases)} notes[/bold]")
    console.print(f"  Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(f"  Model: [cyan]{settings.effective_model}[/cyan]\n")

    # Запускаем валидацию
    reports = []
    for md_path, expected_path in test_cases:
        console.print(f"  Processing {md_path.stem}...", end="")
        report = await validate_note(md_path, expected_path)
        reports.append(report)

        status = "[green]OK[/green]" if report.stage1.passed and report.final.passed else "[red]FAIL[/red]"
        console.print(f" {status} ({report.total_duration_sec:.1f}s)")

    console.print()

    # При verbose - детальный вывод
    if args.verbose:
        for report in reports:
            expected = expected_data[report.note_id]
            print_note_details(report, expected)

    # Итоговая таблица
    console.print()
    all_passed = print_summary_table(reports)

    # При verbose - сохраняем отчёт
    if args.verbose:
        save_report(reports)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
