"""
Integration tests for people extraction from notes.

These tests run actual LLM calls, so they:
- Require a running LLM server
- Are slow (use pytest -m integration to run separately)
- May have some variance in results

Test data structure:
    tests/samples/pages/
        8331.md              # note file
        8331_expected.json   # expected extraction results
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.models import PersonExtractionResponse
from src.tools.people_extractor import extract_people_from_file_async


SAMPLES_DIR = Path(__file__).parent / "samples" / "pages"


@dataclass
class ExpectedPerson:
    name_pattern: str
    is_person: bool
    min_confidence: float = 0.0


@dataclass
class ExpectedResult:
    note_id: str
    expected_people: list[ExpectedPerson]
    forbidden_patterns: list[str]


def load_expected(json_path: Path) -> ExpectedResult:
    """Load expected results from JSON file."""
    data = json.loads(json_path.read_text())
    return ExpectedResult(
        note_id=data["note_id"],
        expected_people=[
            ExpectedPerson(
                name_pattern=p["name_pattern"],
                is_person=p["is_person"],
                min_confidence=p.get("min_confidence", 0.0),
            )
            for p in data["expected_people"]
        ],
        forbidden_patterns=data.get("forbidden_patterns", []),
    )


def find_test_cases() -> list[tuple[Path, Path]]:
    """Find all (note.md, note_expected.json) pairs in samples directory."""
    cases = []
    for md_file in sorted(SAMPLES_DIR.glob("*.md")):
        expected_file = md_file.with_name(f"{md_file.stem}_expected.json")
        if expected_file.exists():
            cases.append((md_file, expected_file))
    return cases


def check_person_matches(canonical_name: str, surface_forms: list[str], pattern: str) -> bool:
    """Check if person matches the expected pattern."""
    all_forms = [canonical_name] + surface_forms
    regex = re.compile(pattern, re.IGNORECASE)
    return any(regex.search(form) for form in all_forms)


def validate_extraction(
    result: PersonExtractionResponse,
    expected: ExpectedResult,
) -> list[str]:
    """
    Validate extraction result against expected.
    Returns list of error messages (empty if all checks pass).
    """
    errors = []

    # Check each expected person is found
    for exp_person in expected.expected_people:
        found = False
        for person in result.people:
            if check_person_matches(
                person.canonical_name_in_note,
                person.surface_forms,
                exp_person.name_pattern,
            ):
                found = True
                # Validate is_person flag
                if person.is_person != exp_person.is_person:
                    errors.append(
                        f"Person matching '{exp_person.name_pattern}': "
                        f"is_person={person.is_person}, expected {exp_person.is_person}"
                    )
                # Validate confidence
                if person.confidence < exp_person.min_confidence:
                    errors.append(
                        f"Person matching '{exp_person.name_pattern}': "
                        f"confidence={person.confidence:.2f}, expected >= {exp_person.min_confidence}"
                    )
                break

        if not found:
            errors.append(f"Expected person not found: pattern='{exp_person.name_pattern}'")

    # Check forbidden patterns are not present
    for forbidden in expected.forbidden_patterns:
        regex = re.compile(forbidden, re.IGNORECASE)
        for person in result.people:
            all_forms = [person.canonical_name_in_note] + person.surface_forms
            if any(regex.search(form) for form in all_forms):
                errors.append(
                    f"Forbidden pattern '{forbidden}' found in person: "
                    f"'{person.canonical_name_in_note}'"
                )

    return errors


async def run_extraction_for_all(cases: list[tuple[Path, Path]]) -> list[PersonExtractionResponse]:
    """Run extraction for all cases concurrently."""
    tasks = [extract_people_from_file_async(md_path) for md_path, _ in cases]
    return await asyncio.gather(*tasks)


class TestPeopleExtraction:
    """Integration tests for people extraction."""

    @pytest.fixture(scope="class")
    def extraction_results(self) -> dict[str, tuple[PersonExtractionResponse, ExpectedResult]]:
        """
        Run all extractions once and cache results.
        This allows async batch processing on the server side.
        """
        cases = find_test_cases()
        if not cases:
            pytest.skip("No test cases found in tests/samples/pages/")

        results = asyncio.run(run_extraction_for_all(cases))

        # Map note_id -> (result, expected)
        results_map = {}
        for (md_path, expected_path), result in zip(cases, results):
            expected = load_expected(expected_path)
            results_map[expected.note_id] = (result, expected)

        return results_map

    @pytest.fixture(scope="class")
    def test_case_ids(self, extraction_results) -> list[str]:
        """Get list of test case IDs."""
        return list(extraction_results.keys())

    def test_extraction_returns_correct_note_id(self, extraction_results):
        """Verify note_id matches in results."""
        for note_id, (result, expected) in extraction_results.items():
            assert result.note_id == expected.note_id, (
                f"Note ID mismatch: got '{result.note_id}', expected '{expected.note_id}'"
            )

    def test_expected_people_found(self, extraction_results):
        """Verify all expected people are found with correct attributes."""
        all_errors = []
        for note_id, (result, expected) in extraction_results.items():
            errors = validate_extraction(result, expected)
            if errors:
                extracted_people = [
                    f"{person.canonical_name_in_note} "
                    f"(forms: {', '.join(person.surface_forms)})"
                    for person in result.people
                ] or ["<no people extracted>"]

                expected_patterns = [ep.name_pattern for ep in expected.expected_people]

                all_errors.append(
                    f"\n[{note_id}]:\n"
                    f"  " + "\n  ".join(errors) + "\n"
                    f"  Extracted people:\n"
                    f"    " + "\n    ".join(extracted_people) + "\n"
                    f"  Expected patterns:\n"
                    f"    " + "\n    ".join(expected_patterns)
                )

        assert not all_errors, "Extraction validation failed:" + "".join(all_errors)

    def test_no_forbidden_patterns(self, extraction_results):
        """Verify forbidden patterns are not extracted as people."""
        all_errors = []
        for note_id, (result, expected) in extraction_results.items():
            for forbidden in expected.forbidden_patterns:
                regex = re.compile(forbidden, re.IGNORECASE)
                for person in result.people:
                    all_forms = [person.canonical_name_in_note] + person.surface_forms
                    if any(regex.search(form) for form in all_forms):
                        all_errors.append(
                            f"[{note_id}]: Forbidden '{forbidden}' in '{person.canonical_name_in_note}'"
                        )

        assert not all_errors, "Forbidden patterns found:\n" + "\n".join(all_errors)


# Standalone runner for debugging
if __name__ == "__main__":
    async def main():
        cases = find_test_cases()
        print(f"Found {len(cases)} test case(s)")

        for md_path, expected_path in cases:
            print(f"\nProcessing: {md_path.name}")
            expected = load_expected(expected_path)

            result = await extract_people_from_file_async(md_path)

            print(f"  Found {len(result.people)} people:")
            for p in result.people:
                print(f"    - {p.canonical_name_in_note} (confidence={p.confidence:.2f})")

            errors = validate_extraction(result, expected)
            if errors:
                print("  ERRORS:")
                for e in errors:
                    print(f"    - {e}")
            else:
                print("  OK")

    asyncio.run(main())
