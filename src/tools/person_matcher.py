from ..llm_client import chat_raw, chat_raw_async, chat_sgr_parse, chat_sgr_parse_async
from ..llm_prompts import add_no_think
from ..models import PersonCandidate, PersonMatchDecision


ANALYSIS_SYSTEM_PROMPT = """
Ты делаешь короткое аналитическое описание для пары кандидатов.

Нужно:
- выделить, что говорит в пользу того, что это один человек;
- указать признаки, что они разные;
- дать грубую оценку (likely_same / likely_different / uncertain).

Не выдай финальный вердикт, просто подготовь заметки (до ~200 токенов).
"""

DECISION_SYSTEM_PROMPT = """
Ты анализируешь двух кандидатов на одну и ту же историческую персону.

Укажи отношение:
1. same_person — если это однозначно один человек.
2. different_person — если они точно разные.
3. unknown — если данных недостаточно (лучше быть осторожным).

Правила:
- НЕ использовать внешние знания.
- Основываться только на данных кандидатов и аналитических заметках.
- Если normalized_full_name совпадает полностью — скорее всего same_person.
- Если normalized_full_name отсутствует, но совпадает фамилия, имя и отчество — same_person.
- Если есть совпадение фамилии и имени, но отчество различается — different_person.
- Если нет имени, но совпадают фамилия + отчество, а год близкий — возможно same_person.
- Если роли совпадают и имена близки — это дополнительный сигнал в пользу same_person.
- Если роли явно разные и нет полного совпадения имён — скорее different_person.
- Если расходятся более двух ключевых параметров — different_person.
- Если данных мало или неоднозначно — unknown.

Верни строго JSON по схеме.
"""


def _build_base_user_content(c1: PersonCandidate, c2: PersonCandidate) -> str:
    return f"""
Сравни двух кандидатов:

[LEFT]
candidate_id: {c1.candidate_id}
normalized_full_name: {c1.normalized_full_name}
canonical_name: {c1.canonical_name_in_note}
name_parts: {c1.name_parts}
year: {c1.note_year_context}
role: {c1.role}
confidence: {c1.person_confidence}

[RIGHT]
candidate_id: {c2.candidate_id}
normalized_full_name: {c2.normalized_full_name}
canonical_name: {c2.canonical_name_in_note}
name_parts: {c2.name_parts}
year: {c2.note_year_context}
role: {c2.role}
confidence: {c2.person_confidence}
"""


def _build_analysis_messages(c1: PersonCandidate, c2: PersonCandidate):
    return [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": add_no_think(_build_base_user_content(c1, c2))},
        ]


def _build_match_messages(
    c1: PersonCandidate,
    c2: PersonCandidate,
    analysis_notes: str | None = None,
    ):
    user_content = _build_base_user_content(c1, c2)
    if analysis_notes:
        user_content += f"""
[ANALYSIS_NOTES]
{analysis_notes.strip()}
"""
    return [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": add_no_think(user_content)},
        ]


async def _generate_analysis_text_async(
    c1: PersonCandidate,
    c2: PersonCandidate,
    max_tokens: int = 200,
    ) -> str:
    resp = await chat_raw_async(
        messages=_build_analysis_messages(c1, c2),
        temperature=0.0,
        max_tokens=max_tokens,
        )
    content = resp["message"].content or ""
    return content.strip()


def _generate_analysis_text(c1: PersonCandidate, c2: PersonCandidate, max_tokens: int = 200) -> str:
    resp = chat_raw(
        messages=_build_analysis_messages(c1, c2),
        temperature=0.0,
        max_tokens=max_tokens,
        )
    content = resp["message"].content or ""
    return content.strip()


async def match_candidates_async(
    c1: PersonCandidate,
    c2: PersonCandidate,
    use_analysis: bool = False,
    ) -> PersonMatchDecision:
    analysis_text: str | None = None
    if use_analysis:
        analysis_text = await _generate_analysis_text_async(c1, c2)
    messages = _build_match_messages(c1, c2, analysis_text)
    return await chat_sgr_parse_async(
        messages=messages,
        model_cls=PersonMatchDecision,
        schema_name="person_matcher",
        temperature=0.0,
        max_tokens=256,
        )


def match_candidates(
    c1: PersonCandidate,
    c2: PersonCandidate,
    use_analysis: bool = False,
    ) -> PersonMatchDecision:
    analysis_text: str | None = None
    if use_analysis:
        analysis_text = _generate_analysis_text(c1, c2)
    messages = _build_match_messages(c1, c2, analysis_text)
    return chat_sgr_parse(
        messages=messages,
        model_cls=PersonMatchDecision,
        schema_name="person_matcher",
        temperature=0.0,
        max_tokens=256,
        )
