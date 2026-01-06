from ..llm_client import chat_sgr_parse, chat_sgr_parse_async
from ..llm_prompts import add_no_think
from ..models import PersonCandidate, PersonMatchDecision


SYSTEM_PROMPT = """
Ты анализируешь двух кандидатов на одну и ту же историческую персону.

Тебе даны:
- normalized_full_name (если есть),
- canonical_name_in_note,
- surface_forms,
- разложение имени по частям (фамилия, имя, отчество),
- год из контекста заметки (если есть),
- confidence.

Укажи отношение:
1. same_person — если это однозначно один человек.
2. different_person — если они точно разные.
3. unknown — если данных недостаточно (лучше быть осторожным).

Правила:
- НЕ использовать внешние знания.
- Основываться только на данных кандидатов.
- Если normalized_full_name совпадает полностью — скорее всего same_person.
- Если normalized_full_name отсутствует, но совпадает фамилия, имя и отчество — same_person.
- Если есть совпадение фамилии и имени, но отчество различается — different_person.
- Если нет имени, но совпадают фамилия + отчество, а год близкий — возможно same_person.
- Если расходятся более двух ключевых параметров — different_person.
- Если данных мало или неоднозначно — unknown.

Верни строго JSON по схеме.
"""


def _build_match_messages(c1: PersonCandidate, c2: PersonCandidate):
    user_content = f"""
Сравни двух кандидатов:

[LEFT]
candidate_id: {c1.candidate_id}
normalized_full_name: {c1.normalized_full_name}
canonical_name: {c1.canonical_name_in_note}
name_parts: {c1.name_parts}
year: {c1.note_year_context}
confidence: {c1.person_confidence}

[RIGHT]
candidate_id: {c2.candidate_id}
normalized_full_name: {c2.normalized_full_name}
canonical_name: {c2.canonical_name_in_note}
name_parts: {c2.name_parts}
year: {c2.note_year_context}
confidence: {c2.person_confidence}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": add_no_think(user_content)},
        ]


async def match_candidates_async(c1: PersonCandidate, c2: PersonCandidate) -> PersonMatchDecision:
    messages = _build_match_messages(c1, c2)
    return await chat_sgr_parse_async(
        messages=messages,
        model_cls=PersonMatchDecision,
        schema_name="person_matcher",
        temperature=0.0,
        max_tokens=None,
        )


def match_candidates(c1: PersonCandidate, c2: PersonCandidate) -> PersonMatchDecision:
    messages = _build_match_messages(c1, c2)
    return chat_sgr_parse(
        messages=messages,
        model_cls=PersonMatchDecision,
        schema_name="person_matcher",
        temperature=0.0,
        max_tokens=None,
        )
