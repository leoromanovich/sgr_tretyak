from typing import List, Tuple
from ..models import PersonCandidate

def block_pairs(candidates: List[PersonCandidate], year_delta: int = 40) -> List[Tuple[PersonCandidate, PersonCandidate]]:
    """
    Возвращает список пар кандидатов, которые потенциально могут быть одним человеком.
    Чтобы ограничить количество LLM-вызовов, сравниваем только тех, кто подходит по фамилии и инициалам.
    """
    pairs = []
    n = len(candidates)

    for i in range(n):
        left = candidates[i]
        ln1 = left.name_parts.last_name
        fn1 = left.name_parts.first_name
        pn1 = left.name_parts.patronymic

        for j in range(i + 1, n):
            right = candidates[j]
            ln2 = right.name_parts.last_name
            fn2 = right.name_parts.first_name
            pn2 = right.name_parts.patronymic

            # фамилия обязательна
            if not ln1 or not ln2 or ln1 != ln2:
                continue

            # имя если есть — первая буква должна совпадать
            if fn1 and fn2 and fn1[0] != fn2[0]:
                continue

            # отчество — то же
            if pn1 and pn2 and pn1[0] != pn2[0]:
                continue

            # если есть годы — проверяем разумный интервал
            if left.note_year_context and right.note_year_context:
                if abs(left.note_year_context - right.note_year_context) > year_delta:
                    continue

            pairs.append((left, right))

    return pairs