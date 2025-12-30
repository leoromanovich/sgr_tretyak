from typing import List, Dict
from rich import print

from ..models import PersonCandidate, PersonMatchDecision, GlobalPerson, EpisodeRef, NameParts
from .person_candidates import load_person_candidates
from .blocking import block_pairs
from .person_matcher import match_candidates

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_people(conf_threshold: float = 0.8) -> List[GlobalPerson]:
    candidates = load_person_candidates()
    if not candidates:
        raise RuntimeError("Нет кандидатов (запусти scan-people).")

    id_to_index = {c.candidate_id: idx for idx, c in enumerate(candidates)}
    uf = DSU(len(candidates))

    print(f"[bold]Всего кандидатов:[/bold] {len(candidates)}")
    print("[bold]Строим пары для сравнения...[/bold]")

    pairs = block_pairs(candidates)
    print(f"[bold]Потенциальных пар:[/bold] {len(pairs)}")

    print("[bold]Запускаем LLM-матчинг...[/bold]")

    for c1, c2 in pairs:
        try:
            decision: PersonMatchDecision = match_candidates(c1, c2)
        except Exception as e:
            print(f"[red]Ошибка при сравнении {c1.candidate_id} vs {c2.candidate_id}:[/red] {e}")
            continue

        if decision.relation == "same_person" and decision.confidence >= conf_threshold:
            uf.union(id_to_index[c1.candidate_id], id_to_index[c2.candidate_id])

    print("[bold]Строим кластеры...[/bold]")

    clusters: Dict[int, List[PersonCandidate]] = {}
    for idx, c in enumerate(candidates):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(c)

    global_persons: List[GlobalPerson] = []
    next_gid = 1

    for root_idx, members in clusters.items():
        # Пытаемся собрать ФИО из name_parts
        def build_full_name_from_parts(m: PersonCandidate) -> str | None:
            lp = m.name_parts.last_name
            fn = m.name_parts.first_name
            pn = m.name_parts.patronymic

            # если вообще нет фамилии — не используем
            if not lp and not fn:
                return None

            # порядок: Имя Отчество Фамилия (если есть имя)
            parts = []
            if fn:
                parts.append(fn)
            if pn:
                parts.append(pn)
            if lp:
                parts.append(lp)

            if not parts:
                return None
            return " ".join(parts)

        name_variants: list[str] = []
        for m in members:
            full = build_full_name_from_parts(m)
            if full:
                name_variants.append(full)

        if name_variants:
            canonical_name = max(name_variants, key=len)
        else:
            # fallback: как раньше
            names = [m.normalized_full_name for m in members if m.normalized_full_name]
            if names:
                canonical_name = max(names, key=len)
            else:
                canonical_name = members[0].canonical_name_in_note

        episodes = [
            EpisodeRef(
                note_id=m.note_id,
                local_person_id=m.local_person_id,
                note_year_context=m.note_year_context,
                snippet_preview=m.snippet_preview,
            )
            for m in members
        ]

        gp = GlobalPerson(
            global_person_id=f"gp_{next_gid:04d}",
            canonical_full_name=canonical_name,
            year_key=None,
            disambiguation_key=None,
            members=[m.candidate_id for m in members],
            episodes=episodes,
        )
        global_persons.append(gp)
        next_gid += 1

    print(f"[bold green]Готово. Кластеров:[/bold green] {len(global_persons)}")
    return global_persons