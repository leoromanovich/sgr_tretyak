import asyncio
from typing import List, Dict
from rich import print, progress

from ..models import PersonCandidate, PersonMatchDecision, GlobalPerson, EpisodeRef, NameParts
from .person_candidates import load_person_candidates
from .blocking import block_pairs
from .person_matcher import match_candidates_async

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


async def cluster_people_async(
    conf_threshold: float = 0.8,
    match_workers: int = 8,
    ) -> List[GlobalPerson]:
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

    limiter = asyncio.Semaphore(max(1, match_workers))

    async def process_pair(c1: PersonCandidate, c2: PersonCandidate):
        async with limiter:
            try:
                decision: PersonMatchDecision = await match_candidates_async(c1, c2)
                return c1, c2, decision, None
            except Exception as exc:
                return c1, c2, None, exc

    tasks = [asyncio.create_task(process_pair(c1, c2)) for c1, c2 in pairs]

    if tasks:
        with progress.Progress() as pbar:
            task = pbar.add_task("[green]Матчинг пар...", total=len(tasks))
            for coro in asyncio.as_completed(tasks):
                c1, c2, decision, err = await coro
                if err:
                    print(f"[red]Ошибка при сравнении {c1.candidate_id} vs {c2.candidate_id}:[/red] {err}")
                elif decision.relation == "same_person" and decision.confidence >= conf_threshold:
                    uf.union(id_to_index[c1.candidate_id], id_to_index[c2.candidate_id])
                pbar.advance(task)

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


def cluster_people(conf_threshold: float = 0.8, match_workers: int = 8) -> List[GlobalPerson]:
    return asyncio.run(
        cluster_people_async(
            conf_threshold=conf_threshold,
            match_workers=match_workers,
            )
        )
