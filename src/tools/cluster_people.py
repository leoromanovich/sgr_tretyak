import asyncio
import re
from typing import List, Optional

from rich import print
from tqdm.auto import tqdm

from ..models import EpisodeRef, GlobalPerson, PersonCandidate, NameParts
from .person_candidates import load_person_candidates
from .person_blocking import build_blocked_groups, load_blocking_config


def normalize_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"[^a-zа-я0-9]", "", value.lower())


def normalize_forms(forms: List[str]) -> set[str]:
    return {token for token in (normalize_token(f) for f in forms) if token}


def cheap_decision(c1: PersonCandidate, c2: PersonCandidate) -> Optional[str]:
    np1 = c1.name_parts
    np2 = c2.name_parts

    year1 = c1.note_year_context
    year2 = c2.note_year_context
    if year1 and year2 and abs(year1 - year2) > 60:
        return "different_person"

    if np1.last_name and np2.last_name and np1.last_name == np2.last_name:
        if np1.first_name and not np2.first_name and np1.patronymic and np2.patronymic and np1.patronymic != np2.patronymic:
            return "different_person"
        if np2.first_name and not np1.first_name and np1.patronymic and np2.patronymic and np1.patronymic != np2.patronymic:
            return "different_person"

    canon1 = normalize_token(c1.canonical_name_in_note)
    canon2 = normalize_token(c2.canonical_name_in_note)
    if canon1 and canon2 and canon1 == canon2:
        if year1 and year2 and abs(year1 - year2) > 30:
            return None
        return "same_person"

    if np1.last_name and np2.last_name and np1.last_name == np2.last_name:
        first_name_match = np1.first_name and np2.first_name and np1.first_name == np2.first_name
        patronymic_match = np1.patronymic and np2.patronymic and np1.patronymic == np2.patronymic

        if first_name_match:
            if year1 and year2 and abs(year1 - year2) > 40:
                return "different_person"
            if (np1.patronymic or np2.patronymic) is None or patronymic_match:
                return "same_person"
        else:
            def share_initial(candidate: PersonCandidate, other_first: Optional[str]) -> bool:
                if not candidate.canonical_name_in_note or not other_first:
                    return False
                parts = candidate.canonical_name_in_note.split()
                initials = "".join(p[0] for p in parts if p).lower()
                return bool(initials) and initials[0] == other_first[0].lower()

            if np1.first_name and not np2.first_name and share_initial(c2, np1.first_name):
                if not np1.patronymic or not np2.patronymic or np1.patronymic == np2.patronymic:
                    return "same_person"
            if np2.first_name and not np1.first_name and share_initial(c1, np2.first_name):
                if not np1.patronymic or not np2.patronymic or np1.patronymic == np2.patronymic:
                    return "same_person"

        if np1.first_name and np2.first_name and np1.first_name != np2.first_name:
            return "different_person"
        if np1.patronymic and np2.patronymic and np1.patronymic != np2.patronymic:
            return "different_person"

    if c1.normalized_full_name and c1.normalized_full_name == c2.normalized_full_name:
        return "same_person"

    forms1 = normalize_forms(c1.surface_forms)
    forms2 = normalize_forms(c2.surface_forms)
    if forms1 and forms2:
        overlap = forms1 & forms2
        if overlap:
            if not year1 or not year2 or abs(year1 - year2) <= 15:
                return "same_person"
            if year1 and year2 and abs(year1 - year2) > 40:
                return "different_person"
        else:
            if np1.last_name and np2.last_name and np1.last_name == np2.last_name:
                if np1.first_name and np2.first_name and np1.first_name != np2.first_name:
                    if np1.first_name[0] != np2.first_name[0]:
                        return "different_person"
                if np1.patronymic and np2.patronymic and np1.patronymic != np2.patronymic:
                    if np1.patronymic[0] != np2.patronymic[0]:
                        return "different_person"

    if np1.last_name and np2.last_name and np1.last_name == np2.last_name:
        def missing_core_parts(parts: NameParts) -> bool:
            return not parts.first_name and not parts.patronymic

        m1 = missing_core_parts(np1)
        m2 = missing_core_parts(np2)
        if m1 ^ m2:
            if (year1 and year2 and abs(year1 - year2) <= 5) and (forms1 & forms2):
                return "same_person"
            return None

    return None


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
    id_to_candidate = {c.candidate_id: c for c in candidates}
    uf = DSU(len(candidates))

    blocking_config = load_blocking_config()

    print(f"[bold]Всего кандидатов:[/bold] {len(candidates)}")
    print("[bold]Строим блоки для сравнения...[/bold]")

    blocks, block_stats = build_blocked_groups(candidates, blocking_config)
    print(f"[bold]Количество блоков:[/bold] {block_stats.total_blocks}")
    if block_stats.oversize_blocks:
        print(
            "[yellow]Слишком большие блоки:[/yellow] "
            f"{block_stats.oversize_blocks}, "
            f"fallback блоков: {block_stats.fallback_blocks}, "
            f"пропущено: {block_stats.skipped_blocks}"
            )

    pairs_set: set[tuple[str, str]] = set()
    for block in blocks:
        block_size = len(block)
        if block_size < 2:
            continue
        for i in range(block_size):
            for j in range(i + 1, block_size):
                left_id = block[i].candidate_id
                right_id = block[j].candidate_id
                if left_id == right_id:
                    continue
                pair_key = tuple(sorted((left_id, right_id)))
                pairs_set.add(pair_key)

    pairs = [(id_to_candidate[a], id_to_candidate[b]) for a, b in pairs_set]

    total_pairs = len(candidates) * (len(candidates) - 1) // 2
    filtered_ratio = 1 - (len(pairs) / total_pairs) if total_pairs else 0.0
    print(f"[bold]Потенциальных пар после блокинга:[/bold] {len(pairs)}")
    print(f"[bold]Отсечено пар:[/bold] {filtered_ratio:.2%}")

    print("[bold]Запускаем алгоритмический матчинг...[/bold]")

    same_person_links = 0
    compared_pairs = 0
    if pairs:
        with tqdm(total=len(pairs), desc="Алгоритмический матчинг", unit="pair", leave=True) as pbar:
            for c1, c2 in pairs:
                compared_pairs += 1
                decision = cheap_decision(c1, c2)
                if decision == "same_person":
                    uf.union(id_to_index[c1.candidate_id], id_to_index[c2.candidate_id])
                    same_person_links += 1
                pbar.update(1)
    else:
        print("[yellow]Нет пар для сравнения.[/yellow]")

    print(f"[green]Готово:[/green] проверено {compared_pairs} пар, совпадений {same_person_links}.")

    print("[bold]Строим кластеры...[/bold]")

    clusters: dict[int, List[PersonCandidate]] = {}
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
