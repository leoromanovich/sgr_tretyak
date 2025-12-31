import asyncio
import re
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import orjson
from rich import print
from tqdm.auto import tqdm

from ..config import settings
from ..models import EpisodeRef, GlobalPerson, PersonCandidate, PersonMatchDecision
from .blocking import block_pairs
from .person_candidates import load_person_candidates
from .person_matcher import match_candidates_async

CACHE_DIR = settings.project_root / "cache"
MATCH_CACHE_PATH = CACHE_DIR / "person_match_cache.jsonl"

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


def load_match_cache() -> Dict[Tuple[str, str], Tuple[str, float]]:
    cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
    if not MATCH_CACHE_PATH.exists():
        return cache

    with open(MATCH_CACHE_PATH, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            data = orjson.loads(line)
            left = data["left_id"]
            right = data["right_id"]
            cache[(left, right)] = (data["relation"], data["confidence"])
    return cache


def append_match_cache(entries: List[Tuple[str, str, str, float]]) -> None:
    if not entries:
        return
    MATCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MATCH_CACHE_PATH, "ab") as f:
        for left, right, relation, confidence in entries:
            payload = {
                "left_id": left,
                "right_id": right,
                "relation": relation,
                "confidence": confidence,
                }
            f.write(orjson.dumps(payload))
            f.write(b"\n")


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
    match_cache = load_match_cache()
    cache_lock = asyncio.Lock()
    new_entries: List[Tuple[str, str, str, float]] = []
    matching_started_at = perf_counter()

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

        canon1 = normalize_token(c1.canonical_name_in_note)
        canon2 = normalize_token(c2.canonical_name_in_note)
        if canon1 and canon2 and canon1 == canon2:
            return "same_person"

        # если обе стороны обладают ФИО (фамилия+имя (+ отчество)), и они совпадают целиком — можем сразу принять same_person
        if np1.last_name and np2.last_name and np1.last_name == np2.last_name:
            first_name_match = np1.first_name and np2.first_name and np1.first_name == np2.first_name
            patronymic_match = np1.patronymic and np2.patronymic and np1.patronymic == np2.patronymic

            if first_name_match:
                # отчество может быть None у обеих — тоже считаем совпадением
                if (np1.patronymic or np2.patronymic) is None or patronymic_match:
                    return "same_person"

            # если и фамилия, и имя заполнены, но разные — явно different
            if np1.first_name and np2.first_name and np1.first_name != np2.first_name:
                return "different_person"
            if np1.patronymic and np2.patronymic and np1.patronymic != np2.patronymic:
                return "different_person"

        # fallback: если normalized_full_name у обеих и совпадает 1-в-1
        if c1.normalized_full_name and c1.normalized_full_name == c2.normalized_full_name:
            return "same_person"

        forms1 = normalize_forms(c1.surface_forms)
        forms2 = normalize_forms(c2.surface_forms)
        if forms1 and forms2:
            overlap = forms1 & forms2
            if overlap:
                if not year1 or not year2 or abs(year1 - year2) <= 15:
                    return "same_person"

        # если surface_forms полностью разные и при этом фамилии совпадают, но имена разные — считаем разных
        if forms1 and forms2 and not (forms1 & forms2):
            if np1.last_name and np2.last_name and np1.last_name == np2.last_name:
                if np1.first_name and np2.first_name and np1.first_name != np2.first_name:
                    return "different_person"

        return None

    async def process_pair(c1: PersonCandidate, c2: PersonCandidate):
        # сначала дешёвая проверка без LLM
        key = tuple(sorted((c1.candidate_id, c2.candidate_id)))

        cached = match_cache.get(key)
        if cached:
            return c1, c2, cached, None

        quick = cheap_decision(c1, c2)
        if quick is not None:
            return c1, c2, quick, None

        async with limiter:
            try:
                decision: PersonMatchDecision = await match_candidates_async(c1, c2)
                async with cache_lock:
                    match_cache[key] = (decision.relation, decision.confidence)
                    new_entries.append((key[0], key[1], decision.relation, decision.confidence))
                return c1, c2, decision, None
            except Exception as exc:
                return c1, c2, None, exc

    tasks = [asyncio.create_task(process_pair(c1, c2)) for c1, c2 in pairs]

    if tasks:
        with tqdm(total=len(tasks), desc="Матчинг пар", unit="pair", leave=True) as pbar:
            for coro in asyncio.as_completed(tasks):
                c1, c2, decision, err = await coro
                if err:
                    print(f"[red]Ошибка при сравнении {c1.candidate_id} vs {c2.candidate_id}:[/red] {err}")
                else:
                    if isinstance(decision, tuple):
                        relation, confidence = decision
                    elif isinstance(decision, str):
                        relation, confidence = decision, 1.0
                    else:
                        relation = decision.relation
                        confidence = decision.confidence
                    if relation == "same_person" and confidence >= conf_threshold:
                        uf.union(id_to_index[c1.candidate_id], id_to_index[c2.candidate_id])
                pbar.update(1)
    append_match_cache(new_entries)
    if tasks:
        elapsed = perf_counter() - matching_started_at
        print(f"[blue]Матчинг:[/blue] {len(tasks)} пар за {elapsed:.1f}с (~{elapsed/len(tasks):.2f}с/пару) "
              f"при {match_workers} воркерах")

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
CACHE_DIR = settings.project_root / "cache"
MATCH_CACHE_PATH = CACHE_DIR / "person_match_cache.jsonl"
