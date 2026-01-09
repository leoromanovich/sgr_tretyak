import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from openai import APITimeoutError
from rich import print
from tqdm.auto import tqdm

from ..config import settings
from ..models import (
    EpisodeRef,
    GlobalPerson,
    PersonCandidate,
    PersonMatchDecision,
    NameParts,
    )

CACHE_DIR = settings.project_root / "cache"
CLUSTERS_CACHE_PATH = CACHE_DIR / "persons_clusters.json"


def _log_match_timeout(c1: PersonCandidate, c2: PersonCandidate, error: Exception) -> None:
    """
    Записываем информацию о паре, для которой матчинг упал по таймауту.
    """
    trace_dir = settings.project_root / "log_tracing" / "timeouts"
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    payload = {
        "timestamp": timestamp,
        "error": str(error),
        "left_candidate_id": c1.candidate_id,
        "right_candidate_id": c2.candidate_id,
        "left_note_id": c1.note_id,
        "right_note_id": c2.note_id,
        }
    trace_path = trace_dir / f"match_timeout_{timestamp}_{c1.candidate_id}_{c2.candidate_id}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_clusters_to_cache(
    persons: List[GlobalPerson],
    conf_threshold: float,
    match_workers: int,
    use_match_analysis: bool,
    cache_path: Path = CLUSTERS_CACHE_PATH,
    ) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "conf_threshold": conf_threshold,
        "match_workers": match_workers,
        "total_clusters": len(persons),
        "use_match_analysis": use_match_analysis,
        "persons": [p.model_dump() for p in persons],
        }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[green]Сохранено в кэш:[/green] {cache_path.relative_to(settings.project_root)} "
        f"(use_match_analysis={use_match_analysis})"
        )


def load_clusters_from_cache(cache_path: Path = CLUSTERS_CACHE_PATH) -> tuple[List[GlobalPerson], dict]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    persons_data = data.get("persons")
    if not isinstance(persons_data, list):
        raise RuntimeError(f"Некорректный формат кэша: {cache_path}")
    persons = [GlobalPerson.model_validate(item) for item in persons_data]
    meta = {
        "generated_at": data.get("generated_at"),
        "conf_threshold": data.get("conf_threshold"),
        "match_workers": data.get("match_workers"),
        "total_clusters": data.get("total_clusters", len(persons)),
        "use_match_analysis": data.get("use_match_analysis", False),
    }
    return persons, meta


def load_or_cluster_global_persons(
    conf_threshold: float = 0.8,
    match_workers: int = 8,
    use_match_analysis: bool = False,
    cache_path: Path = CLUSTERS_CACHE_PATH,
    strict_match: bool = False,
    ) -> List[GlobalPerson]:
    try:
        persons, meta = load_clusters_from_cache(cache_path)
        cached_conf = meta.get("conf_threshold")
        cached_analysis = bool(meta.get("use_match_analysis"))
        mismatches: List[str] = []
        if cached_conf is not None and abs(cached_conf - conf_threshold) > 1e-6:
            mismatches.append(
                f"conf_threshold={cached_conf} (запрошено {conf_threshold})"
                )
        if cached_analysis != use_match_analysis:
            mismatches.append(
                f"use_match_analysis={cached_analysis} (запрошено {use_match_analysis})"
                )

        if mismatches:
            msg = "; ".join(mismatches)
            if strict_match:
                print(f"[yellow]Несовпадение параметров кэша ({msg}) — пересчитываем.[/yellow]")
                return cluster_people(
                    conf_threshold=conf_threshold,
                    match_workers=match_workers,
                    use_match_analysis=use_match_analysis,
                    )
            print(f"[yellow]Предупреждение: используем кэш с параметрами {msg}.[/yellow]")

        print(
            f"[green]Используем кэш кластеров:[/green] "
            f"{cache_path.relative_to(settings.project_root)} "
            f"(clusters={meta.get('total_clusters')})"
            )
        return persons
    except FileNotFoundError:
        print("[yellow]Кэш кластеров не найден — запускаем кластеризацию.[/yellow]")
    return cluster_people(
        conf_threshold=conf_threshold,
        match_workers=match_workers,
        use_match_analysis=use_match_analysis,
        )

from .embedding_index import (
    build_neighbor_pairs,
    build_or_load_index,
    load_embedding_config,
    log_index_stats,
    query_top_k,
    )
from .person_candidates import load_person_candidates
from .person_matcher import match_candidates_async
from .person_blocking import build_blocked_groups, load_blocking_config


def normalize_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"[^a-zа-я0-9]", "", value.lower())


def normalize_forms(forms: List[str]) -> set[str]:
    return {token for token in (normalize_token(f) for f in forms) if token}


def _meaningful_form_overlap(
    forms1: set[str],
    forms2: set[str],
    c1: PersonCandidate,
    c2: PersonCandidate,
    ) -> set[str]:
    overlap = forms1 & forms2
    if not overlap:
        return set()
    last1 = _normalized_last_name(c1)
    last2 = _normalized_last_name(c2)
    meaningful: set[str] = set()
    for token in overlap:
        if (last1 and last1 in token) or (last2 and last2 in token):
            meaningful.add(token)
    return meaningful


@dataclass
class PairFilterConfig:
    max_year_diff: int = 25
    min_pair_score: float = 1.5


@dataclass
class PairFilterStats:
    low_confidence: int = 0
    year_gap: int = 0
    weak_signal: int = 0

    def total(self) -> int:
        return self.low_confidence + self.year_gap + self.weak_signal


def load_pair_filter_config(config_path: Optional[Path] = None) -> PairFilterConfig:
    path = config_path or (settings.project_root / "config.yaml")
    config = PairFilterConfig()
    if not path.exists():
        return config

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    matching = data.get("matching", {}) if isinstance(data, dict) else {}
    if not isinstance(matching, dict):
        return config

    max_year = matching.get("max_year_diff")
    if isinstance(max_year, int) and max_year >= 0:
        config.max_year_diff = max_year

    min_score = matching.get("min_pair_score")
    if isinstance(min_score, (int, float)) and min_score > 0:
        config.min_pair_score = float(min_score)

    return config


def _normalized_initial(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return normalize_token(value[0])


def _normalized_last_name(candidate: PersonCandidate) -> Optional[str]:
    if candidate.normalized_last_name:
        normalized = normalize_token(candidate.normalized_last_name)
        if normalized:
            return normalized
    value = normalize_token(candidate.name_parts.last_name)
    if value:
        return value
    if candidate.canonical_name_in_note:
        parts = [normalize_token(p) for p in re.split(r"[\s,/]+", candidate.canonical_name_in_note) if p]
        parts = [p for p in parts if p]
        if parts:
            return parts[-1]
    return None


def _normalized_first_name(candidate: PersonCandidate) -> Optional[str]:
    return normalize_token(candidate.name_parts.first_name)


def _normalized_patronymic(candidate: PersonCandidate) -> Optional[str]:
    return normalize_token(candidate.name_parts.patronymic)


def _canonical_tokens(candidate: PersonCandidate, last_name: Optional[str]) -> set[str]:
    tokens: set[str] = set()
    if not candidate.canonical_name_in_note:
        return tokens

    for raw in re.split(r"[\s,/]+", candidate.canonical_name_in_note):
        token = normalize_token(raw)
        if not token:
            continue
        if last_name and token == last_name:
            continue
        tokens.add(token)
    return tokens


def _name_similarity_score(
    full_left: Optional[str],
    full_right: Optional[str],
    init_left: Optional[str],
    init_right: Optional[str],
    ) -> float:
    if full_left and full_right:
        if full_left == full_right:
            return 1.0
        if full_left[0] == full_right[0]:
            return 0.5
        return 0.0

    left_initial = _normalized_initial(full_left) or _normalized_initial(init_left)
    right_initial = _normalized_initial(full_right) or _normalized_initial(init_right)
    if left_initial and right_initial and left_initial == right_initial:
        return 0.35
    return 0.0


def _pair_similarity_score(
    c1: PersonCandidate,
    c2: PersonCandidate,
    config: PairFilterConfig,
    ) -> float:
    score = 0.0
    last1 = _normalized_last_name(c1)
    last2 = _normalized_last_name(c2)

    if last1 and last2:
        if last1 != last2:
            return 0.0
        score += 1.0
    elif last1 or last2:
        score += 0.25

    score += _name_similarity_score(
        _normalized_first_name(c1),
        _normalized_first_name(c2),
        c1.first_initial,
        c2.first_initial,
        )
    score += _name_similarity_score(
        _normalized_patronymic(c1),
        _normalized_patronymic(c2),
        c1.patronymic_initial,
        c2.patronymic_initial,
        )

    forms1 = normalize_forms(c1.surface_forms)
    forms2 = normalize_forms(c2.surface_forms)
    if forms1 and forms2:
        overlap = _meaningful_form_overlap(forms1, forms2, c1, c2)
        if overlap:
            score += 0.75 if len(overlap) > 1 else 0.5

    canon1 = _canonical_tokens(c1, last1)
    canon2 = _canonical_tokens(c2, last2)
    if canon1 and canon2:
        overlap = canon1 & canon2
        if overlap:
            score += 1.0 if len(overlap) > 1 else 0.5

    if c1.note_year_context and c2.note_year_context and config.max_year_diff > 0:
        diff = abs(c1.note_year_context - c2.note_year_context)
        if diff <= 5:
            score += 0.75
        elif diff <= config.max_year_diff:
            score += 0.5

    return score


def should_consider_for_llm(
    c1: PersonCandidate,
    c2: PersonCandidate,
    config: PairFilterConfig,
    stats: PairFilterStats,
    ) -> bool:
    low_buckets = {"low", "very_low"}
    if c1.confidence_bucket in low_buckets and c2.confidence_bucket in low_buckets:
        stats.low_confidence += 1
        return False

    if config.max_year_diff and c1.note_year_context and c2.note_year_context:
        if abs(c1.note_year_context - c2.note_year_context) > config.max_year_diff:
            stats.year_gap += 1
            return False

    pair_score = _pair_similarity_score(c1, c2, config)
    if pair_score < config.min_pair_score:
        stats.weak_signal += 1
        return False

    return True


def cheap_decision(c1: PersonCandidate, c2: PersonCandidate) -> Optional[str]:
    np1 = c1.name_parts
    np2 = c2.name_parts

    def normalized_full_match() -> bool:
        return bool(
            c1.normalized_full_name
            and c2.normalized_full_name
            and c1.normalized_full_name == c2.normalized_full_name
        )

    def first_last_match() -> bool:
        return bool(
            np1.last_name
            and np2.last_name
            and np1.last_name == np2.last_name
            and np1.first_name
            and np2.first_name
            and np1.first_name == np2.first_name
        )

    full_name_match = normalized_full_match()
    strong_name_match = full_name_match or first_last_match()

    if c1.role and c2.role:
        role1 = c1.role.strip().lower()
        role2 = c2.role.strip().lower()
        if role1 and role2:
            if role1 == role2 and strong_name_match:
                return "same_person"
            if role1 != role2 and not full_name_match:
                return "different_person"

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

    if full_name_match:
        return "same_person"

    forms1 = normalize_forms(c1.surface_forms)
    forms2 = normalize_forms(c2.surface_forms)
    if forms1 and forms2:
        overlap = _meaningful_form_overlap(forms1, forms2, c1, c2)
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
    use_match_analysis: bool = False,
    ) -> List[GlobalPerson]:
    candidates = load_person_candidates()
    if not candidates:
        raise RuntimeError("Нет кандидатов (запусти scan-people).")

    id_to_index = {c.candidate_id: idx for idx, c in enumerate(candidates)}
    id_to_candidate = {c.candidate_id: c for c in candidates}
    uf = DSU(len(candidates))

    print(f"[bold]Всего кандидатов:[/bold] {len(candidates)}")

    embedding_config = load_embedding_config()
    pairs: list[tuple[PersonCandidate, PersonCandidate]] = []

    if embedding_config.top_k > 0:
        print(
            "[bold]Строим индекс эмбеддингов...[/bold] "
            f"(model={embedding_config.model_name}, top_k={embedding_config.top_k})"
            )
        embeddings, index = build_or_load_index(candidates, embedding_config)
        neighbors = query_top_k(
            embeddings,
            index,
            embedding_config.top_k,
            embedding_config.index_type,
            )
        log_index_stats(index, embedding_config.index_type, neighbors)
        pairs_set = build_neighbor_pairs(
            [c.candidate_id for c in candidates],
            neighbors,
            )
        pairs = [(id_to_candidate[a], id_to_candidate[b]) for a, b in pairs_set]
    else:
        blocking_config = load_blocking_config()
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
    print(f"[bold]Потенциальных пар после фильтрации:[/bold] {len(pairs)}")
    print(f"[bold]Отсечено пар:[/bold] {filtered_ratio:.2%}")

    print("[bold]Запускаем алгоритмический матчинг...[/bold]")

    pair_filter_config = load_pair_filter_config()
    filter_stats = PairFilterStats()
    same_person_links = 0
    compared_pairs = 0
    llm_pairs: list[tuple[PersonCandidate, PersonCandidate]] = []
    if pairs:
        with tqdm(total=len(pairs), desc="Алгоритмический матчинг", unit="pair", leave=True) as pbar:
            for c1, c2 in pairs:
                compared_pairs += 1
                decision = cheap_decision(c1, c2)
                if decision == "same_person":
                    uf.union(id_to_index[c1.candidate_id], id_to_index[c2.candidate_id])
                    same_person_links += 1
                elif decision is None:
                    if should_consider_for_llm(c1, c2, pair_filter_config, filter_stats):
                        llm_pairs.append((c1, c2))
                pbar.update(1)
    else:
        print("[yellow]Нет пар для сравнения.[/yellow]")

    if filter_stats.total():
        print(
            "[bold]Пропущено пар до LLM:[/bold] "
            f"{filter_stats.total()} "
            f"(low_confidence={filter_stats.low_confidence}, "
            f"year_gap={filter_stats.year_gap}, "
            f"weak_signal={filter_stats.weak_signal})"
            )

    print(
        f"[green]Готово:[/green] проверено {compared_pairs} пар, совпадений {same_person_links}."
        )

    llm_links = 0
    if llm_pairs:
        print(f"[bold]Запускаем LLM-матчинг:[/bold] {len(llm_pairs)} пар.")
        sem = asyncio.Semaphore(max(1, match_workers))

        async def _match_pair(c1: PersonCandidate, c2: PersonCandidate):
            async with sem:
                try:
                    decision = await match_candidates_async(
                        c1,
                        c2,
                        use_analysis=use_match_analysis,
                        )
                except APITimeoutError as exc:
                    _log_match_timeout(c1, c2, exc)
                    print(
                        f"[yellow]LLM timeout для пары "
                        f"{c1.candidate_id} vs {c2.candidate_id}, "
                        "fallback → different_person[/yellow]"
                        )
                    decision = PersonMatchDecision(
                        left_candidate_id=c1.candidate_id,
                        right_candidate_id=c2.candidate_id,
                        relation="different_person",
                        confidence=0.0,
                    )
                return c1, c2, decision

        tasks = [asyncio.create_task(_match_pair(c1, c2)) for c1, c2 in llm_pairs]
        with tqdm(total=len(tasks), desc="LLM матчинг", unit="pair", leave=True) as pbar:
            for task in asyncio.as_completed(tasks):
                c1, c2, decision = await task
                if decision.relation == "same_person" and decision.confidence >= conf_threshold:
                    uf.union(id_to_index[c1.candidate_id], id_to_index[c2.candidate_id])
                    llm_links += 1
                pbar.update(1)

    if llm_pairs:
        print(f"[green]LLM-матчинг завершён:[/green] совпадений {llm_links}.")

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


def cluster_people(
    conf_threshold: float = 0.8,
    match_workers: int = 8,
    use_match_analysis: bool = False,
    ) -> List[GlobalPerson]:
    persons = asyncio.run(
        cluster_people_async(
            conf_threshold=conf_threshold,
            match_workers=match_workers,
            use_match_analysis=use_match_analysis,
            )
        )
    save_clusters_to_cache(
        persons,
        conf_threshold=conf_threshold,
        match_workers=match_workers,
        use_match_analysis=use_match_analysis,
        )
    return persons
