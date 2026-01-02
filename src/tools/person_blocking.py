from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
import math

import yaml

from ..config import settings
from ..models import PersonCandidate
from .name_normalizer import (
    infer_name_parts_from_full_name,
    initial_from_name,
    normalize_name_for_blocking,
    )


KEY_LAST_NAME_FIRST_INITIAL = "last_name_first_initial"
KEY_LAST_NAME_PATRONYMIC_INITIAL = "last_name_patronymic_initial"
KEY_LAST_NAME_YEAR_BUCKET = "last_name_year_bucket"
KEY_LAST_NAME_FIRST_NAME = "last_name_first_name"

DEFAULT_KEYS: Dict[str, bool] = {
    KEY_LAST_NAME_FIRST_INITIAL: True,
    KEY_LAST_NAME_PATRONYMIC_INITIAL: True,
    KEY_LAST_NAME_YEAR_BUCKET: True,
    KEY_LAST_NAME_FIRST_NAME: True,
    }


@dataclass
class BlockingConfig:
    year_bucket_size: int = 10
    max_block_size: int = 200
    keys_enabled: Dict[str, bool] = field(default_factory=lambda: DEFAULT_KEYS.copy())
    fallback_key: str = KEY_LAST_NAME_FIRST_NAME


@dataclass
class BlockingStats:
    total_blocks: int = 0
    oversize_blocks: int = 0
    fallback_blocks: int = 0
    skipped_blocks: int = 0


def load_blocking_config(config_path: Optional[Path] = None) -> BlockingConfig:
    path = config_path or (settings.project_root / "config.yaml")
    config = BlockingConfig()
    if not path.exists():
        return config

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    blocking = data.get("blocking", {}) if isinstance(data, dict) else {}
    if not isinstance(blocking, dict):
        return config

    year_bucket_size = blocking.get("year_bucket_size")
    if isinstance(year_bucket_size, int) and year_bucket_size > 0:
        config.year_bucket_size = year_bucket_size

    max_block_size = blocking.get("max_block_size")
    if isinstance(max_block_size, int) and max_block_size > 0:
        config.max_block_size = max_block_size

    keys_cfg = blocking.get("keys")
    if isinstance(keys_cfg, dict):
        for key, default in DEFAULT_KEYS.items():
            value = keys_cfg.get(key)
            if isinstance(value, bool):
                config.keys_enabled[key] = value
            else:
                config.keys_enabled[key] = default

    fallback_key = blocking.get("fallback_key")
    if isinstance(fallback_key, str) and fallback_key:
        config.fallback_key = fallback_key

    return config


def _normalize_last_name(value: Optional[str]) -> Optional[str]:
    normalized = normalize_name_for_blocking(value)
    if not normalized:
        return None
    return normalized.replace(" ", "")


def _infer_parts(candidate: PersonCandidate) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if candidate.normalized_full_name:
        return infer_name_parts_from_full_name(candidate.normalized_full_name)
    return infer_name_parts_from_full_name(candidate.canonical_name_in_note)


def _candidate_name_fields(candidate: PersonCandidate) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    inferred_last, inferred_first, inferred_patronymic = _infer_parts(candidate)

    last_name = candidate.normalized_last_name or _normalize_last_name(
        candidate.name_parts.last_name or inferred_last
        )
    first_name = normalize_name_for_blocking(candidate.name_parts.first_name or inferred_first, remove_titles=False)
    patronymic = normalize_name_for_blocking(
        candidate.name_parts.patronymic or inferred_patronymic,
        remove_titles=False,
        )

    first_initial = candidate.first_initial or initial_from_name(first_name)
    patronymic_initial = candidate.patronymic_initial or initial_from_name(patronymic)

    return last_name, first_name, first_initial, patronymic_initial


def _year_bucket(year: Optional[int], bucket_size: int) -> Optional[str]:
    if not year or bucket_size <= 0:
        return None
    start = int(math.floor(year / bucket_size) * bucket_size)
    end = start + bucket_size - 1
    return f"{start}-{end}"


def generate_block_key(
    candidate: PersonCandidate,
    key_type: str,
    config: BlockingConfig,
    ) -> Optional[str]:
    last_name, first_name, first_initial, patronymic_initial = _candidate_name_fields(candidate)
    if not last_name:
        return None

    if key_type == KEY_LAST_NAME_FIRST_INITIAL:
        if not first_initial:
            return None
        return f"{last_name}|{first_initial}"

    if key_type == KEY_LAST_NAME_PATRONYMIC_INITIAL:
        if not patronymic_initial:
            return None
        return f"{last_name}|{patronymic_initial}"

    if key_type == KEY_LAST_NAME_YEAR_BUCKET:
        bucket = _year_bucket(candidate.note_year_context, config.year_bucket_size)
        if not bucket:
            return None
        return f"{last_name}|{bucket}"

    if key_type == KEY_LAST_NAME_FIRST_NAME:
        if not first_name:
            return None
        first_key = first_name.replace(" ", "")
        return f"{last_name}|{first_key}"

    return None


def build_blocks_for_key(
    candidates: Iterable[PersonCandidate],
    key_type: str,
    config: BlockingConfig,
    ) -> Dict[Tuple[str, str], List[PersonCandidate]]:
    blocks: Dict[Tuple[str, str], List[PersonCandidate]] = defaultdict(list)
    for candidate in candidates:
        key = generate_block_key(candidate, key_type, config)
        if not key:
            continue
        blocks[(key_type, key)].append(candidate)
    return blocks


def build_blocked_groups(
    candidates: List[PersonCandidate],
    config: BlockingConfig,
    ) -> Tuple[List[List[PersonCandidate]], BlockingStats]:
    stats = BlockingStats()

    base_blocks: Dict[Tuple[str, str], List[PersonCandidate]] = defaultdict(list)
    for key_type, enabled in config.keys_enabled.items():
        if not enabled:
            continue
        for block_key, members in build_blocks_for_key(candidates, key_type, config).items():
            base_blocks[block_key].extend(members)

    final_blocks: List[List[PersonCandidate]] = []

    fallback_order = [config.fallback_key, KEY_LAST_NAME_FIRST_NAME, KEY_LAST_NAME_FIRST_INITIAL]
    fallback_order = [key for key in fallback_order if key]

    for (key_type, key_value), members in base_blocks.items():
        if len(members) <= 1:
            continue
        if config.max_block_size and len(members) > config.max_block_size:
            stats.oversize_blocks += 1
            fallback_blocks = None
            for fallback_key in fallback_order:
                if fallback_key == key_type:
                    continue
                candidate_blocks = build_blocks_for_key(members, fallback_key, config)
                if candidate_blocks:
                    fallback_blocks = list(candidate_blocks.values())
                    break
            if fallback_blocks:
                stats.fallback_blocks += len(fallback_blocks)
                for block in fallback_blocks:
                    if len(block) <= 1:
                        continue
                    if config.max_block_size and len(block) > config.max_block_size:
                        stats.skipped_blocks += 1
                        continue
                    final_blocks.append(block)
            else:
                stats.skipped_blocks += 1
            continue

        final_blocks.append(members)

    stats.total_blocks = len(final_blocks)
    return final_blocks, stats
