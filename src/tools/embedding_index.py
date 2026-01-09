from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import yaml
from rich import print
from tqdm.auto import tqdm

from ..config import settings
from ..models import PersonCandidate


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SUPPORTED_INDEX_TYPES = {"hnswlib", "annoy", "faiss"}


@dataclass
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 32
    top_k: int = 12
    index_type: str = "hnswlib"
    use_mps: bool = False
    cache_path: Path = settings.project_root / "cache" / "embeddings"


def _resolve_cache_path(value: str | Path) -> Path:
    if isinstance(value, Path):
        path = value
    else:
        path = Path(value)
    if not path.is_absolute():
        return settings.project_root / path
    return path


def load_embedding_config(config_path: Optional[Path] = None) -> EmbeddingConfig:
    path = config_path or (settings.project_root / "config.yaml")
    config = EmbeddingConfig()
    if not path.exists():
        return config

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    embedding = data.get("embedding", {}) if isinstance(data, dict) else {}
    if not isinstance(embedding, dict):
        return config

    model_name = embedding.get("model_name")
    if isinstance(model_name, str) and model_name:
        config.model_name = model_name

    batch_size = embedding.get("batch_size")
    if isinstance(batch_size, int) and batch_size > 0:
        config.batch_size = batch_size

    top_k = embedding.get("top_k")
    if isinstance(top_k, int) and top_k > 0:
        config.top_k = top_k

    index_type = embedding.get("index_type")
    if isinstance(index_type, str) and index_type in SUPPORTED_INDEX_TYPES:
        config.index_type = index_type

    use_mps = embedding.get("use_mps")
    if isinstance(use_mps, bool):
        config.use_mps = use_mps

    cache_path = embedding.get("cache_path")
    if isinstance(cache_path, str) and cache_path:
        config.cache_path = _resolve_cache_path(cache_path)

    return config


def _profile_year_info(candidate: PersonCandidate) -> Optional[str]:
    if candidate.note_year_context:
        return str(candidate.note_year_context)
    return None


def build_person_profile_text(candidate: PersonCandidate) -> str:
    name_parts = candidate.name_parts
    parts: List[str] = []

    if candidate.canonical_name_in_note:
        parts.append(f"Канон: {candidate.canonical_name_in_note}")

    if candidate.normalized_full_name:
        parts.append(f"Нормализованное ФИО: {candidate.normalized_full_name}")

    names = [name for name in [name_parts.last_name, name_parts.first_name, name_parts.patronymic] if name]
    if names:
        parts.append(f"Части имени: {' '.join(names)}")

    if candidate.surface_forms:
        unique_forms = list(dict.fromkeys(candidate.surface_forms))
        parts.append("Формы: " + ", ".join(unique_forms))

    if candidate.role:
        parts.append(f"Роль: {candidate.role}")

    year_info = _profile_year_info(candidate)
    if year_info:
        parts.append(f"Год: {year_info}")

    if candidate.snippet_preview:
        parts.append(f"Сниппет: {candidate.snippet_preview}")

    return "\n".join(parts)


def _load_embedding_model(model_name: str, use_mps: bool):
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        raise RuntimeError(
            "Не удалось импортировать transformers/torch. "
            "Установи зависимости из pyproject.toml."
            )

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cpu"
    if use_mps and torch.backends.mps.is_available():
        device = "mps"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, device


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * input_mask_expanded).sum(dim=1)
    counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_profiles(profiles: Sequence[str], config: EmbeddingConfig) -> np.ndarray:
    if not profiles:
        return np.zeros((0, 0), dtype=np.float32)

    tokenizer, model, device = _load_embedding_model(config.model_name, config.use_mps)
    import torch

    embeddings: List[np.ndarray] = []
    batch_size = max(1, config.batch_size)

    with torch.no_grad():
        progress = tqdm(
            total=len(profiles),
            desc="Получение эмбеддингов",
            unit="profile",
            leave=True,
            )
        try:
            for start in range(0, len(profiles), batch_size):
                batch = profiles[start:start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model_output = model(**encoded)
                pooled = _mean_pooling(model_output, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy().astype(np.float32))
                progress.update(len(batch))
        finally:
            progress.close()

    return np.vstack(embeddings)


def encode_profiles(profiles: List[str]) -> np.ndarray:
    config = load_embedding_config()
    return _encode_profiles(profiles, config)


def _slugify(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _cache_paths(config: EmbeddingConfig) -> dict[str, Path]:
    config.cache_path.mkdir(parents=True, exist_ok=True)
    slug = _slugify(config.model_name)
    return {
        "embeddings": config.cache_path / f"embeddings_{slug}.npy",
        "meta": config.cache_path / f"embeddings_{slug}.json",
        "index": config.cache_path / f"index_{slug}.{config.index_type}",
        }


def _load_cached_embeddings(
    candidates: Sequence[PersonCandidate],
    config: EmbeddingConfig,
    ) -> Optional[np.ndarray]:
    paths = _cache_paths(config)
    if not (paths["embeddings"].exists() and paths["meta"].exists()):
        return None

    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    candidate_ids = [c.candidate_id for c in candidates]
    if meta.get("model_name") != config.model_name:
        return None
    if meta.get("candidate_ids") != candidate_ids:
        return None
    if meta.get("index_type") != config.index_type:
        return None

    return np.load(paths["embeddings"], allow_pickle=False)


def _save_embeddings_cache(
    candidates: Sequence[PersonCandidate],
    embeddings: np.ndarray,
    config: EmbeddingConfig,
    ) -> None:
    paths = _cache_paths(config)
    meta = {
        "model_name": config.model_name,
        "index_type": config.index_type,
        "candidate_ids": [c.candidate_id for c in candidates],
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(paths["embeddings"], embeddings)


def _build_hnsw_index(embeddings: np.ndarray):
    if importlib.util.find_spec("hnswlib") is None:
        raise RuntimeError("hnswlib не установлен. Добавь его в окружение.")

    import hnswlib

    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=16)
    index.add_items(embeddings, np.arange(embeddings.shape[0]))
    index.set_ef(64)
    return index


def _build_annoy_index(embeddings: np.ndarray):
    if importlib.util.find_spec("annoy") is None:
        raise RuntimeError("annoy не установлен. Добавь его в окружение.")

    from annoy import AnnoyIndex

    dim = embeddings.shape[1]
    index = AnnoyIndex(dim, "angular")
    for idx, vector in enumerate(embeddings):
        index.add_item(idx, vector)
    index.build(20)
    return index


def build_ann_index(embeddings: np.ndarray, index_type: str = "hnswlib"):
    if index_type == "hnswlib":
        return _build_hnsw_index(embeddings)
    if index_type == "annoy":
        return _build_annoy_index(embeddings)
    if index_type == "faiss":
        raise RuntimeError("faiss не подключен в этом проекте. Используй hnswlib или annoy.")
    raise ValueError(f"Неизвестный index_type: {index_type}")


def _save_index(index, config: EmbeddingConfig, dim: int) -> None:
    paths = _cache_paths(config)
    if config.index_type == "hnswlib":
        index.save_index(str(paths["index"]))
        return
    if config.index_type == "annoy":
        index.save(str(paths["index"]))
        return
    if config.index_type == "faiss":
        raise RuntimeError("faiss не подключен в этом проекте.")
    raise ValueError(f"Неизвестный index_type: {config.index_type}")


def _load_index(config: EmbeddingConfig, dim: int):
    paths = _cache_paths(config)
    if not paths["index"].exists():
        return None

    if config.index_type == "hnswlib":
        if importlib.util.find_spec("hnswlib") is None:
            raise RuntimeError("hnswlib не установлен.")
        import hnswlib
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(str(paths["index"]))
        index.set_ef(64)
        return index
    if config.index_type == "annoy":
        if importlib.util.find_spec("annoy") is None:
            raise RuntimeError("annoy не установлен.")
        from annoy import AnnoyIndex
        index = AnnoyIndex(dim, "angular")
        if not index.load(str(paths["index"])):
            return None
        return index
    if config.index_type == "faiss":
        raise RuntimeError("faiss не подключен в этом проекте.")
    raise ValueError(f"Неизвестный index_type: {config.index_type}")


def build_or_load_index(
    candidates: Sequence[PersonCandidate],
    config: EmbeddingConfig,
    ):
    start = time.perf_counter()
    embeddings = _load_cached_embeddings(candidates, config)

    if embeddings is None:
        profiles = [build_person_profile_text(c) for c in candidates]
        embeddings = _encode_profiles(profiles, config)
        _save_embeddings_cache(candidates, embeddings, config)
        print(
            "[green]Эмбеддинги построены[/green] "
            f"({len(candidates)} профилей за {time.perf_counter() - start:.2f} сек)."
            )
    else:
        print("[green]Эмбеддинги загружены из кэша.[/green]")

    index = _load_index(config, embeddings.shape[1])
    if index is None:
        index_start = time.perf_counter()
        index = build_ann_index(embeddings, config.index_type)
        _save_index(index, config, embeddings.shape[1])
        print(
            "[green]Индекс построен[/green] "
            f"({time.perf_counter() - index_start:.2f} сек)."
            )
    else:
        print("[green]Индекс загружен из кэша.[/green]")

    return embeddings, index


def _index_size(index_type: str, index) -> int:
    if index_type == "hnswlib":
        return index.get_current_count()
    if index_type == "annoy":
        return index.get_n_items()
    if index_type == "faiss":
        return index.ntotal
    return 0


def query_top_k(
    embeddings: np.ndarray,
    index,
    top_k: int,
    index_type: str,
    ) -> List[List[int]]:
    if embeddings.size == 0:
        return []
    k = min(top_k + 1, embeddings.shape[0])

    if index_type == "hnswlib":
        labels, _ = index.knn_query(embeddings, k=k)
        return labels.tolist()
    if index_type == "annoy":
        neighbors: List[List[int]] = []
        for idx in range(embeddings.shape[0]):
            neighbors.append(index.get_nns_by_item(idx, k))
        return neighbors
    if index_type == "faiss":
        distances, labels = index.search(embeddings, k)
        return labels.tolist()
    raise ValueError(f"Неизвестный index_type: {index_type}")


def build_neighbor_pairs(
    candidate_ids: Sequence[str],
    neighbors: Sequence[Sequence[int]],
    ) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for idx, neighbor_indices in enumerate(neighbors):
        for neighbor_idx in neighbor_indices:
            if neighbor_idx == idx:
                continue
            left = candidate_ids[idx]
            right = candidate_ids[neighbor_idx]
            pair = tuple(sorted((left, right)))
            pairs.add(pair)
    return pairs


def log_index_stats(
    index,
    index_type: str,
    neighbors: Sequence[Sequence[int]],
    ) -> None:
    total_neighbors = 0
    for idx, neighbor_indices in enumerate(neighbors):
        total_neighbors += sum(1 for n in neighbor_indices if n != idx)
    avg_neighbors = total_neighbors / len(neighbors) if neighbors else 0.0
    print(f"[bold]Размер индекса:[/bold] {_index_size(index_type, index)}")
    print(f"[bold]Среднее число соседей:[/bold] {avg_neighbors:.2f}")
