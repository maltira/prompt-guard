"""эмбеддинги текстов локальной моделью с дисковым кэшем (аналог llm_cache для эмбеддингов)"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data.schema import read_jsonl

CACHE_DIR = Path("results/cache/embeddings")
SPLITS_DIR = Path("data/splits")


class Encoder:
    def __init__(self, model_name: str, prefix: str | None = None):
        self.model_name = model_name
        # e5 требует префикс query: перед каждым текстом, у bge-m3 и большинства других моделей префикса нет
        self.prefix = prefix if prefix is not None else ("query: " if "e5" in model_name.lower() else "")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                # модель уже в кэше: грузим без обращения к сети, это быстрее и не зависит от связи
                self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
            except OSError:
                # первый запуск: модели ещё нет в кэше, скачиваем
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            [self.prefix + t for t in texts], batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype(np.float32)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_embeddings(encoder, split: str, splits_dir: Path = SPLITS_DIR,
                      cache_dir: Path = CACHE_DIR) -> np.ndarray | None:
    """эмбеддинги выборки из кэша или None, если кэша нет или он устарел (ничего не считает).

    кэш считается устаревшим, если изменились модель, префикс или файл выборки
    """
    sha = _file_sha256(splits_dir / f"{split}.jsonl")
    short = encoder.model_name.split("/")[-1]
    npy, meta = cache_dir / f"{short}__{split}.npy", cache_dir / f"{short}__{split}.json"
    if npy.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if (m["split_sha256"], m["prefix"], m["model"]) == (sha, encoder.prefix, encoder.model_name):
            return np.load(npy)
    return None


def embed_split(encoder, split: str, splits_dir: Path = SPLITS_DIR,
                cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """матрица эмбеддингов выборки, строки в порядке записей файла; при отсутствии кэша считает"""
    cached = cached_embeddings(encoder, split, splits_dir, cache_dir)
    if cached is not None:
        return cached
    src = splits_dir / f"{split}.jsonl"
    sha = _file_sha256(src)
    short = encoder.model_name.split("/")[-1]
    npy, meta = cache_dir / f"{short}__{split}.npy", cache_dir / f"{short}__{split}.json"
    recs = read_jsonl(src)
    t0 = time.time()
    vecs = encoder.encode([r.text for r in recs])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    meta.write_text(json.dumps({
        "model": encoder.model_name, "prefix": encoder.prefix, "split": split,
        "split_sha256": sha, "n": len(recs), "dim": int(vecs.shape[1]),
        "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return vecs
