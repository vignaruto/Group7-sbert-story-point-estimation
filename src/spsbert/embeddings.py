"""Sentence-embedding backends with an on-disk cache.

* SBERTEncoder  - the real thing (sentence-transformers); used for all reported results.
* HashingEncoder - deterministic bag-of-words hashing, ONLY for offline smoke tests (never SBERT!).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import EmbeddingConfig
from .utils import resolve_device, sha1_text

LOGGER = logging.getLogger("spsbert.embeddings")


class Encoder(Protocol):
    name: str

    def encode(self, texts: list) -> np.ndarray: ...


class HashingEncoder:
    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(n_features=dim, alternate_sign=True, norm="l2", ngram_range=(1, 2))

    def encode(self, texts: list) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


class SBERTEncoder:
    def __init__(self, cfg: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e
        device = resolve_device(cfg.device)
        self.name = cfg.model_name
        self.model = SentenceTransformer(cfg.model_name, device=device)
        if cfg.max_seq_length:
            self.model.max_seq_length = int(cfg.max_seq_length)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)   # library may reset its level on import
        self.batch_size = cfg.batch_size
        self.normalize = cfg.normalize
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension  # renamed in newer versions
        LOGGER.info("Loaded SBERT '%s' on %s (dim=%s, max_seq_length=%s)", cfg.model_name, device, dim_fn(), self.model.max_seq_length)

    def encode(self, texts: list) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


def build_encoder(cfg: EmbeddingConfig) -> Encoder:
    if cfg.backend == "hashing":
        LOGGER.warning("Using the HASHING encoder: smoke-test only, results are NOT SBERT results.")
        return HashingEncoder(cfg.hashing_dim)
    return SBERTEncoder(cfg)


class EmbeddingStore:
    """Encode once, cache forever: key = hash(encoder settings, all texts)."""

    def __init__(self, encoder: Encoder, cfg: EmbeddingConfig):
        self.encoder, self.cfg = encoder, cfg
        self.dir = Path(cfg.cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, texts: list) -> np.ndarray:
        settings = f"{self.cfg.backend}|{self.encoder.name}|{self.cfg.max_seq_length}|{self.cfg.normalize}|{self.cfg.hashing_dim}"
        key = sha1_text(settings, *texts)
        path = self.dir / f"{key}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return arr
            LOGGER.warning("Stale cache %s ignored", path.name)
        arr = self.encoder.encode(texts)
        if arr.shape[0] != len(texts) or not np.isfinite(arr).all():
            raise RuntimeError("Encoder returned wrong shape or non-finite values")
        np.save(path, arr)
        return arr
