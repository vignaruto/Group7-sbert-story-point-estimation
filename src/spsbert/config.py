"""Typed, validated configuration (YAML -> dataclasses). Unknown keys raise (catches typos)."""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .utils import sha1_text

ALL_METHODS = ["mean_baseline", "median_baseline", "tfidf_ridge", "sbert_ridge", "sbert_cnn"]


@dataclass
class DataConfig:
    csv_path: str = "data/tawos_issues.csv"
    projects_file: Optional[str] = None      # text file, one project name per line
    projects: Optional[list] = None          # explicit list (overrides projects_file)
    aliases_file: Optional[str] = None       # csv mapping the assignment's short project names to TAWOS names
    min_issues: int = 100                    # projects with fewer usable issues are skipped
    max_story_point: Optional[float] = None  # optional outlier cap (None = keep all)
    deduplicate: bool = True                 # drop exact duplicate texts within a project


@dataclass
class SplitConfig:
    test_size: float = 0.20
    n_folds: int = 5


@dataclass
class EmbeddingConfig:
    backend: str = "sbert"                   # "sbert" | "hashing" (offline smoke tests only)
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 64
    normalize: bool = True
    max_seq_length: Optional[int] = 384
    cache_dir: str = "cache/embeddings"
    device: str = "auto"                     # auto | cpu | cuda
    hashing_dim: int = 256


@dataclass
class CNNConfig:
    channels: list = field(default_factory=lambda: [16, 32, 64])  # three conv layers
    kernel_size: int = 5
    pool_size: int = 2
    adaptive_pool: int = 0                   # 0 = head sees the whole feature map
    pool_type: str = "avg"                   # avg | max
    fc_hidden: int = 64
    target_transform: str = "log1p"          # log1p | none
    loss: str = "huber"                      # huber | l1 | mse
    standardize_inputs: bool = True
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    inner_val_fraction: float = 0.10         # early-stopping hold-out taken from the TRAIN part only
    grid: dict = field(
        default_factory=lambda: {"lr": [1e-3], "dropout": [0.1, 0.3, 0.5], "weight_decay": [1e-4, 1e-2]}
    )
    final_seeds: list = field(default_factory=lambda: [42, 43, 44])
    device: str = "auto"

    def __post_init__(self) -> None:
        # partial grid overrides (e.g. only 'lr') keep the defaults for the other keys
        defaults = {"lr": [1e-3], "dropout": [0.1, 0.3, 0.5], "weight_decay": [1e-4, 1e-2]}
        self.grid = {**defaults, **self.grid}


@dataclass
class RidgeConfig:
    alphas: list = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    tfidf_max_features: int = 20000
    tfidf_ngram: list = field(default_factory=lambda: [1, 2])
    tfidf_alphas: list = field(default_factory=lambda: [0.1, 1.0, 10.0])


@dataclass
class EvalConfig:
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    methods: list = field(default_factory=lambda: list(ALL_METHODS))


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _build(cls, d or {})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Hash of everything that affects numerical results (not output/data locations)."""
        d = self.to_dict()
        d.pop("output_dir", None)
        d["data"].pop("csv_path", None)
        d["embedding"].pop("cache_dir", None)
        return sha1_text(yaml.safe_dump(d, sort_keys=True))[:16]

    def validate(self) -> "Config":
        if not 0.0 < self.split.test_size < 1.0:
            raise ValueError("split.test_size must be in (0, 1)")
        if self.split.n_folds < 2:
            raise ValueError("split.n_folds must be >= 2")
        if self.embedding.backend not in ("sbert", "hashing"):
            raise ValueError("embedding.backend must be 'sbert' or 'hashing'")
        if len(self.cnn.channels) != 3:
            raise ValueError("cnn.channels must list exactly 3 values (three-layer CNN)")
        if self.cnn.target_transform not in ("log1p", "none"):
            raise ValueError("cnn.target_transform must be 'log1p' or 'none'")
        if self.cnn.pool_type not in ("max", "avg"):
            raise ValueError("cnn.pool_type must be 'max' or 'avg'")
        if self.cnn.loss not in ("huber", "l1", "mse"):
            raise ValueError("cnn.loss must be huber|l1|mse")
        if not 0.0 < self.cnn.inner_val_fraction < 0.5:
            raise ValueError("cnn.inner_val_fraction must be in (0, 0.5)")
        for key in ("lr", "dropout", "weight_decay"):
            if key not in self.cnn.grid or not self.cnn.grid[key]:
                raise ValueError(f"cnn.grid must define a non-empty list for '{key}'")
        if not self.cnn.final_seeds:
            raise ValueError("cnn.final_seeds must not be empty")
        bad = [m for m in self.evaluation.methods if m not in ALL_METHODS]
        if bad:
            raise ValueError(f"unknown methods {bad}; choose from {ALL_METHODS}")
        if self.data.min_issues < 4 * self.split.n_folds:
            raise ValueError("data.min_issues is too small for the requested folds")
        return self


def _build(cls: Any, d: dict) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in d.items():
        f = fields[k]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default) and isinstance(v, dict):
            kwargs[k] = _build(type(default), v)
        else:
            kwargs[k] = copy.deepcopy(v)
    return cls(**kwargs)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Any = None, overrides: Any = None) -> Config:
    """Load YAML, then apply 'a.b.c=value' overrides (value parsed as YAML)."""
    raw: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got '{item}'")
        key, value = item.split("=", 1)
        node: dict = {}
        cur = node
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = yaml.safe_load(value)
        _deep_update(raw, node)
    return Config.from_dict(raw).validate()
