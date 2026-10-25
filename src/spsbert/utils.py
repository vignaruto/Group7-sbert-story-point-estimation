"""Shared helpers: seeding, logging, JSON I/O, hashing, environment capture."""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Required for deterministic CUDA matmuls; must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # no download progress-bar spam in notebooks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")          # hides the harmless "unauthenticated requests" notice; set HF_TOKEN for faster downloads

LOGGER = logging.getLogger("spsbert")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for non-CNN usage
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@functools.lru_cache(maxsize=1)
def cuda_ok() -> bool:
    """True only if a GPU exists AND a tiny convolution really runs on it (e.g. Kaggle P100 vs new PyTorch builds can fail)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.nn.Conv1d(1, 2, 3).cuda()(torch.randn(4, 1, 32, device="cuda")).sum().item()
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_device(pref: str = "auto") -> str:
    """'cuda' if requested/auto and genuinely usable, otherwise 'cpu' (never crashes on a broken GPU)."""
    if pref in ("auto", "cuda") and cuda_ok():
        return "cuda"
    if pref == "cuda":
        LOGGER.warning("cuda requested but not usable -> using cpu")
    return "cpu"


def slugify(name: str) -> str:
    """Filesystem-safe slug ('C++ driver (X)' -> 'Cpp_driver_X')."""
    s = name.replace("+", "p")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "project"


def sha1_text(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            return None if np.isnan(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _nan_to_none(o: Any) -> Any:
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, np.ndarray):
        return _nan_to_none(o.tolist())
    if isinstance(o, np.floating):
        return _nan_to_none(float(o))
    return o


def write_json(obj: Any, path: str | Path) -> None:
    """Atomic JSON write (tmp + rename) so interrupted runs never leave corrupt files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(obj), f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(level: str = "INFO", logfile: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "sentence_transformers", "matplotlib", "PIL", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)   # keep our own INFO lines readable


def environment_info() -> dict[str, Any]:
    """Versions needed to reproduce a run."""
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "sentence_transformers", "transformers", "yaml"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - optional dependency
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info
