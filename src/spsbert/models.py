"""Regressors and the method registry (strategy pattern).

Every method exposes `fit(X, y) -> self` and `predict(X) -> np.ndarray`, and is built by a factory
`factory(params, seed)`, so the experiment engine treats the CNN, Ridge and baselines uniformly.
"""
from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .config import Config
from .utils import resolve_device, set_seed

LOGGER = logging.getLogger("spsbert.models")


# --------------------------------------------------------------------------- target transform
def _fwd(name: str) -> Callable:
    return np.log1p if name == "log1p" else (lambda a: a)


def _inv(name: str) -> Callable:
    return np.expm1 if name == "log1p" else (lambda a: a)


# --------------------------------------------------------------------------- trivial baselines
class ConstantRegressor:
    """Predict the training mean or median for every issue."""

    def __init__(self, strategy: str):
        assert strategy in ("mean", "median")
        self.strategy = strategy
        self.value_: Optional[float] = None

    def fit(self, X: Any, y: np.ndarray) -> "ConstantRegressor":
        y = np.asarray(y, float)
        self.value_ = float(np.mean(y) if self.strategy == "mean" else np.median(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.value_, dtype=float)


# --------------------------------------------------------------------------- sklearn wrapper
class TransformedSklearn:
    """sklearn estimator on a transformed target; predictions are inverted and clipped to the training range."""

    def __init__(self, estimator_factory: Callable, transform: str = "log1p"):
        self.factory, self.transform = estimator_factory, transform

    def fit(self, X: Any, y: np.ndarray) -> "TransformedSklearn":
        y = np.asarray(y, float)
        self.lo_, self.hi_ = float(y.min()), float(y.max())
        self.est_ = self.factory()
        self.est_.fit(X, _fwd(self.transform)(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(_inv(self.transform)(self.est_.predict(X)), self.lo_, self.hi_)


# --------------------------------------------------------------------------- 3-layer CNN
def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_cnn_module(in_dim: int, channels: list, kernel_size: int, pool_size: int,
                     adaptive_pool: int, fc_hidden: int, dropout: float, pool_type: str = "max"):
    """Three Conv1d blocks (Conv-BN-ReLU-Pool) over the embedding vector, then an MLP head.

    The 1-D embedding of length D is treated as a one-channel signal of length D. The head sees the
    whole feature map (``adaptive_pool == 0``, default) so no position information is discarded;
    ``adaptive_pool > 0`` adds a global adaptive pooling to that many positions (smaller, lossier).
    """
    torch, nn = _import_torch()
    assert len(channels) == 3, "three-layer CNN required"
    Pool = nn.MaxPool1d if pool_type == "max" else nn.AvgPool1d
    layers: list = []
    c_in, length = 1, in_dim
    for c in channels:
        layers += [nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(c),
                   nn.ReLU(), Pool(pool_size, ceil_mode=True)]
        c_in = c
        length = -(-length // pool_size)          # ceil division, matches ceil_mode=True
    if adaptive_pool and adaptive_pool > 0:
        agg = nn.AdaptiveMaxPool1d(adaptive_pool) if pool_type == "max" else nn.AdaptiveAvgPool1d(adaptive_pool)
        flat = c_in * adaptive_pool
    else:
        agg, flat = nn.Identity(), c_in * length

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(*layers)
            self.pool = agg
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat, fc_hidden),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1),
            )

        def forward(self, x):  # x: (B, D)
            return self.head(self.pool(self.features(x.unsqueeze(1)))).squeeze(-1)

    return Net()


class CNNRegressor:
    """Three-layer CNN regressor with early stopping on an inner hold-out of the TRAINING data only.

    The validation fold (or the test set) is never used for early stopping, so CV scores are unbiased
    with respect to the number of epochs.
    """

    def __init__(self, params: dict, seed: int, cfg: Config):
        self.p, self.seed, self.c = params, seed, cfg.cnn
        self.best_epoch_: Optional[int] = None
        self.n_params_: Optional[int] = None

    def _device(self):
        torch, _ = _import_torch()
        return torch.device(resolve_device(self.c.device))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNRegressor":
        torch, nn = _import_torch()
        c = self.c
        set_seed(self.seed)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, float)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(2, int(round(n * c.inner_val_fraction)))
        va_i, tr_i = perm[:n_val], perm[n_val:]
        if len(tr_i) < 4:
            raise ValueError("Too little training data for the CNN")

        self.lo_, self.hi_ = float(y[tr_i].min()), float(y[tr_i].max())
        z = _fwd(c.target_transform)(y)
        self.z_mu_, self.z_sd_ = float(z[tr_i].mean()), float(z[tr_i].std() or 1.0)
        zt = ((z - self.z_mu_) / self.z_sd_).astype(np.float32)
        if c.standardize_inputs:
            self.x_mu_ = X[tr_i].mean(0, keepdims=True)
            sd = X[tr_i].std(0, keepdims=True)
            # floor the scale so (near-)constant / sparse dimensions are not blown up
            self.x_sd_ = np.maximum(sd, 0.1 * float(np.median(sd)) + 1e-6).astype(np.float32)
        else:
            self.x_mu_, self.x_sd_ = np.zeros((1, X.shape[1]), np.float32), np.ones((1, X.shape[1]), np.float32)

        dev = self._device()
        self.dev_ = dev
        Xt = torch.from_numpy(np.clip((X - self.x_mu_) / self.x_sd_, -10, 10).astype(np.float32)).to(dev)
        yt = torch.from_numpy(zt).to(dev)
        self.net_ = build_cnn_module(X.shape[1], c.channels, c.kernel_size, c.pool_size, c.adaptive_pool,
                                     c.fc_hidden, float(self.p["dropout"]), c.pool_type).to(dev)
        self.n_params_ = int(sum(p.numel() for p in self.net_.parameters() if p.requires_grad))
        opt = torch.optim.Adam(self.net_.parameters(), lr=float(self.p["lr"]),
                               weight_decay=float(self.p["weight_decay"]))
        loss_fn = {"huber": nn.HuberLoss(delta=1.0), "l1": nn.L1Loss(), "mse": nn.MSELoss()}[c.loss]
        gen = torch.Generator().manual_seed(self.seed)
        tr_t, va_t = torch.from_numpy(tr_i), torch.from_numpy(va_i)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(c.max_epochs):
            self.net_.train()
            order = tr_t[torch.randperm(len(tr_t), generator=gen)]
            for s in range(0, len(order), c.batch_size):
                b = order[s:s + c.batch_size].to(dev)
                if len(b) < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net_(Xt[va_t.to(dev)]), yt[va_t.to(dev)]))
            if v < best - 1e-6:
                best, bad, self.best_epoch_ = v, 0, epoch + 1
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                bad += 1
                if bad >= c.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch, _ = _import_torch()
        X = np.clip((np.asarray(X, np.float32) - self.x_mu_) / self.x_sd_, -10, 10).astype(np.float32)
        self.net_.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                out.append(self.net_(torch.from_numpy(X[s:s + 1024]).to(self.dev_)).cpu().numpy())
        z = np.concatenate(out) * self.z_sd_ + self.z_mu_
        z = np.clip(z, _fwd(self.c.target_transform)(self.lo_), _fwd(self.c.target_transform)(self.hi_))
        return np.clip(_inv(self.c.target_transform)(z), self.lo_, self.hi_)


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    feature: str                     # "embedding" or "text"
    grid: list                       # list of hyper-parameter dicts
    factory: Callable                # (params, seed) -> regressor
    seeds: list                      # seeds used for the final (test) fits; >1 only if stochastic


def _product(grid: dict) -> list:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def build_method_specs(cfg: Config) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    tt = cfg.cnn.target_transform
    specs = {
        "mean_baseline": MethodSpec("mean_baseline", "Mean", "embedding", [{}],
                                    lambda p, s: ConstantRegressor("mean"), [cfg.seed]),
        "median_baseline": MethodSpec("median_baseline", "Median", "embedding", [{}],
                                      lambda p, s: ConstantRegressor("median"), [cfg.seed]),
        "tfidf_ridge": MethodSpec(
            "tfidf_ridge", "TF-IDF+Ridge", "text", [{"alpha": a} for a in cfg.ridge.tfidf_alphas],
            lambda p, s: TransformedSklearn(
                lambda: make_pipeline(
                    TfidfVectorizer(max_features=cfg.ridge.tfidf_max_features,
                                    ngram_range=tuple(cfg.ridge.tfidf_ngram), sublinear_tf=True),
                    Ridge(alpha=p["alpha"])), tt),
            [cfg.seed]),
        "sbert_ridge": MethodSpec(
            "sbert_ridge", "SBERT+Ridge", "embedding", [{"alpha": a} for a in cfg.ridge.alphas],
            lambda p, s: TransformedSklearn(lambda: Ridge(alpha=p["alpha"]), tt), [cfg.seed]),
        "sbert_cnn": MethodSpec(
            "sbert_cnn", "SBERT+CNN", "embedding", _product(cfg.cnn.grid),
            lambda p, s: CNNRegressor(p, s, cfg), list(cfg.cnn.final_seeds)),
    }
    return {m: specs[m] for m in cfg.evaluation.methods}
