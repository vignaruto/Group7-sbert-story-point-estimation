"""Random 80:20 development/test split + 5-fold CV on the development part (no leakage by construction)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, train_test_split


@dataclass
class ProjectSplit:
    dev_idx: np.ndarray                         # indices of the 80 % development portion
    test_idx: np.ndarray                        # indices of the 20 % held-out test portion
    folds: list                                 # list of (train_idx, val_idx), indices into the project rows

    def to_json(self, keys: list) -> dict:
        keys = np.asarray(keys)
        fold_of = {}
        for k, (_, va) in enumerate(self.folds):
            for i in va:
                fold_of[str(keys[i])] = k
        return {
            "n": int(len(self.dev_idx) + len(self.test_idx)),
            "test_issue_keys": keys[self.test_idx].tolist(),
            "dev_issue_keys": keys[self.dev_idx].tolist(),
            "validation_fold_of_dev_issue": fold_of,
        }


def make_split(n: int, test_size: float, n_folds: int, seed: int) -> ProjectSplit:
    """Deterministic split. `n` rows must already be in a canonical order (we sort by issue key)."""
    if n < 2 * n_folds + 2:
        raise ValueError(f"Too few issues ({n}) for {n_folds}-fold CV with a {test_size:.0%} test set")
    idx = np.arange(n)
    dev, test = train_test_split(idx, test_size=test_size, shuffle=True, random_state=seed)
    dev, test = np.sort(dev), np.sort(test)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [(dev[tr], dev[va]) for tr, va in kf.split(dev)]
    split = ProjectSplit(dev_idx=dev, test_idx=test, folds=folds)
    assert_valid_split(split, n)
    return split


def assert_valid_split(split: ProjectSplit, n: int) -> None:
    """Hard guarantees: disjoint dev/test, full coverage, every dev item validated exactly once."""
    dev, test = set(split.dev_idx.tolist()), set(split.test_idx.tolist())
    assert not (dev & test), "dev and test overlap"
    assert len(dev) + len(test) == n and (dev | test) == set(range(n)), "split does not cover all issues"
    seen: list = []
    for tr, va in split.folds:
        assert not (set(tr.tolist()) & set(va.tolist())), "train/val overlap inside a fold"
        assert not ((set(tr.tolist()) | set(va.tolist())) & test), "test data leaked into CV"
        seen.extend(va.tolist())
    assert sorted(seen) == sorted(dev), "each dev issue must be a validation item exactly once"
