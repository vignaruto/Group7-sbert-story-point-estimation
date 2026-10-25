import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spsbert.aggregate import aggregate
from spsbert.config import load_config
from spsbert.embeddings import EmbeddingStore, HashingEncoder
from spsbert.config import EmbeddingConfig
from spsbert.experiment import run_all
from spsbert.models import CNNRegressor, ConstantRegressor, build_cnn_module
from spsbert.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_constant_regressors():
    y = np.array([1, 2, 3, 100.0])
    assert ConstantRegressor("mean").fit(None, y).predict([0, 0])[0] == pytest.approx(26.5)
    assert ConstantRegressor("median").fit(None, y).predict([0])[0] == pytest.approx(2.5)


def test_cnn_architecture_has_three_conv_layers_and_right_shape():
    torch = pytest.importorskip("torch")
    net = build_cnn_module(768, [16, 32, 64], 5, 2, 0, 64, 0.1)
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == 3
    assert net(torch.randn(4, 768)).shape == (4,)
    assert build_cnn_module(24, [16, 32, 64], 5, 2, 0, 64, 0.1)(torch.randn(4, 24)).shape == (4,)   # tiny dims too


def _toy(n=625, d=64, seed=0):
    """Dense signal spread over all embedding dimensions (like real sentence embeddings)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) / np.sqrt(d)
    y = np.clip(np.round(4 + 2 * s + 0.5 * rng.normal(size=n)), 1, None)
    return X, y


def test_cnn_learns_dense_signal_and_is_reproducible():
    pytest.importorskip("torch")
    X, y = _toy()
    cfg = load_config(None)
    params = {"lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-4}
    m1 = CNNRegressor(params, 0, cfg).fit(X[:500], y[:500])
    m2 = CNNRegressor(params, 0, cfg).fit(X[:500], y[:500])
    p1, p2 = m1.predict(X[500:]), m2.predict(X[500:])
    assert np.allclose(p1, p2, atol=1e-5)                          # same seed -> same output
    cnn_mae = np.abs(p1 - y[500:]).mean()
    med_mae = np.abs(np.median(y[:500]) - y[500:]).mean()
    assert cnn_mae < 0.95 * med_mae                                # clearly learns
    assert p1.min() >= y[:500].min() - 1e-6 and p1.max() <= y[:500].max() + 1e-6   # clipped to train range


def test_cnn_flatten_and_adaptive_heads_both_work():
    torch = pytest.importorskip("torch")
    for ap, pt in ((0, "avg"), (0, "max"), (16, "max"), (16, "avg")):
        net = build_cnn_module(100, [8, 8, 8], 3, 2, ap, 16, 0.1, pt)
        assert net(torch.randn(3, 100)).shape == (3,)


def test_embedding_cache(tmp_path):
    cfg = EmbeddingConfig(backend="hashing", hashing_dim=16, cache_dir=str(tmp_path))
    enc = HashingEncoder(16)
    calls = {"n": 0}
    orig = enc.encode

    def counting(t):
        calls["n"] += 1
        return orig(t)

    enc.encode = counting
    store = EmbeddingStore(enc, cfg)
    a, b = store.get(["hello world", "foo bar"]), store.get(["hello world", "foo bar"])
    assert calls["n"] == 1 and np.array_equal(a, b) and a.shape == (2, 16)


def test_export_script_with_sqlite(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_tawos_to_csv import export_issues
    con = sqlite3.connect(":memory:")
    con.executescript("""CREATE TABLE Project(ID INTEGER, Name TEXT);
        CREATE TABLE Issue(ID INTEGER, Project_ID INTEGER, Issue_Key TEXT, Title TEXT, Description_Text TEXT,
                           Creation_Date TEXT, Story_Point REAL);
        INSERT INTO Project VALUES (1,'P1');
        INSERT INTO Issue VALUES (1,1,'P1-1','t','d','2020-01-01',3),(2,1,'P1-2','t2','d2','2020-01-02',NULL);""")
    out = tmp_path / "o.csv"
    assert export_issues(con, out) == 1
    assert list(pd.read_csv(out).columns) == ["Project_Name", "Issue_Key", "Title", "Description_Text",
                                              "Creation_Date", "Story_Point"]


def test_end_to_end_smoke_pipeline(tmp_path):
    pytest.importorskip("torch")
    csv = make_synthetic(tmp_path / "s.csv", n_projects=2, sizes=(130, 150))
    cfg = load_config(ROOT / "config" / "smoke_test.yaml",
                      [f"data.csv_path={csv}", f"output_dir={tmp_path / 'res'}",
                       f"embedding.cache_dir={tmp_path / 'cache'}"])
    results = run_all(cfg)
    assert len(results) == 2
    for r in results:
        assert set(r["methods"]) == set(cfg.evaluation.methods)
        assert r["n_dev"] + r["n_test"] == r["n_issues"]
        for m in r["methods"].values():
            assert len(m["cv"]["fold_mae"]) == 5 and len(m["cv"]["fold_mdae"]) == 5
            assert m["test"]["mae"] >= 0 and m["test"]["mdae"] >= 0
            lo, hi = m["test"]["mae_ci95"]
            assert lo <= hi
    # the synthetic target carries signal -> text models must beat the median baseline on average;
    # the CNN (tiny data, 130 issues) only has to be sane here: learning is asserted in the dense-signal test.
    mae = {k: np.mean([r["methods"][k]["test"]["mae"] for r in results]) for k in results[0]["methods"]}
    assert mae["sbert_ridge"] < mae["median_baseline"] and mae["tfidf_ridge"] < mae["median_baseline"]
    assert np.isfinite(mae["sbert_cnn"]) and mae["sbert_cnn"] < 1.5 * mae["median_baseline"]
    # splits file: test issues never appear in development list
    sp = json.loads(next((tmp_path / "res" / "splits").glob("*.json")).read_text())
    assert not set(sp["test_issue_keys"]) & set(sp["dev_issue_keys"])
    # caching: a second run reuses results
    again = run_all(cfg)
    assert [r["config_hash"] for r in again] == [r["config_hash"] for r in results]
    agg = aggregate(tmp_path / "res")
    assert len(agg["macro"]) == len(cfg.evaluation.methods)
    assert (tmp_path / "res" / "figures" / "test_mae_per_project.png").exists()


def test_sbert_encoder_wiring_with_fake_library(monkeypatch):
    """The real SBERT path cannot download weights offline; verify arguments/shape handling with a stand-in."""
    import types

    seen = {}

    class FakeST:
        def __init__(self, name, device=None):
            seen["name"], seen["device"] = name, device
            self.max_seq_length = 512

        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
            seen.update(batch_size=batch_size, normalize=normalize_embeddings)
            return np.ones((len(texts), 8))

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    from spsbert.embeddings import SBERTEncoder

    enc = SBERTEncoder(EmbeddingConfig(model_name="m", max_seq_length=128, batch_size=7, device="cpu"))
    out = enc.encode(["a", "b", "c"])
    assert out.shape == (3, 8) and out.dtype == np.float32
    assert seen == {"name": "m", "device": "cpu", "batch_size": 7, "normalize": True}
    assert enc.model.max_seq_length == 128
