"""Preflight 'doctor' (finds setup problems early), data auto-discovery and a diagnostics bundle."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from .sqldump import convert
from .utils import cuda_ok


def _ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:  # noqa: BLE001
        try:
            return int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 1e6
        except Exception:  # noqa: BLE001
            return None


def find_data_file(csv_path: str | Path, roots: list | None = None) -> Path | None:
    """Return csv_path if present; else search typical places (Kaggle/Colab/Drive/cwd) for the CSV or a TAWOS SQL dump."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        return csv_path
    roots = roots or [Path.cwd(), Path("/kaggle/input"), Path("/content"), Path("/content/drive/MyDrive")]
    for r in roots:
        if not Path(r).exists():
            continue
        for pat in ("", "*/", "*/*/", "*/*/*/", "*/*/*/*/"):
            hits = sorted(Path(r).glob(pat + "tawos_issues.csv"))
            if hits:
                return hits[0]
            for ext in ("*.sql", "*.sql.gz", "*.zip"):
                hits = [h for h in sorted(Path(r).glob(pat + ext)) if "tawos" in h.name.lower()]
                if hits:
                    return hits[0]
    return None


def prepare_data(csv_path: str | Path) -> Path | None:
    """Make sure the issues CSV exists: reuse, copy a found CSV, or convert a found SQL dump. Returns None if nothing found."""
    csv_path = Path(csv_path)
    src = find_data_file(csv_path)
    if src is None or src == csv_path:
        return src
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".csv":
        shutil.copy(src, csv_path)
    else:
        print(f"Converting {src} -> {csv_path} (can take a few minutes) ...")
        print(convert(src, csv_path))
    return csv_path


def run_doctor(csv_path: str | Path = "data/tawos_issues.csv", bench: bool = True, roots: list | None = None) -> list:
    rows: list = []

    def add(name, status, detail="", value=None):
        rows.append({"check": name, "status": status, "detail": str(detail), "value": value})

    add("Python >= 3.10", "PASS" if sys.version_info >= (3, 10) else "FAIL", platform.python_version())
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "sentence_transformers", "yaml", "matplotlib", "reportlab"):
        try:
            m = __import__(mod)
            add(f"package {mod}", "PASS", getattr(m, "__version__", "?"))
        except Exception as e:  # noqa: BLE001
            add(f"package {mod}", "FAIL", f"{e}  -> pip install -r requirements.txt")
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            ok = cuda_ok()
            add("GPU (CUDA)", "PASS" if ok else "FAIL", name if ok else f"{name} is present but unusable with this PyTorch build -> choose GPU T4 instead (Kaggle: Settings > Accelerator)")
        else:
            where = ("Kaggle: Settings > Accelerator > GPU T4 x2, then Run All again" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ
                     else "Colab: Runtime > Change runtime type > T4 GPU" if "google.colab" in sys.modules else "needs an NVIDIA GPU with CUDA PyTorch")
            add("GPU (CUDA)", "WARN", f"NO GPU (torch {torch.__version__}) -> full run is ~20x slower on CPU. {where}")
    except Exception:  # noqa: BLE001
        pass
    ram = _ram_gb()
    add("RAM >= 8 GB", "PASS" if ram and ram >= 8 else "WARN", f"{ram:.1f} GB" if ram else "unknown")
    free = shutil.disk_usage(".").free / 1e9
    add("free disk >= 5 GB", "PASS" if free >= 5 else "WARN", f"{free:.1f} GB")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=6)
        add("internet (huggingface.co, needed to download SBERT)", "PASS")
    except Exception as e:  # noqa: BLE001
        add("internet (huggingface.co, needed to download SBERT)", "FAIL", f"{e}  (Kaggle: Settings > Internet ON)")
    src = find_data_file(csv_path, roots)
    add("TAWOS data found", "PASS" if src else "FAIL", src or "put tawos_issues.csv (or the TAWOS .sql/.zip) in data/ - see docs/RUN_GUIDE.md")
    if bench:
        try:
            import numpy as np
            from .config import load_config
            from .models import CNNRegressor
            X = np.random.default_rng(0).normal(size=(1000, 768)).astype("float32")
            y = np.clip(np.round(4 + 2 * X[:, 0]), 1, None)
            params = {"lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-4}
            CNNRegressor(params, 0, load_config(None, ["cnn.max_epochs=1", "cnn.patience=50"])).fit(X, y)   # warm-up: CUDA/cuDNN start-up is not training time
            cfg = load_config(None, ["cnn.max_epochs=4", "cnn.patience=50"])
            t = time.time()
            CNNRegressor(params, 0, cfg).fit(X, y)
            ep = (time.time() - t) / 4
            est = ep * 30 * 33 / 60
            add("speed benchmark", "PASS" if est < 30 else "WARN",
                f"{ep:.2f}s/epoch for 1000 issues -> ~{est:.0f} min per 1000-issue project (default grid); scales ~linearly with issues",
                value=ep)
        except Exception as e:  # noqa: BLE001
            add("speed benchmark", "WARN", e)
    return rows


def make_bundle(output_dir: str | Path, dest: str | Path, warnings_log: list, doctor_rows: list) -> Path:
    """Zip everything a teammate/helper needs to diagnose a run (no raw data)."""
    out = Path(output_dir)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("warnings.txt", "\n".join(sorted(set(warnings_log))) or "none")
        z.writestr("doctor.json", json.dumps(doctor_rows, indent=2))
        z.writestr("system.txt", f"{platform.platform()}\nenv KAGGLE={'KAGGLE_KERNEL_RUN_TYPE' in os.environ}")
        for rel in ("run.log", "run_metadata.json", "data_cleaning_report.json"):
            if (out / rel).exists():
                z.write(out / rel, rel)
        for f in (out / "tables").glob("*.csv") if (out / "tables").exists() else []:
            z.write(f, f"tables/{f.name}")
    return Path(dest)
