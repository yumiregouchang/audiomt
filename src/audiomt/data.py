"""ESC-50: download, metadata, official folds and clip loading.

The dataset is never committed to Git. `audiomt download` fetches it into `data_dir`
(default data/ESC-50), or `data_dir` in configs/experiment.yaml can point to an existing copy.
Dataset: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM MM 2015.
Licence CC BY-NC 3.0 (the ESC-10 subset is CC BY 3.0).
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from audiomt.constants import SUBSETS

ESC50_REPO = "https://github.com/karolpiczak/ESC-50"
ESC50_ZIP = ESC50_REPO + "/archive/refs/heads/master.zip"
SR = 44100  # ESC-50 native sample rate
N_CLIPS = 2000
REQUIRED_COLUMNS = ("filename", "fold", "target", "category", "esc10")


class DatasetNotFound(FileNotFoundError):
    """Raised when data_dir does not contain ESC-50."""


def meta_path(root: Path) -> Path:
    return Path(root) / "meta" / "esc50.csv"


def is_complete(root: Path, expected: int = N_CLIPS) -> bool:
    root = Path(root)
    return meta_path(root).exists() and len(list((root / "audio").glob("*.wav"))) == expected


def load_meta(root: Path, subset: str = "esc50", per_fold: int = 0) -> pd.DataFrame:
    """Read meta/esc50.csv.

    subset: "esc50" (all 50 classes) or "esc10".
    per_fold: keep at most this many clips per class in each fold (0 = all; ESC-50 has 8).
              Limiting per fold, not per class, keeps all 5 folds populated for a quick smoke run.
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    path = meta_path(root)
    if not path.exists():
        raise DatasetNotFound(
            f"ESC-50 not found: {path} does not exist. Run `audiomt download` or set data_dir "
            "in configs/experiment.yaml to your ESC-50 folder."
        )
    meta = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    meta["esc10"] = meta["esc10"].astype(str).str.lower().eq("true")
    if subset == "esc10":
        meta = meta[meta["esc10"]]
    if per_fold:
        meta = meta.groupby(["category", "fold"], sort=False).head(per_fold)
    return meta.reset_index(drop=True)


def folds(meta: pd.DataFrame) -> list[int]:
    return sorted(int(f) for f in meta["fold"].unique())


def split(meta: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (train, test) for one official fold: train on the other folds, test on this one."""
    is_test = (meta["fold"] == fold).to_numpy()
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def load_clip(root: Path, filename: str, sr: int = SR) -> np.ndarray:
    """Load one clip as mono float32 at `sr` (ESC-50 files are already 44.1 kHz mono)."""
    y, file_sr = sf.read(Path(root) / "audio" / filename, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def download_esc50(target: Path) -> Path:
    """Fetch ESC-50 into `target` with `git clone --depth 1`, or the GitHub zip if git is missing."""
    target = Path(target)
    if is_complete(target):
        print(f"ESC-50 already present at {target}")
        return target
    if target.exists():
        raise FileExistsError(f"{target} exists but is incomplete; delete it and run again.")
    target.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    clone = [git, "clone", "--depth", "1", ESC50_REPO, str(target)] if git else None
    cloned = clone is not None and subprocess.run(clone, check=False).returncode == 0
    if not cloned:
        if target.exists():  # a failed clone can leave a partial folder behind
            shutil.rmtree(target)
        zip_path = target.parent / "ESC-50-master.zip"
        print(f"Downloading {ESC50_ZIP} (about 0.6 GB) ...")
        urllib.request.urlretrieve(ESC50_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target.parent)
        (target.parent / "ESC-50-master").rename(target)
        zip_path.unlink()
    if not is_complete(target):
        raise RuntimeError(f"Download finished but {target} does not contain {N_CLIPS} clips.")
    print(f"OK: {N_CLIPS} clips in {target}")
    return target
