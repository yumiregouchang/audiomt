"""Model adapters.

Every model is split into two parts:
  - a Featurizer: clips -> fixed-length vectors (deterministic, so the vectors can be cached),
  - a small scikit-learn head trained on those vectors.
The experiment runner only talks to AudioClassifier (fit / predict / predict_proba), so adding a
third model (e.g. CLAP for RQ3) means adding a featurizer, not changing the runner.

torch and transformers are imported lazily: the MFCC model, the tests and CI do not need them.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Protocol

import librosa
import numpy as np
from scipy.signal import resample_poly
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audiomt.constants import MODEL_NAMES
from audiomt.data import SR

AST_DEFAULT = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000


class Featurizer(Protocol):
    id: str  # cache namespace; must change whenever the features would change

    def __call__(self, clips: Sequence[np.ndarray], sr: int) -> np.ndarray: ...


def to_sample_rate(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(y, dtype=np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(y, sr_to // g, sr_from // g).astype(np.float32)


def mfcc_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """20 MFCCs + deltas, summarised by mean and std over time: 80 values (same as the pilot)."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


class MfccFeaturizer:
    id = "mfcc20-delta-meanstd-v1"

    def __call__(self, clips: Sequence[np.ndarray], sr: int) -> np.ndarray:
        return np.stack([mfcc_features(y, sr) for y in clips])


class AstFeaturizer:
    """768-d pooler output of a pretrained Audio Spectrogram Transformer (not fine-tuned).

    model_id="random-init" builds the same architecture with random weights and downloads nothing:
    only for testing the code path and timing.
    """

    def __init__(self, model_id: str = AST_DEFAULT, batch_size: int = 8, device: str = "auto"):
        self.model_id = model_id
        self.id = f"ast-pooler:{model_id}"
        self.batch_size = batch_size
        self.device_pref = device
        self.seconds = 0.0  # pure model time, for the runtime estimate
        self.clips = 0
        self._model: Any = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import ASTConfig, ASTFeatureExtractor, ASTModel
        except ImportError as e:
            raise ImportError('the AST model needs torch and transformers: pip install -e ".[ast]"') from e
        except OSError as e:  # e.g. Windows "WinError 126/127 ... torch\\lib\\shm.dll"
            raise ImportError(
                f"torch is installed but its DLLs failed to load ({e}). Check that "
                '`python -c "import torch"` works in this environment. If it does not, install the latest '
                "Microsoft Visual C++ Redistributable (x64), https://aka.ms/vc14/vc_redist.x64.exe, "
                "and open a new terminal."
            ) from e

        if self.model_id == "random-init":
            self._fe, model = ASTFeatureExtractor(), ASTModel(ASTConfig())
        else:
            self._fe = ASTFeatureExtractor.from_pretrained(self.model_id)
            model = ASTModel.from_pretrained(self.model_id)
        if self.device_pref == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.device_pref
        self._torch = torch
        self._model = model.to(self.device).eval()

    def __call__(self, clips: Sequence[np.ndarray], sr: int) -> np.ndarray:
        if self._model is None:
            self._load()
        out = []
        for i in range(0, len(clips), self.batch_size):
            chunk = [to_sample_rate(y, sr, AST_SR) for y in clips[i:i + self.batch_size]]
            start = time.perf_counter()
            inputs = self._fe(chunk, sampling_rate=AST_SR, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.inference_mode():
                emb = self._model(**inputs).pooler_output.float().cpu().numpy()
            self.seconds += time.perf_counter() - start
            self.clips += len(chunk)
            out.append(emb)
        return np.concatenate(out).astype(np.float32)

    @property
    def seconds_per_clip(self) -> float | None:
        return self.seconds / self.clips if self.clips else None


class PlattSVC(ClassifierMixin, BaseEstimator):
    """RBF SVM: predict() is the SVM's own decision; predict_proba() is Platt-scaled.

    Platt scaling = a sigmoid fitted on cross-validated decision values (cv folds inside the training
    data; ensemble=False keeps one SVM trained on all of it). This replaces SVC(probability=True),
    deprecated in scikit-learn 1.9. Violations are judged with predict(), so the calibration never
    changes which clips count as violations; it only feeds the confidence drop ΔP.
    """

    def __init__(self, C: float = 10.0, gamma: str | float = "scale", cv: int = 5):
        self.C = C
        self.gamma = gamma
        self.cv = cv

    def fit(self, X: np.ndarray, y: np.ndarray) -> PlattSVC:
        smallest_class = int(np.unique(y, return_counts=True)[1].min())
        cv = max(2, min(self.cv, smallest_class))  # tiny smoke runs have < 5 clips per class
        svm = SVC(kernel="rbf", C=self.C, gamma=self.gamma)
        self.calibrated_ = CalibratedClassifierCV(svm, method="sigmoid", cv=cv, ensemble=False).fit(X, y)
        fitted = self.calibrated_.calibrated_classifiers_[0]
        self.svm_ = getattr(fitted, "estimator", None) or fitted.base_estimator  # name before sklearn 1.2
        self.classes_ = self.calibrated_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svm_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrated_.predict_proba(X)


@dataclass
class AudioClassifier:
    """Featurizer + scikit-learn head. Violations are judged with predict(); predict_proba() is for ΔP."""

    name: str
    featurizer: Featurizer
    make_head: Callable[[], Any]
    sr: int = SR
    head: Any = field(default=None, repr=False)

    def features(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        return self.featurizer(clips, self.sr)

    def fit(self, clips: Sequence[np.ndarray], labels: np.ndarray) -> AudioClassifier:
        return self.fit_features(self.features(clips), labels)

    def fit_features(self, X: np.ndarray, labels: np.ndarray) -> AudioClassifier:
        self.head = self.make_head().fit(X, labels)
        return self

    def predict(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        return self.predict_features(self.features(clips))

    def predict_proba(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        return self.predict_proba_features(self.features(clips))

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.head.predict(X)

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.head.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        self._check_fitted()
        return self.head.classes_

    def _check_fitted(self) -> None:
        if self.head is None:
            raise RuntimeError(f"{self.name}: call fit() first")


def make_mfcc_svm(seed: int = 0) -> AudioClassifier:
    # seed is unused: SVM training and the stratified calibration folds are deterministic.
    def head():
        return make_pipeline(StandardScaler(), PlattSVC(C=10, gamma="scale"))

    return AudioClassifier("mfcc_svm", MfccFeaturizer(), head)


def make_ast_logreg(model_id: str = AST_DEFAULT, batch_size: int = 8, device: str = "auto",
                    seed: int = 0) -> AudioClassifier:
    def head():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))

    return AudioClassifier("ast_logreg", AstFeaturizer(model_id, batch_size, device), head)


def build_model(name: str, cfg: Any) -> AudioClassifier:
    if name == "mfcc_svm":
        return make_mfcc_svm(cfg.seed)
    if name == "ast_logreg":
        return make_ast_logreg(cfg.ast.model_id, cfg.ast.batch_size, cfg.ast.device, cfg.seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
