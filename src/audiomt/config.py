"""Experiment settings: configs/experiment.yaml, overridable from the command line.

Unknown keys are an error, so a typo in the YAML fails loudly instead of being ignored.
Relative paths are relative to the folder audiomt is run from (the repo root).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from audiomt.constants import MODEL_NAMES, SUBSETS
from audiomt.models import AST_DEFAULT

DEFAULT_CONFIG = Path("configs/experiment.yaml")


@dataclass
class AstSettings:
    model_id: str = AST_DEFAULT
    batch_size: int = 8
    device: str = "auto"  # auto, cpu or cuda


@dataclass
class Config:
    data_dir: Path = Path("data/ESC-50")
    subset: str = "esc50"
    per_fold: int = 0  # clips per class per fold, 0 = all (ESC-50 has 8); small values = smoke run
    results_dir: Path = Path("results")
    seed: int = 0
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    ast: AstSettings = field(default_factory=AstSettings)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        if isinstance(self.ast, dict):
            self.ast = AstSettings(**_known(AstSettings, self.ast, "ast"))
        if self.subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, got {self.subset!r}")
        bad = [m for m in self.models if m not in MODEL_NAMES]
        if bad:
            raise ValueError(f"unknown model(s) {bad}; choose from {MODEL_NAMES}")
        if not 0 <= self.per_fold <= 8:
            raise ValueError("per_fold must be between 0 (all) and 8")

    @property
    def cache_path(self) -> Path:
        return self.results_dir / "cache" / "features.sqlite"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_dir"], d["results_dir"] = str(self.data_dir), str(self.results_dir)
        return d


def _known(cls: type, raw: dict[str, Any], where: str) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; allowed: {sorted(allowed)}")
    return raw


def load_config(path: Path | None = None, **overrides: Any) -> Config:
    """Read YAML (if any), then apply non-None overrides from the command line."""
    raw: dict[str, Any] = {}
    if path is None and DEFAULT_CONFIG.exists():
        path = DEFAULT_CONFIG
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _known(Config, raw, str(path))
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**raw)
