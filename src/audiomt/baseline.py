"""`audiomt baseline`: accuracy of each model on the official ESC-50 folds.

For each fold: train the head on the other four folds, predict this fold. Features are computed once
per clip and cached, so re-running (or adding a model) only computes what is missing.

Outputs in results/baseline/:
  preds_<model>.csv      one row per clip: fold, true and predicted class, p(true), p(pred), correct
                         -> v0.2 uses the correct rows as source inputs for metamorphic testing
  confusion_<model>.csv  true class x predicted class counts over all folds
  summary.json           per-model fold accuracies, mean, sd, timing and run metadata (merged across runs)
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audiomt.cache import FeatureCache
from audiomt.config import Config
from audiomt.data import folds, load_clip, load_meta, split
from audiomt.models import AstFeaturizer, AudioClassifier, build_model
from audiomt.repro import now, run_info

CHUNK = 32  # clips loaded and cached per step; an interrupted run loses at most one chunk

Log = Callable[[str], None]


def featurize_all(model: AudioClassifier, meta: pd.DataFrame, data_dir: Path, cache: FeatureCache,
                  log: Log = print) -> np.ndarray:
    """Feature matrix in meta row order; computes and caches only the clips not cached yet."""
    names = meta["filename"].tolist()
    fid = model.featurizer.id
    todo = cache.missing(fid, names)
    if todo:
        log(f"  {model.name}: computing features for {len(todo)} clips ({len(names) - len(todo)} cached)")
        start = time.perf_counter()
        for i in range(0, len(todo), CHUNK):
            batch = todo[i:i + CHUNK]
            X = model.features([load_clip(data_dir, n, model.sr) for n in batch])
            cache.put_many(fid, dict(zip(batch, X, strict=True)))
            done, elapsed = i + len(batch), time.perf_counter() - start
            left = elapsed / done * (len(todo) - done)
            log(f"    {done}/{len(todo)} clips, {elapsed:.0f} s elapsed, ~{left:.0f} s left")
    else:
        log(f"  {model.name}: all {len(names)} feature vectors cached")
    vectors = cache.get_all(fid)
    return np.stack([vectors[n] for n in names])


def cross_validate(
    model: AudioClassifier, X: np.ndarray, meta: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, float]]:
    y = meta["target"].to_numpy()
    category = dict(zip(meta["target"], meta["category"], strict=True))
    parts, accuracy = [], {}
    for fold in folds(meta):
        train, test = split(meta, fold)
        model.fit_features(X[train], y[train])
        pred = model.predict_features(X[test])
        proba = model.predict_proba_features(X[test])
        col = {c: j for j, c in enumerate(model.classes_)}
        rows = np.arange(len(test))
        p_true = np.array([proba[r, col[t]] if t in col else np.nan
                           for r, t in zip(rows, y[test], strict=True)])
        p_pred = proba[rows, [col[p] for p in pred]]
        accuracy[fold] = float(np.mean(pred == y[test]))
        parts.append(pd.DataFrame({
            "model": model.name,
            "fold": fold,
            "filename": meta["filename"].to_numpy()[test],
            "target": y[test],
            "category": [category[t] for t in y[test]],
            "pred": pred,
            "pred_category": [category.get(p, str(p)) for p in pred],
            "correct": pred == y[test],
            "p_true": p_true.round(6),
            "p_pred": p_pred.round(6),
        }))
    return pd.concat(parts, ignore_index=True), accuracy


def confusion(preds: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    order = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    table = pd.crosstab(preds["category"], preds["pred_category"])
    return table.reindex(index=order, columns=order, fill_value=0)


def run_baseline(
    cfg: Config, models: dict[str, AudioClassifier] | None = None, log: Log = print
) -> dict[str, Any]:
    """Run the baseline for cfg.models (or the given model objects, used by the tests)."""
    meta = load_meta(cfg.data_dir, cfg.subset, cfg.per_fold)
    if models is None:
        models = {name: build_model(name, cfg) for name in cfg.models}
    out_dir = cfg.results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    info = run_info(cfg.to_dict(), "baseline")
    log(f"{len(meta)} clips, {meta['category'].nunique()} classes, folds {folds(meta)}")

    with FeatureCache(cfg.cache_path) as cache:
        for name, model in models.items():
            t0 = time.perf_counter()
            X = featurize_all(model, meta, cfg.data_dir, cache, log)
            preds, acc = cross_validate(model, X, meta)
            preds.to_csv(out_dir / f"preds_{name}.csv", index=False)
            confusion(preds, meta).to_csv(out_dir / f"confusion_{name}.csv")
            values = list(acc.values())
            entry = {
                "n_clips": len(meta),
                "n_classes": int(meta["category"].nunique()),
                "fold_accuracy": {str(k): round(v, 4) for k, v in acc.items()},
                "mean": round(float(np.mean(values)), 4),
                "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None,
                "n_correct": int(preds["correct"].sum()),
                "featurizer": model.featurizer.id,
                "seconds": round(time.perf_counter() - t0, 1),
                "run": {**info, "finished_utc": now()},
            }
            if isinstance(model.featurizer, AstFeaturizer) and model.featurizer.seconds_per_clip:
                entry["ast_seconds_per_clip"] = round(model.featurizer.seconds_per_clip, 3)
            summary[name] = entry
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("")
    log(format_table({k: summary[k] for k in models}))
    log(f"\nSaved per-clip predictions, confusion matrices and summary.json to {out_dir}")
    return summary


def format_table(summary: dict[str, Any]) -> str:
    fold_ids = sorted({f for e in summary.values() for f in e["fold_accuracy"]}, key=int)
    head = f"{'model':<12}" + "".join(f"{'fold ' + f:>9}" for f in fold_ids) + f"{'mean':>9}{'sd':>8}"
    lines = [head, "-" * len(head)]
    for name, e in summary.items():
        cells = "".join(f"{e['fold_accuracy'].get(f, float('nan')):>9.3f}" for f in fold_ids)
        sd = f"{e['sd']:>8.3f}" if e["sd"] is not None else f"{'-':>8}"
        lines.append(f"{name:<12}{cells}{e['mean']:>9.3f}{sd}")
    return "\n".join(lines)
