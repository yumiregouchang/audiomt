"""Command-line entry point: `audiomt <command>`.

v0.1: download, baseline.  Planned: listen, run (v0.2), generate (v0.3), report (v1.0).
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from audiomt import __version__
from audiomt.constants import MODEL_NAMES, SUBSETS  # light: no numpy, sklearn or torch here


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path,
                        help="YAML settings (default: configs/experiment.yaml if present)")
    common.add_argument("--data-dir", type=Path, help="ESC-50 folder (overrides the config)")
    common.add_argument("--results-dir", type=Path, help="where results and the feature cache go")

    p = argparse.ArgumentParser(prog="audiomt",
                                description="LLM-assisted metamorphic testing of audio classifiers")
    p.add_argument("--version", action="version", version=f"audiomt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download", parents=[common], help="download ESC-50 into data_dir")

    b = sub.add_parser("baseline", parents=[common], help="5-fold accuracy of each model; caches features")
    b.add_argument("--models", help=f"comma-separated subset of {','.join(MODEL_NAMES)}")
    b.add_argument("--subset", choices=SUBSETS, help="esc50 (all 50 classes) or esc10")
    b.add_argument("--per-fold", type=int,
                   help="clips per class per fold for a quick smoke run (1-8, 0 = all)")
    return p


def _preload_torch(models: str | None) -> None:
    """Import torch before scikit-learn, librosa and sqlite when the AST model may run.

    On Windows, `import torch` fails with "WinError 127 ... torch\\lib\\shm.dll" if another library
    has already loaded an older copy of a DLL that torch shares (the MSVC runtime, OpenMP).
    Loading torch first avoids that. A failure here is ignored: the AST model reports it, with advice,
    when it is actually used, and the MFCC model does not need torch at all.
    """
    if models is not None and "ast_logreg" not in models.split(","):
        return
    with contextlib.suppress(Exception):
        import torch  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "baseline":
        _preload_torch(args.models)
    from audiomt.config import load_config  # imported here so --help and --version stay fast
    from audiomt.data import DatasetNotFound

    overrides = {"data_dir": args.data_dir, "results_dir": args.results_dir}
    if args.command == "baseline":
        overrides.update(subset=args.subset, per_fold=args.per_fold,
                         models=args.models.split(",") if args.models else None)
    try:
        cfg = load_config(args.config, **overrides)
    except (ValueError, TypeError, FileNotFoundError) as e:  # bad YAML keys/values or missing config file
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        if args.command == "download":
            from audiomt.data import download_esc50

            download_esc50(cfg.data_dir)
        elif args.command == "baseline":
            from audiomt.baseline import run_baseline

            run_baseline(cfg)
    except (DatasetNotFound, FileExistsError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
