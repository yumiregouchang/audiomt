# Changelog

## v0.1.0 — baselines

- `audiomt download`: fetches ESC-50 (git clone, or zip if git is missing).
- `audiomt baseline`: 5-fold accuracy on the official ESC-50 folds for MFCC + SVM and AST + logistic
  regression; per-clip predictions, confusion matrices and a summary with run metadata.
- Feature cache (SQLite): interrupted runs resume; repeated runs reuse features.
- `configs/experiment.yaml` with validation: unknown keys and bad values fail loudly.
- Tests on synthetic audio (no dataset or model download in CI), including the AST code path with a stand-in
  backend; explicit ruff rule set.

## v0.0 — starter

- Operation library (`ops.py`) and MR schema with LLM-reply validation (`spec.py`).
