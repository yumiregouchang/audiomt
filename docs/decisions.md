# Design log

One entry per decision: what, why, what else was considered, and where it goes in the thesis.
Add an entry the day a decision is made; ten minutes a week is enough. Chapter 4 (Method) is
largely this file, rewritten as prose.

| Date | Decision | Why | Alternatives considered | Thesis |
|---|---|---|---|---|
| 2026-11-13 | Use all 50 ESC-50 classes | AST runs at 0.76 s/clip, so the full experiment (18 hand-written plus the LLM-generated variants) takes about 14 h, two nights | ESC-10 only; lighter 16-16 AST | 4.1 Data, G1 |
| 2026-11-13 | AST is not fine-tuned; only its 768-d pooled embedding is used, with logistic regression | Saves compute; both models then have simple heads, so differences come from the features | Fine-tuning AST | 4.2 Models |
| 2026-11-13 | Fixed hyperparameters (SVM: RBF, C=10, gamma=scale; LR: max_iter=3000), no tuning on test folds | Avoids leaking test data into model choice | Nested CV tuning | 4.2 Models |
| 2026-11-16 | Official ESC-50 folds; each fold is tested once with a model trained on the other four | Clips from the same source recording stay in one fold, so there is no leakage | Random splits | 4.1 Data |
| 2026-11-16 | Features cached in SQLite, keyed by (featurizer id, clip, variant) | AST is the bottleneck; one file, resumable, standard library | One .npy per clip; one big .npz | 4.4 Tool |
| 2026-11-16 | MFCC-SVM probabilities by Platt scaling (`CalibratedClassifierCV`, sigmoid, `ensemble=False`); `predict()` uses the SVM decision | `SVC(probability=True)` is deprecated in scikit-learn 1.9. Violations are judged with `predict()`, so calibration only affects ΔP | `SVC(probability=True)` | 4.3 Metrics |
| 2026-11-16 | Accuracy sd over the five folds uses ddof = 1 | Folds are a sample of possible splits | ddof = 0 | 5.1 Results |
| 2026-11-16 | Every result file stores config, git commit and package versions | Any number in the thesis can be traced to the exact code | Notes by hand | 4.4 Tool, reproducibility |
| 2026-11-16 | G1 passed: keep all 50 ESC-50 classes | 5-fold accuracy: MFCC-SVM 0.527 ± 0.036, AST + LR 0.961 ± 0.019; AST 1.05 s/clip, so v0.2 (1922 AST source clips × 18 transforms) takes about 10 h | ESC-10; lighter AST | 4.1 Data, 5.1 Results |
| 2026-11-16 | The CLI imports torch before scikit-learn and librosa | On Windows, torch failed with WinError 127 (shm.dll) when it was loaded after them | Run AST in a separate process | 4.4 Tool |

## Template

| Date | Decision | Why | Alternatives considered | Thesis |
|---|---|---|---|---|
|  |  |  |  |  |
