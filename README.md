# audiomt — LLM-assisted metamorphic testing of audio classifiers

[![CI](https://github.com/yumiregouchang/audiomt/actions/workflows/ci.yml/badge.svg)](https://github.com/yumiregouchang/audiomt/actions/workflows/ci.yml)

Audio classifiers have no test oracle: for a new recording, nobody can say what the "right" output is.
**Metamorphic testing** sidesteps this by checking relations instead of single outputs. If a dog bark is
classified as *dog*, the same clip a little quieter, with light background noise or shifted in time should
still be *dog*; a change of label is a **violation**.

audiomt runs such checks on environmental sound classifiers (ESC-50). Its metamorphic relations (MRs) come
from two sources: written by hand, or **proposed by a large language model**. The LLM never writes code.
It returns small JSON specs that are validated against a fixed library of audio operations, so both kinds
of MR run through exactly the same, tested code.

> Status: **v0.1, baselines.** Data loading, two models (MFCC + SVM and a pretrained Audio Spectrogram
> Transformer), feature caching and 5-fold evaluation. See the roadmap below.

## Install

```bash
git clone https://github.com/yumiregouchang/audiomt.git
cd audiomt
pip install -e ".[ast,dev]"   # ast = torch + transformers, only needed for the AST model
```

## Quick start

```bash
audiomt download                                   # ESC-50 (~0.6 GB) into data/ESC-50
audiomt baseline --models mfcc_svm --per-fold 2    # 1-minute smoke run
audiomt baseline                                   # both models, all 2000 clips
```

To use an ESC-50 copy you already have, set `data_dir` in `configs/experiment.yaml`.
The AST model needs about 0.8 s per clip on a laptop CPU (around 25 minutes for ESC-50). Features are
cached in `results/cache/features.sqlite`, so an interrupted run resumes and later runs are instant.

Outputs in `results/baseline/`:

| File | Content |
|---|---|
| `preds_<model>.csv` | one row per clip: fold, true and predicted class, p(true), p(pred), correct |
| `confusion_<model>.csv` | confusion matrix over all five folds |
| `summary.json` | per-fold accuracy, mean, sd, timing, plus config, git commit and package versions |

## Design

- **`ops.py`**: the only transformations an MR may use (gain, noise at a given SNR, filters, reverb, pitch,
  time stretch, silence, shift, resampling). It is the single source of truth: the validator and the LLM
  prompt are both generated from it.
- **`spec.py`**: the MR schema (pydantic). LLM replies are parsed and validated, never executed.
- **`models.py`**: every model is a cacheable *featurizer* plus a small scikit-learn *head*, behind one
  `fit / predict / predict_proba` interface, so adding a model does not touch the experiment code.
- **`cache.py`**: SQLite feature store keyed by (featurizer, clip, variant).
- **`repro.py`**: every result records its config, git commit and package versions.
- **Tests** use synthetic audio only, so CI never downloads the dataset or a model.

```
configs/experiment.yaml   settings (CLI options override them)
docs/decisions.md         design log: what was decided, when and why
src/audiomt/              data, ops, spec, models, cache, baseline, repro, cli
tests/                    pytest + hypothesis, synthetic audio
```

## Roadmap

| Version | Adds |
|---|---|
| **v0.1** | ESC-50 loader, MFCC-SVM and AST models, feature cache, `audiomt baseline` |
| v0.2 | 18 hand-written MRs (6 types × 3 strengths), resumable `audiomt run`, listening check, violation statistics |
| v0.3 | `audiomt generate`: LLM-proposed MRs, a 7-stage filter, comparison with the hand-written MRs |
| v1.0 | HTML report, interactive demo, release |

## Development

```bash
ruff check .
pytest -q -m "not slow"    # "slow" tests need torch or downloads
```

## Data and licence

ESC-50: K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015.
Licensed CC BY-NC 3.0 (ESC-10 subset CC BY 3.0). The dataset is downloaded by the user, never redistributed.

Code licence: MIT, see [LICENSE](LICENSE).

Developed as part of a bachelor's thesis in software engineering at Tampere University.
