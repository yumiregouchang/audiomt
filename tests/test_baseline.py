import json

import pandas as pd
import pytest

from audiomt.baseline import run_baseline
from audiomt.cli import main
from audiomt.config import Config, load_config
from audiomt.models import AudioClassifier, MfccFeaturizer, make_mfcc_svm


class CountingFeaturizer(MfccFeaturizer):
    def __init__(self):
        self.calls = 0

    def __call__(self, clips, sr):
        self.calls += len(clips)
        return super().__call__(clips, sr)


def test_baseline_end_to_end(fake_esc50, tmp_path):
    cfg = Config(data_dir=fake_esc50, results_dir=tmp_path / "results", models=["mfcc_svm"])
    summary = run_baseline(cfg, log=lambda _: None)

    entry = summary["mfcc_svm"]
    assert list(entry["fold_accuracy"]) == ["1", "2", "3", "4", "5"]
    assert entry["mean"] == 1.0  # two well-separated tones
    assert entry["run"]["config"]["models"] == ["mfcc_svm"]

    out = tmp_path / "results" / "baseline"
    preds = pd.read_csv(out / "preds_mfcc_svm.csv")
    assert len(preds) == 20 and preds["filename"].is_unique  # every clip predicted exactly once
    assert preds["p_true"].between(0, 1).all()
    conf = pd.read_csv(out / "confusion_mfcc_svm.csv", index_col=0)
    assert conf.to_numpy().sum() == 20
    assert json.loads((out / "summary.json").read_text())["mfcc_svm"]["n_correct"] == 20


def test_features_are_cached_between_runs(fake_esc50, tmp_path):
    cfg = Config(data_dir=fake_esc50, results_dir=tmp_path / "results", models=["mfcc_svm"])
    feat = CountingFeaturizer()
    model = AudioClassifier("mfcc_svm", feat, make_mfcc_svm().make_head)
    run_baseline(cfg, models={"mfcc_svm": model}, log=lambda _: None)
    assert feat.calls == 20
    run_baseline(cfg, models={"mfcc_svm": model}, log=lambda _: None)
    assert feat.calls == 20  # second run read everything from the cache


def test_cli_baseline_and_errors(fake_esc50, tmp_path, capsys):
    results = tmp_path / "results"
    assert main(["baseline", "--data-dir", str(fake_esc50), "--results-dir", str(results),
                 "--models", "mfcc_svm", "--per-fold", "1"]) == 0
    assert "mean" in capsys.readouterr().out
    assert len(pd.read_csv(results / "baseline" / "preds_mfcc_svm.csv")) == 10

    assert main(["baseline", "--data-dir", str(tmp_path / "missing"), "--models", "mfcc_svm"]) == 1
    assert "audiomt download" in capsys.readouterr().err
    assert main(["baseline", "--data-dir", str(fake_esc50), "--models", "resnet"]) == 2


def test_config_rejects_typos(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data_dri: somewhere\n")
    with pytest.raises(ValueError, match="data_dri"):
        load_config(bad)
    nested = tmp_path / "nested.yaml"
    nested.write_text("ast:\n  batchsize: 4\n")
    with pytest.raises(ValueError, match="batchsize"):
        load_config(nested)


def test_repo_config_is_valid():
    cfg = load_config()  # configs/experiment.yaml in the repo root
    assert cfg.models == ["mfcc_svm", "ast_logreg"]
    assert cfg.ast.batch_size == 8


def test_summary_is_saved_after_each_model(fake_esc50, tmp_path):
    class Broken(MfccFeaturizer):
        id = "broken-v1"

        def __call__(self, clips, sr):
            raise RuntimeError("model crashed")

    cfg = Config(data_dir=fake_esc50, results_dir=tmp_path / "results", models=["mfcc_svm"])
    models = {"mfcc_svm": make_mfcc_svm(),
              "ast_logreg": AudioClassifier("ast_logreg", Broken(), make_mfcc_svm().make_head)}
    with pytest.raises(RuntimeError):
        run_baseline(cfg, models=models, log=lambda _: None)
    summary = json.loads((tmp_path / "results" / "baseline" / "summary.json").read_text())
    assert list(summary) == ["mfcc_svm"]  # the first model's results survive the second one crashing


def test_cli_import_is_light():
    """The CLI must be able to load torch before sklearn, librosa or sqlite (Windows DLL order)."""
    import subprocess
    import sys

    code = ("import sys, audiomt.cli; "
            "print(sorted(m for m in ('sklearn', 'librosa', 'sqlite3', 'numpy') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
