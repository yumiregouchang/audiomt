import numpy as np
import pytest

from audiomt.cache import FeatureCache
from audiomt.models import AST_SR, AstFeaturizer, make_mfcc_svm, mfcc_features, to_sample_rate
from tests.synth import SR, tone


def two_tone_set(n: int = 12):
    clips = [tone(440, seed=i) for i in range(n)] + [tone(2500, seed=100 + i) for i in range(n)]
    return clips, np.array([0] * n + [1] * n)


def test_mfcc_features_have_80_values():
    f = mfcc_features(tone(440))
    assert f.shape == (80,) and f.dtype == np.float32 and np.isfinite(f).all()


def test_mfcc_svm_separates_two_tones():
    clips, labels = two_tone_set()
    model = make_mfcc_svm().fit(clips[::2], labels[::2])
    assert (model.predict(clips[1::2]) == labels[1::2]).all()
    proba = model.predict_proba(clips[1::2])
    assert proba.shape == (12, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == [0, 1]


def test_unfitted_model_says_so():
    with pytest.raises(RuntimeError, match="fit"):
        make_mfcc_svm().predict([tone(440)])


def test_resampling_for_ast_keeps_duration():
    y = tone(440, seconds=1.0)
    assert to_sample_rate(y, SR, AST_SR).shape == (AST_SR,)
    assert to_sample_rate(y, SR, SR) is not None and to_sample_rate(y, SR, SR).shape == y.shape


def test_feature_cache_round_trip(tmp_path):
    path = tmp_path / "cache" / "features.sqlite"
    vecs = {"a.wav": np.arange(4, dtype=np.float32), "b.wav": np.ones(4, dtype=np.float32)}
    with FeatureCache(path) as cache:
        cache.put_many("feat-v1", vecs)
        assert cache.missing("feat-v1", ["a.wav", "c.wav"]) == ["c.wav"]
        assert cache.missing("feat-v2", ["a.wav"]) == ["a.wav"]  # other featurizer = other namespace
        assert cache.missing("feat-v1", ["a.wav"], variant="mr123") == ["a.wav"]
    with FeatureCache(path) as cache:  # survives reopening
        got = cache.get_all("feat-v1")
    assert set(got) == {"a.wav", "b.wav"}
    assert np.array_equal(got["a.wav"], vecs["a.wav"])


@pytest.mark.slow
def test_ast_random_init_gives_768_dims():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    emb = AstFeaturizer("random-init", batch_size=2)([tone(440), tone(2500), tone(1000)], SR)
    assert emb.shape == (3, 768) and np.isfinite(emb).all()


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def to(self, device):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


def _fake_backend(seen_lengths, batch_sizes):
    """Stand-ins for torch and transformers, so the AST code path runs in CI without downloads."""
    import contextlib
    import types

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                  inference_mode=contextlib.nullcontext)

    class Extractor:
        def __call__(self, chunk, sampling_rate, return_tensors):
            assert sampling_rate == AST_SR and return_tensors == "pt"
            seen_lengths.extend(len(c) for c in chunk)
            batch_sizes.append(len(chunk))
            return {"input_values": _FakeTensor(np.zeros((len(chunk), 1024, 128)))}

        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

    class Model:
        def __call__(self, input_values):
            n = input_values.arr.shape[0]
            return types.SimpleNamespace(pooler_output=_FakeTensor(np.ones((n, 768))))

        def to(self, device):
            return self

        def eval(self):
            return self

        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

    transformers = types.SimpleNamespace(ASTConfig=lambda: None, ASTFeatureExtractor=Extractor,
                                         ASTModel=lambda config=None: Model())
    transformers.ASTModel.from_pretrained = Model.from_pretrained
    return torch, transformers


def test_ast_featurizer_resamples_and_batches(monkeypatch):
    import sys

    seen_lengths, batch_sizes = [], []
    torch, transformers = _fake_backend(seen_lengths, batch_sizes)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    feat = AstFeaturizer("some/checkpoint", batch_size=2)
    emb = feat([tone(440, seconds=1.0)] * 5, SR)
    assert emb.shape == (5, 768) and emb.dtype == np.float32
    assert seen_lengths == [AST_SR] * 5  # 44.1 kHz -> 16 kHz before the model
    assert batch_sizes == [2, 2, 1]
    assert feat.device == "cpu" and feat.seconds_per_clip is not None
    assert feat.id == "ast-pooler:some/checkpoint"


def test_platt_svc_predicts_with_the_svm_itself():
    from sklearn.svm import SVC

    from audiomt.models import PlattSVC

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.0, size=(30, 5)) for c in (0, 1.5, 3)])
    y = np.repeat([0, 1, 2], 30)
    head = PlattSVC().fit(X, y)
    plain = SVC(kernel="rbf", C=10, gamma="scale").fit(X, y)
    assert (head.predict(X) == plain.predict(X)).all()  # calibration does not change decisions
    proba = head.predict_proba(X)
    assert proba.shape == (90, 3) and np.allclose(proba.sum(axis=1), 1)


def test_torch_dll_failure_gets_a_clear_message(monkeypatch):
    import sys

    class FailingTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "torch":
                raise OSError("[WinError 127] Error loading torch\\lib\\shm.dll")
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [FailingTorch(), *sys.meta_path])
    with pytest.raises(ImportError, match="Visual C\\+\\+"):
        AstFeaturizer("some/checkpoint")._load()
