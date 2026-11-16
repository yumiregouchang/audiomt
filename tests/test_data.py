import numpy as np
import pytest
import soundfile as sf

from audiomt.data import DatasetNotFound, folds, is_complete, load_clip, load_meta, split


def test_meta_columns_and_subsets(fake_esc50):
    meta = load_meta(fake_esc50)
    assert len(meta) == 20
    assert meta["esc10"].dtype == bool
    esc10 = load_meta(fake_esc50, subset="esc10")
    assert set(esc10["category"]) == {"low_tone"}


def test_per_fold_limit_keeps_every_fold(fake_esc50):
    meta = load_meta(fake_esc50, per_fold=1)
    assert len(meta) == 10  # 2 classes x 5 folds x 1
    assert folds(meta) == [1, 2, 3, 4, 5]
    assert meta.groupby(["category", "fold"]).size().eq(1).all()


def test_official_folds_partition_the_data(fake_esc50):
    meta = load_meta(fake_esc50)
    seen = []
    for fold in folds(meta):
        train, test = split(meta, fold)
        assert set(train).isdisjoint(test)
        assert len(train) + len(test) == len(meta)
        assert (meta.loc[test, "fold"] == fold).all()
        seen.extend(test)
    assert sorted(seen) == list(range(len(meta)))  # every clip is tested exactly once


def test_load_clip_is_mono_float32_at_44k(fake_esc50, tmp_path):
    name = load_meta(fake_esc50)["filename"][0]
    y = load_clip(fake_esc50, name)
    assert y.dtype == np.float32 and y.ndim == 1 and y.size == 22050

    stereo_22k = np.stack([y[::2], y[::2]], axis=1)  # other rate + 2 channels -> resampled and mixed down
    sf.write(fake_esc50 / "audio" / "stereo.wav", stereo_22k, 22050)
    z = load_clip(fake_esc50, "stereo.wav")
    assert z.ndim == 1 and abs(z.size - 22050) <= 2


def test_missing_dataset_has_a_helpful_error(tmp_path):
    with pytest.raises(DatasetNotFound, match="audiomt download"):
        load_meta(tmp_path / "nowhere")
    assert not is_complete(tmp_path / "nowhere")


def test_bad_subset_is_rejected(fake_esc50):
    with pytest.raises(ValueError):
        load_meta(fake_esc50, subset="esc100")
