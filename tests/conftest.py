"""Shared fixtures: a tiny fake ESC-50 folder made of synthetic tones, so no test needs the real dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf

from tests.synth import SR, tone

CLASSES = {0: ("low_tone", 440.0, True), 1: ("high_tone", 2500.0, False)}  # target: (category, Hz, esc10)


@pytest.fixture
def fake_esc50(tmp_path: Path) -> Path:
    """2 classes x 5 folds x 2 clips = 20 clips, laid out like ESC-50 (meta/esc50.csv + audio/*.wav)."""
    root = tmp_path / "ESC-50"
    (root / "audio").mkdir(parents=True)
    (root / "meta").mkdir()
    rows = []
    for target, (category, freq, esc10) in CLASSES.items():
        for fold in range(1, 6):
            for take in range(2):
                name = f"{fold}-{target}{take:02d}-A-{target}.wav"
                seed = 100 * target + 10 * fold + take
                y = tone(freq * (1 + 0.01 * take), seed=seed)
                sf.write(root / "audio" / name, y, SR, subtype="PCM_16")
                rows.append({"filename": name, "fold": fold, "target": target, "category": category,
                             "esc10": esc10, "src_file": 1000 + seed, "take": "A"})
    pd.DataFrame(rows).sort_values("filename").to_csv(root / "meta" / "esc50.csv", index=False)
    return root
