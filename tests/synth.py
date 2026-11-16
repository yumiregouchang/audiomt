"""Synthetic audio for tests: no dataset or model download needed."""
import numpy as np

SR = 44100


def tone(freq: float, seconds: float = 0.5, seed: int = 0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.normal(size=t.size)).astype(np.float32)
