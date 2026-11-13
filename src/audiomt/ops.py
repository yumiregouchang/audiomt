"""Operation library: the only transformations an MR (hand-written or LLM-generated) may use.

Every op takes a mono float32 signal y at sample rate sr and returns a signal of the SAME length.
OPS is the single source of truth: the validator and the LLM prompt are both generated from it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gain(y, sr, db):
    return np.clip(y * 10 ** (db / 20), -1, 1)


def noise(y, sr, color, snr_db, seed=0):
    n = len(y)
    spec = np.fft.rfft(_rng(seed).normal(size=n))
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1]
    spec *= {"white": 1.0, "pink": f ** -0.5, "brown": f ** -1.0}[color]
    w = np.fft.irfft(spec, n)
    p_sig = np.mean(y ** 2)
    if p_sig == 0:
        return y.copy()
    w *= np.sqrt(p_sig / 10 ** (snr_db / 10) / np.mean(w ** 2))
    return np.clip(y + w, -1, 1)


def lowpass(y, sr, cutoff_hz):
    return sosfiltfilt(butter(6, min(cutoff_hz, sr / 2 - 1), "low", fs=sr, output="sos"), y)


def highpass(y, sr, cutoff_hz):
    return sosfiltfilt(butter(4, cutoff_hz, "high", fs=sr, output="sos"), y)


def reverb(y, sr, rt60_s, wet, seed=0):
    t = np.arange(int(rt60_s * sr)) / sr
    ir = _rng(seed).normal(size=t.size) * np.exp(-6.91 * t / rt60_s)
    wet_sig = fftconvolve(y, ir)[: len(y)]
    wet_sig *= np.max(np.abs(y)) / (np.max(np.abs(wet_sig)) + 1e-12)
    return np.clip((1 - wet) * y + wet * wet_sig, -1, 1)


def pitch_shift(y, sr, semitones):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def time_stretch(y, sr, rate):
    return fit_length(librosa.effects.time_stretch(y, rate=rate), len(y))


def pad_silence(y, sr, seconds):
    return fit_length(np.concatenate([np.zeros(int(seconds * sr), y.dtype), y]), len(y))


def shift(y, sr, seconds):
    return np.roll(y, int(seconds * sr))


def resample(y, sr, sr_hz):
    sr_hz = int(sr_hz)
    g = np.gcd(sr, sr_hz)
    return fit_length(resample_poly(resample_poly(y, sr_hz // g, sr // g), sr // g, sr_hz // g), len(y))


@dataclass(frozen=True)
class Param:
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpDef:
    fn: Callable
    doc: str
    params: dict[str, Param] = field(default_factory=dict)


OPS: dict[str, OpDef] = {
    "gain": OpDef(gain, "change loudness", {"db": Param(-20, 20)}),
    "noise": OpDef(noise, "add background noise", {"color": Param(choices=("white", "pink", "brown")),
                                                   "snr_db": Param(10, 40)}),
    "lowpass": OpDef(lowpass, "remove high frequencies", {"cutoff_hz": Param(1000, 16000)}),
    "highpass": OpDef(highpass, "remove low frequencies", {"cutoff_hz": Param(20, 1000)}),
    "reverb": OpDef(reverb, "add room reverberation", {"rt60_s": Param(0.1, 1.5), "wet": Param(0, 0.5)}),
    "pitch_shift": OpDef(pitch_shift, "shift pitch", {"semitones": Param(-4, 4)}),
    "time_stretch": OpDef(time_stretch, "speed up or slow down", {"rate": Param(0.8, 1.25)}),
    "pad_silence": OpDef(pad_silence, "insert silence at the start", {"seconds": Param(0, 2)}),
    "shift": OpDef(shift, "circular time shift", {"seconds": Param(-2.5, 2.5)}),
    "resample": OpDef(resample, "downsample and upsample back", {"sr_hz": Param(4000, 32000)}),
}
