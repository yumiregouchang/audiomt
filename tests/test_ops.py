import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from audiomt.ops import OPS, gain, noise, resample, shift
from audiomt.spec import operation_catalog, parse_llm_reply

SR = 22050
rng = np.random.default_rng(1)
Y = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR) + 0.01 * rng.normal(size=SR * 2)).astype("float32")


def mid(p):
    return p.choices[0] if p.choices else (p.lo + p.hi) / 2


def test_every_op_keeps_length_and_is_finite():
    for name, d in OPS.items():
        out = d.fn(Y, SR, **{k: mid(p) for k, p in d.params.items()})
        assert out.shape == Y.shape, name
        assert np.isfinite(out).all(), name


@settings(max_examples=25, deadline=None)
@given(st.floats(10, 40))
def test_noise_hits_requested_snr(snr):
    out = noise(Y, SR, "pink", snr)
    measured = 10 * np.log10(np.mean(Y ** 2) / np.mean((out - Y) ** 2))
    assert abs(measured - snr) < 0.5


def test_identities():
    assert np.allclose(gain(Y, SR, 0.0), Y)
    assert np.allclose(shift(Y, SR, len(Y) / SR), Y)
    assert np.allclose(resample(Y, SR, SR), Y, atol=1e-6)


def test_llm_reply_is_validated_not_executed():
    reply = """```json
    [{"name": "distant_recording", "ops": [{"op": "lowpass", "cutoff_hz": 4000}, {"op": "gain", "db": -9}],
      "expected": "same_label", "applies_to": "all", "rationale": "farther away"},
     {"name": "chipmunk", "ops": [{"op": "pitch_shift", "semitones": 12}], "rationale": "x"},
     {"name": "evil", "ops": [{"op": "os.system", "cmd": "rm -rf /"}], "rationale": "x"}]
    ```"""
    valid, rejected = parse_llm_reply(reply)
    assert [m.name for m in valid] == ["distant_recording"]
    assert len(rejected) == 2
    out = valid[0].apply(Y, SR)
    assert out.shape == Y.shape and out.dtype == np.float32
    assert "pitch_shift" in operation_catalog()
