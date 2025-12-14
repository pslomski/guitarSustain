import os
import numpy as np
from scipy.io import wavfile

from wavefile import WaveFile


def _write_sine_int16(path, sr=44100, duration_s=0.1, freq=440.0, amp=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (amp * np.sin(2 * np.pi * freq * t) * 32767.0).astype(np.int16)
    wavfile.write(path, sr, samples)


def test_load_and_normalize(tmp_path):
    wf = WaveFile()
    p = tmp_path / "test.wav"
    _write_sine_int16(str(p), sr=44100)

    fs, data = wf.load(str(p))

    # sampling rate preserved
    assert fs == 44100

    # returned data should be float
    assert data.dtype == float

    # values should be in roughly [-1, 1]
    assert data.min() >= -1.0 - 1e-6
    assert data.max() <= 1.0 + 1e-6


def test_normalize_int32_and_float(tmp_path):
    wf = WaveFile()

    # int32: write a small file then read directly using scipy to pass to normalize
    arr32 = (np.linspace(-2147483648, 2147483647, num=1000)).astype(np.int32)
    # scipy wavfile.write expects ints and a valid sample rate, but to avoid
    # writing a large file we'll just call normalize directly here
    normed = wf.normalize(arr32)
    assert normed.dtype == float
    assert np.isfinite(normed).all()

    # float input should be returned as float with same values
    floats = np.random.RandomState(0).randn(100).astype(float)
    out = wf.normalize(floats)
    assert out.dtype == float
    assert np.allclose(out, floats)
