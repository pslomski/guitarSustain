"""Utilities for detecting the signal tail (end) based on RMS envelope.

Functions
- frame_rms: compute non-overlapping RMS envelopes and frame times
- detect_tail: find the first frame where RMS stays below a threshold for
  a given duration and return an estimated end sample/time
- detect_tail_from_data: convenience wrapper that computes the envelope
  and runs detection; supports a simple heuristic if no threshold provided

Example usage:
    from tail import detect_tail_from_data
    idx_sample, t_end = detect_tail_from_data(data, fs, frame_ms=20,
                                              threshold=1e-4,
                                              min_silence_sec=0.2)
"""

from typing import Optional, Tuple
import numpy as np


def frame_rms(
    data: np.ndarray, fs: int, frame_ms: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute non-overlapping RMS envelope.

    Args:
        data: 1-D or 2-D array (stereo) audio samples. If stereo, mixed to mono.
        fs: sampling rate in Hz
        frame_ms: frame length in milliseconds

    Returns:
        rms: array of RMS values (one per frame)
        times: array of frame start times (seconds)
    """
    if data.ndim > 1:
        data = data.mean(axis=1)
    frame_size = max(1, int((frame_ms / 1000.0) * fs))
    rms = []
    times = []
    N = len(data)
    for i in range(0, N, frame_size):
        frame = data[i : i + frame_size]
        if frame.size == 0:
            break
        rms.append(np.sqrt(np.mean(frame.astype(float) ** 2)))
        times.append(i / fs)
    return np.array(rms), np.array(times)


def detect_tail(
    envelope_rms: np.ndarray,
    frame_size: int,
    fs: int,
    threshold: float,
    min_silence_sec: float = 0.2,
) -> Optional[Tuple[int, float, int]]:
    """Detect start of tail: first frame where RMS stays <= threshold for at
    least `min_silence_sec` seconds.

    Args:
        envelope_rms: array of RMS values (one per frame)
        frame_size: frame length in samples (integer)
        fs: sampling rate
        threshold: amplitude threshold (same units as envelope_rms)
        min_silence_sec: minimum duration of sustained silence (seconds)

    Returns:
        (end_sample, end_time_s, start_frame_idx) where end_sample is the sample
        index where silence starts, end_time_s is the time in seconds, and
        start_frame_idx is the frame index that begins the silent run.
        Returns None if no tail/silence satisfying constraints is found.
    """
    if envelope_rms.size == 0:
        return None
    min_silence_frames = max(1, int(np.ceil(min_silence_sec * fs / frame_size)))

    # boolean array for frames below or equal to threshold
    below = envelope_rms <= threshold

    # find a run of True of length >= min_silence_frames
    consec = 0
    for i, val in enumerate(below):
        if val:
            consec += 1
            if consec >= min_silence_frames:
                start_frame = i - consec + 1
                end_sample = start_frame * frame_size
                end_time_s = end_sample / fs
                return int(end_sample), float(end_time_s), int(start_frame)
        else:
            consec = 0
    return None


def detect_tail_from_data(
    data: np.ndarray,
    fs: int,
    frame_ms: int = 20,
    threshold: Optional[float] = None,
    threshold_db: Optional[float] = None,
    min_silence_sec: float = 0.2,
) -> Optional[Tuple[int, float, int]]:
    """Convenience wrapper: compute RMS envelope and detect tail.

    If neither `threshold` nor `threshold_db` is provided, a simple heuristic
    is used: estimate the noise floor from the last 20% of frames and set
    threshold = median_noise * 2.0 (this works when the true tail is present).

    Args:
        data: audio samples
        fs: sampling rate
        frame_ms: RMS frame length in ms
        threshold: amplitude threshold
        threshold_db: if provided, threshold in dBFS (negative for signals < 1.0)
        min_silence_sec: required silence duration to declare tail

    Returns:
        same as `detect_tail` or None
    """
    envelope_rms, times = frame_rms(data, fs, frame_ms=frame_ms)
    frame_size = max(1, int((frame_ms / 1000.0) * fs))

    if threshold_db is not None:
        # convert from dB (amplitude dBFS) to linear amplitude
        threshold = 10.0 ** (threshold_db / 20.0)

    if threshold is None:
        # heuristic: use last 20% of frames as noise floor
        n = len(envelope_rms)
        n_tail = max(1, int(np.ceil(0.2 * n)))
        noise_seg = envelope_rms[-n_tail:]
        median_noise = float(np.median(noise_seg))
        threshold = max(median_noise * 2.0, np.finfo(float).eps)

    return detect_tail(
        envelope_rms,
        frame_size=frame_size,
        fs=fs,
        threshold=threshold,
        min_silence_sec=min_silence_sec,
    )


if __name__ == "__main__":
    # quick demo using simulated signal
    import matplotlib.pyplot as plt

    fs = 44100
    t = np.arange(0, 2.0, 1 / fs)
    # impulse + exponential decay then silence
    sig = np.exp(-5 * t) * (t < 0.6)
    # add small noise
    sig += 0.0005 * np.random.randn(len(sig))

    res = detect_tail_from_data(sig, fs, frame_ms=20, min_silence_sec=0.12)
    print("detect_tail_from_data ->", res)

    rms, times = frame_rms(sig, fs, frame_ms=20)
    plt.plot(np.arange(len(sig)) / fs, sig, alpha=0.5)
    plt.plot(times, rms, marker="o")
    if res is not None:
        end_sample, end_time, start_frame = res
        plt.axvline(end_time, color="r", linestyle="--", label=f"end {end_time:.3f}s")
    plt.legend()
    plt.show()
