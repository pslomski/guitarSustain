import numpy as np


def detect_impulse_start_by_envelope(
    envelope_rms, fs, frame_size, pre_seconds=0.1, thresh_factor=6
):
    # estimate noise floor from the first pre_seconds of audio
    n_pre = max(1, int(pre_seconds * fs / frame_size))
    noise_seg = envelope_rms[:n_pre]
    mu = np.mean(noise_seg)
    sigma = np.std(noise_seg)
    thresh = mu + thresh_factor * sigma
    # find first envelope frame above threshold
    above = np.where(envelope_rms > thresh)[0]
    if above.size == 0:
        return None  # no onset found
    frame_idx = above[0]
    time_s = frame_idx * frame_size / fs
    return frame_idx, time_s, thresh


def detect_impulse_start_by_derivative(
    envelope_rms, fs, frame_size, backtrack_frac=0.1
):
    eps = np.finfo(float).eps
    log_env = np.log(np.clip(envelope_rms, eps, None))
    d = np.diff(log_env)
    max_idx = np.argmax(d) + 1  # +1 because diff shifts index
    peak_val = envelope_rms[max_idx]
    # backtrack to fraction of peak to estimate start
    frac = backtrack_frac * peak_val
    candidates = np.where(envelope_rms[:max_idx] <= frac)[0]
    if candidates.size:
        start_idx = candidates[-1] + 1
    else:
        start_idx = max(0, max_idx - int(0.01 * fs / frame_size))  # fallback
    time_s = start_idx * frame_size / fs
    return start_idx, time_s, max_idx


from scipy.signal import find_peaks


def detect_impulse_by_peak(data, fs, height=None, prominence=None):
    abs_data = np.abs(data)
    peaks, props = find_peaks(abs_data, height=height, prominence=prominence)
    if peaks.size == 0:
        return None
    first_peak = peaks[0]
    time_s = first_peak / fs
    return first_peak, time_s, props


def detect_by_matched_filter(data, template, fs):
    # template should be shorter than data and normalized
    corr = np.correlate(data, template[::-1], mode="valid")
    idx = np.argmax(np.abs(corr))
    time_s = idx / fs
    return idx, time_s, corr[idx]
