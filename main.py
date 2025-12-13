import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.io import wavfile
from scipy.optimize import curve_fit


def loadWaveFile(filename):
    print(f"Loading wave file: {filename}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fs, data = wavfile.read(filename)
    return fs, normalize(data)


def normalize(data):
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    else:
        data = data.astype(float)
    return data


def get_frame_size(fs, frame_size_ms):
    return int((frame_size_ms / 1000) * fs)


def envelopeRMS(data, frame_size):
    return np.array(
        [
            np.sqrt(np.mean(data[i : i + frame_size] ** 2))
            for i in range(0, len(data), frame_size)
        ]
    )


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


def funExp(x, a, b):
    return a * np.exp(-b * x)


def funLin(x, m, c):
    return -m * x + c


def dumpWaveToTxt(data, fs, output_file="wave_dump.txt"):
    time = np.arange(len(data)) / fs
    with open(output_file, "w") as f:
        f.write("time(s)\tamplitude\n")
        for t, amp in zip(time, data):
            f.write(f"{t:.6f}\t{amp}\n")
    print(f"Wave data dumped to {output_file}")


frame_size_ms = 20  # frame size for RMS calculation in milliseconds


def main():
    # filename = "data/gibson.wav"
    # filename = "data/hagstrom.wav"
    # filename = "data/squire.wav"
    # filename = "data/telecaster.wav"
    filename = "data/witkowski.wav"
    fs, data = loadWaveFile(filename)
    print(f"Sampling frequency: {fs} Hz")
    frameSize = get_frame_size(fs, frame_size_ms)
    print(f"Frame size: {frameSize} samples ({frameSize/fs*1000:.1f} ms)")
    envelope_rms = envelopeRMS(data, frameSize)

    onset = detect_impulse_start_by_derivative(envelope_rms, fs, frameSize)
    if onset is not None:
        start, time_s, max_idx = onset
        print(
            f"Impulse onset detected at frame {start}, time {time_s:.3f} s, max_idx {max_idx}"
        )
    time_rms = np.arange(0, len(envelope_rms)) * frameSize / fs

    t_start = max_idx + 10  # fit from detected onset frame
    time_fit = time_rms[t_start:]
    envelope_rms_fit = envelope_rms[t_start:]
    popt, pcov = curve_fit(funExp, time_fit, envelope_rms_fit)
    perr = np.sqrt(np.diag(pcov))
    tHalf = np.log(2) / popt[1] * 1000  # in milliseconds
    print(f"t half:{int(tHalf)} ms, error:{perr}")

    envelope_rms_lin_fit = np.log(envelope_rms[t_start:])
    poptLin, pcovLin = curve_fit(funLin, time_fit, envelope_rms_lin_fit)
    perrLin = np.sqrt(np.diag(pcovLin))
    tHalfLin = np.log(2) / poptLin[0] * 1000  # in milliseconds
    print(f"t half:{int(tHalfLin)} ms, error:{perrLin}")

    time = np.arange(len(data)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label="Sygnał oryginalny", alpha=0.5)
    plt.plot(time_rms, envelope_rms, label="RMS (10 ms)", color="green")
    plt.plot(time_fit, funExp(time_fit, *popt), label="e^(-at)", color="blue")
    # plt.plot(time_fit, funLin(time_fit, *poptLin), label="ax+b", color="red")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Krzywa zaniku sygnału")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
