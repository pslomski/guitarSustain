import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from wavefile import WaveFile
import peak


def get_frame_size(fs, frame_size_ms):
    return int((frame_size_ms / 1000) * fs)


def envelopeRMS(data, frame_size):
    return np.array(
        [
            np.sqrt(np.mean(data[i : i + frame_size] ** 2))
            for i in range(0, len(data), frame_size)
        ]
    )


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
    waveFile = WaveFile()
    fs, data = waveFile.load(filename)
    print(f"Sampling frequency: {fs} Hz")
    frameSize = get_frame_size(fs, frame_size_ms)
    print(f"Frame size: {frameSize} samples ({frameSize/fs*1000:.1f} ms)")
    envelope_rms = envelopeRMS(data, frameSize)

    onset = peak.detect_impulse_start_by_derivative(envelope_rms, fs, frameSize)
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
    plt.title("Signal and it's RMS")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
