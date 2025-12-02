import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import curve_fit

def loadWaveFile(filename):
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
    return np.array([
        np.sqrt(np.mean(data[i:i+frame_size]**2))
        for i in range(0, len(data), frame_size)
    ])

def funExp(x, a, b):
    return a * np.exp(-b * x) 

def funLin(x, m, c):
    return -m * x + c

def main():
    # filename = "data/mono.wav"
    filename = "data/gibson.wav"
    fs, data = loadWaveFile(filename)
    print(f"Sampling frequency: {fs} Hz")
    frameSize = get_frame_size(fs, frame_size_ms=20)
    envelope_rms = envelopeRMS(data, frameSize)
    envelope_logrms = np.log(envelope_rms)  # Avoid log(0)
    time_rms = np.arange(0, len(envelope_rms)) * frameSize / fs

    t_start = int(get_frame_size(fs, 10)/20)
    print(f"t_start:{t_start}")
    time_fit = time_rms[t_start:]
    envelope_rms_fit = envelope_rms[t_start:]
    envelope_logrms_fit = envelope_logrms[t_start:]
    popt, pcov = curve_fit(funExp, time_fit, envelope_rms_fit)
    perr = np.sqrt(np.diag(pcov))
    tHalf = np.log(2) / popt[1]
    print(f"t_half:{tHalf} popt:{popt} perr: {perr}")

    poptLin, pcovLin = curve_fit(funLin, time_fit, envelope_logrms_fit)
    perr = np.sqrt(np.diag(pcovLin))
    tHalf = np.log(2) / poptLin[0]
    print(f"t_half:{tHalf} popt:{poptLin} perr: {perr}")

    time = np.arange(len(data)) / fs
    plt.figure(figsize=(12,6))
    plt.plot(time, data, label="Sygnał oryginalny", alpha=0.5)
    plt.plot(time_rms, envelope_rms, label="RMS (20 ms)", color="green")
    plt.plot(time_rms, funExp(time_rms, *popt), label="e-t", color="blue")
    ax2 = plt.twinx()
    ax2.plot(time_fit, envelope_logrms_fit, label="log(RMS)", color="blue")
    ax2.plot(time_rms, funLin(time_rms, *poptLin), label="log(RMS)", color="blue")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Porównanie obwiedni sygnału różnymi metodami")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
