import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, lfilter

# --- 1. Wczytanie sygnału ---
fs, data = wavfile.read("sample.wav")

# Normalizacja do [-1, 1]
if data.dtype == np.int16:
    data = data / 32768.0
elif data.dtype == np.int32:
    data = data / 2147483648.0
else:
    data = data.astype(float)

# --- 2. Obwiednia: wartość bezwzględna + filtr LP ---
abs_signal = np.abs(data)
N = int(0.01 * fs)  # okno 10 ms
b = np.ones(N)/N
a = 1
envelope_abs = lfilter(b, a, abs_signal)

# --- 3. Obwiednia RMS w oknach ---
frame_size = int(0.02 * fs)  # 20 ms
envelope_rms = np.array([
    np.sqrt(np.mean(data[i:i+frame_size]**2))
    for i in range(0, len(data), frame_size)
])
time_rms = np.arange(0, len(envelope_rms)) * frame_size / fs

# --- 4. Obwiednia Hilberta ---
analytic_signal = hilbert(data)
envelope_hilbert = np.abs(analytic_signal)

# --- 5. Wizualizacja ---
time = np.arange(len(data)) / fs

plt.figure(figsize=(12,6))
plt.plot(time, data, label="Sygnał oryginalny", alpha=0.5)
# plt.plot(time, envelope_abs, label="Abs + filtr LP", color="red")
plt.plot(time_rms, envelope_rms, label="RMS (20 ms)", color="green")
# plt.plot(time, envelope_hilbert, label="Hilbert", color="orange")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.title("Porównanie obwiedni sygnału różnymi metodami")
plt.legend()
plt.show()
