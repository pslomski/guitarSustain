import numpy as np
import warnings
from scipy.io import wavfile


class WaveFile():
    def __init__(self):
        pass

    def load(self, filename):
        print(f"Loading wave file: {filename}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fs, data = wavfile.read(filename)
        return fs, self.normalize(data)


    def normalize(self, data):
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        else:
            data = data.astype(float)
        return data
