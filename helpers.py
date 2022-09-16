import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile

def lerp(a, b, t):
    return (1 - t) * a + t * b

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_audio_mono(filename):
    sample_rate, data = wavfile.read(filename)
    audio_raw = np.array(list(zip(*data))[0]) / 32768

    return (sample_rate, audio_raw)

def norm_sig(signal):
    signal_min, signal_max = np.min(signal), np.max(signal)

    signal_norm = (((signal - signal_min) / (signal_max - signal_min)) * 2) - 1

    (signal_norm, signal_min, signal_max)

def denorm_sig(signal, signal_min, signal_max):
    signal_denorm = (((signal + 1) / 2) * (signal_max - signal_min)) + signal_min

    signal_denorm