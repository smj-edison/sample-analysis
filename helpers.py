import numpy as np
from math import floor, ceil
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

def apply_filter(b, a, x, xm1, xm2):
    return b[0]*x + b[1]*xm1 + b[2]*xm2 - a[1]*xm1 + a[2]*xm2

def load_audio_mono(filename):
    sample_rate, data = wavfile.read(filename)

    if len(data.shape) > 1:
        audio_raw = np.array(list(zip(*data))[0]) / 32768
    else:
        audio_raw = np.array(data) / 32768

    return (sample_rate, audio_raw)

def norm_sig(signal):
    signal_min, signal_max = np.min(signal), np.max(signal)

    signal_norm = (((signal - signal_min) / (signal_max - signal_min)) * 2) - 1

    return (signal_norm, signal_min, signal_max)

def denorm_sig(signal, signal_min, signal_max):
    signal_denorm = (((signal + 1) / 2) * (signal_max - signal_min)) + signal_min

    return signal_denorm

def calc_rms(signal):
    return np.sqrt(np.mean(signal**2))

def resample_to(signal, output_length):
    return np.interp(np.linspace(0, 1, output_length), np.linspace(0, 1, len(signal)), signal)

def table_lookup(table, table_freq, pos):
    pos_in_table = pos * len(table) * table_freq

    return lerp(table[floor(pos_in_table) % len(table)], table[ceil(pos_in_table) % len(table)], pos_in_table % 1)

# https://stackoverflow.com/questions/1125666/how-do-you-do-bicubic-or-other-non-linear-interpolation-of-re-sampled-audio-da
def hermite_interp(x0, x1, x2, x3, t):
    c0 = x1
    c1 = 0.5 * (x2 - x0)
    c2 = x0 - (2.5 * x1) + (2 * x2) - (0.5 * x3)
    c3 = (0.5 * (x3 - x0)) + (1.5 * (x1 - x2))

    return (((((c3 * t) + c2) * t) + c1) * t) + c0
