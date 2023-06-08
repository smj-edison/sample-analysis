from math import floor, pi, e

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import zoom_fft

from helpers import (
    load_audio_mono,
)

fs, sample = load_audio_mono("./test-samples/069-A-nt.wav")
freq = 440


def amp_to_db(amp):
    return 20.0 * np.log10(amp)


def comb_amp_response(freq, fs, M, alpha):
    omega = 2 * pi * (freq / fs)

    b0 = 1
    bm = alpha

    return abs(b0 + bm * (e ** (omega * M * 1j)))


def calculate_harmonics(sample, freq, fs, harmonics=16):
    # spectral profile
    sp = np.abs(zoom_fft(sample, [freq / 2, freq * harmonics], fs=fs))
    freqs = np.linspace(freq / 2, freq * 16, len(sample))

    # expected frequencies
    exp_freqs = np.arange(freq, freq * 16, freq)

    search_width = int(len(sample) / harmonics / 2)

    out_freqs = []
    out_amps = []

    for target_freq in exp_freqs:
        # where in the spectral profile is the frequency we're looking for?
        index = np.argmin(np.abs(freqs - target_freq))

        # look around it to find the peak
        search_at = int(index - search_width)

        peak_idx = search_at + np.argmax(sp[search_at : (search_at + search_width)])

        out_freqs.append(freqs[peak_idx])
        out_amps.append(sp[peak_idx])

    return out_freqs, out_amps


harmonics = calculate_harmonics(sample, freq, fs)
plt.stem(freqs, amp_to_db(amps / np.max(amps)), bottom=-120)
plt.xscale("log")

M = int(fs / (freq * 2))
alpha = -0.7

shifted = (sample * alpha)[:-M]
signal = sample[M:]

filtered = (shifted + signal) / comb_amp_response(freq, fs, M, alpha)

# spectral profile
sp = np.abs(zoom_fft(sample, [freq / 2, freq * 16], fs=fs))
sp_filtered = np.abs(zoom_fft(filtered, [freq / 2, freq * 16], fs=fs))

wavfile.write("filtered.wav", fs, filtered)
