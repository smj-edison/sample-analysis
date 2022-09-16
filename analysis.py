import numpy as np
import pywt

from math import pi, floor
import cmath
from scipy.signal import zoom_fft, hilbert
import matplotlib.pyplot as plt

from helpers import butter_lowpass_filter, denorm_sig, norm_sig

two_pi = pi * 2

def calc_freqs(audio, sample_rate, min_freq, max_freq, steps):
    freqs_to_check = np.linspace(min_freq, max_freq, steps, endpoint=False)
    widths = sample_rate / freqs_to_check

    a, _ = pywt.cwt(audio, widths, 'cmor1.5-1.0')

    # plt.imshow(abs(a), extent=[0, audio_length, analysis_range[1], analysis_range[0]], cmap='Spectral', aspect='auto',
    #            vmax=abs(a).max(), vmin=abs(a).min())
    # plt.show()

    maxes = abs(a).argmax(axis=0)

    return freqs_to_check[maxes]

def calc_amp(audio):
    return abs(hilbert(audio))

def get_sig_freq_and_amp(sig, sample_rate, min_freq, max_freq, lp_freq=20, freq_steps=20):
    # smooth down signals, then move them back to their original range
    freq = calc_freqs(sig, sample_rate, min_freq, max_freq, freq_steps)
    freq_norm, freq_min, freq_max = norm_sig(freq)
    freq_smoothed = butter_lowpass_filter(freq_norm, lp_freq, sample_rate)
    freq_final = denorm_sig(freq_smoothed, freq_min, freq_max)

    amp = calc_amp(sig)
    amp_norm, amp_min, amp_max = norm_sig(amp)
    amp_smoothed = butter_lowpass_filter(amp_norm, lp_freq, sample_rate)
    amp_final = denorm_sig(amp_smoothed, amp_min, amp_max)

    return (freq_final, amp_final)

def calc_trem_shape(sig_unnorm, sample_rate, min_trem_speed=1, max_trem_speed=9, plot_results=False):
    sig, sig_min, sig_max = norm_sig(sig_unnorm)

    analysis = zoom_fft(sig, [min_trem_speed, max_trem_speed], fs=sample_rate)
    analysis_freqs = np.linspace(min_trem_speed, max_trem_speed, len(sig))

    # identify the frequency with most amplitude
    biggest_freq = np.argmax(abs(analysis))

    phase = (cmath.phase(analysis[biggest_freq]) + two_pi) % two_pi
    remaining_phase = two_pi - phase
    freq = analysis_freqs[biggest_freq]

    # get phase offset in terms of position in time
    offset = (remaining_phase / two_pi * sample_rate) / freq
    freq_in_time = sample_rate / freq

    # chop the signal up based on detected frequency (cutting off the beginning)
    slices = []
    for i in np.arange(offset, len(sig), freq_in_time):
        slices.append(sig[floor(i):(floor(freq_in_time) + floor(i))])

    # discard the last one if it's too short
    if len(slices[-1]) != len(slices[0]):
        slices.pop()

    trem_chunks = np.stack(slices)
    avg_trem = np.mean(trem_chunks, axis=0)
    avg_trem_final = denorm_sig(avg_trem, sig_min, sig_max)

    if plot_results:
        plt.plot(sig_unnorm)

        for i in np.arange(offset, len(sig), freq_in_time):
            plt.axvline(x=(i + 0.0), color="black")
            plt.plot(range(floor(i), floor(i) + len(slices[0]), 1), avg_trem_final, color="r")

        plt.show()

    return avg_trem_final, offset
