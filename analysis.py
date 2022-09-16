import numpy as np
import pywt
from scipy import signal

from helpers import butter_lowpass_filter

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
    return abs(signal.hilbert(audio))

def analyze_trem(audio, sample_rate, min_freq, max_freq, max_trem_speed=9, freq_steps=20):
    # smooth down signals, then move them back to their original range
    freq = calc_freqs(audio, sample_rate, min_freq, max_freq, freq_steps)
    freq_min, freq_max = np.min(freq), np.max(freq)
    freq_norm = ((freq - freq_min) / (freq_max - freq_min)) - 0.5
    freq_smoothed = butter_lowpass_filter(freq_norm, max_trem_speed, sample_rate)
    freq_final = (freq_smoothed * (freq_max - freq_min)) + freq_min

    amp = calc_amp(audio)
    amp_min, amp_max = np.min(amp), np.max(amp)
    amp_norm = ((amp - amp_min) / (amp_max - amp_min)) - 0.5
    amp_smoothed = butter_lowpass_filter(amp_norm, max_trem_speed, sample_rate)
    amp_final = (amp_smoothed * (amp_max - amp_min)) + amp_min

    return (freq_final, amp_final)
