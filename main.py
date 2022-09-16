import math
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import cmath

import scipy.interpolate
from scipy import signal
from scipy.signal import butter, lfilter, freqz, zoom_fft
from scipy.fft import rfft, fftfreq, fftshift

from helpers import load_audio_mono, norm_sig, denorm_sig
from analysis import calc_freqs, calc_amp, analyze_trem

# tuning parameters
assumed_max_trem_speed = 9
assumed_min_trem_speed = 1

analysis_range = [420, 460]


sample_rate, nontremmed = load_audio_mono("./069-A-nt.wav")
sample_rate, tremmed = load_audio_mono("./069-A.wav")

nt_freq, nt_amp = analyze_trem(nontremmed, sample_rate, analysis_range[0], analysis_range[1])
t_freq, t_amp = analyze_trem(tremmed, sample_rate, analysis_range[0], analysis_range[1])


min_len = min(len(nt_freq), len(t_freq))
nt_freq = nt_freq[0:min_len]
t_freq = t_freq[0:min_len]

plt.show()

reference_sig = t_freq_smoothed
sig = zoom_fft(reference_sig, [assumed_min_trem_speed, assumed_max_trem_speed], fs=sample_rate)

freqs = np.linspace(assumed_min_trem_speed, assumed_max_trem_speed, len(reference_sig))

# identify broad range
biggest_freq = np.argmax(abs(sig))

phase = (cmath.phase(sig[biggest_freq]) + (math.pi*2)) % (math.pi*2)
remaining_phase = (math.pi*2) - phase
freq = freqs[biggest_freq]

# get phase offset in terms of position in time
offset = (remaining_phase / (math.pi*2) * sample_rate) / freq
freq_in_time = sample_rate / freq


# chop the signal up (cutting off the beginning)
slices = []
for i in np.arange(offset, len(reference_sig), freq_in_time):
    slices.append(reference_sig[floor(i):(floor(freq_in_time) + floor(i))])

# discard the last one if it's too short
if len(slices[-1]) != len(slices[0]):
    slices.pop()

trem_chunks = np.stack(slices)
avg_trem = np.mean(trem_chunks, axis=0)
avg_trem_final = (avg_trem * (np.max(t_freq) - np.min(t_freq))) + np.min(t_freq)

reference_sig_final = (reference_sig * (np.max(t_freq) - np.min(t_freq))) + np.min(t_freq)

plt.plot(reference_sig_final)

for i in np.arange(offset, len(reference_sig_final), freq_in_time):
    plt.axvline(x=(i + 0.0), color="black")
    plt.plot(range(floor(i), floor(i) + len(slices[0]), 1), avg_trem_final, color="r")

plt.show()

# find the biggest frequency

# freqs = (freqs - freqs.min()) / (freqs.max() - freqs.min()) - 0.5
# amplitudes = (amplitudes - amplitudes.min()) / (amplitudes.max() - amplitudes.min()) - 0.5

# start = 5000
# end = len(freqs) - (48000 * 5)

# freqs = freqs[start:end]
# amplitudes = amplitudes[start:end]

# plt.plot(signal.savgol_filter(freqs, 51, 3))
# plt.plot(amplitudes)
# plt.show()
