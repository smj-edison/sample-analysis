import math
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import cmath

import scipy.interpolate
from scipy import signal
from scipy.signal import butter, lfilter, freqz, zoom_fft
from scipy.fft import rfft, fftfreq, fftshift
from scipy.io import wavfile

import pywt

# tuning parameters
assumed_max_trem_speed = 9
assumed_min_trem_speed = 1

analysis_range = [420, 460]

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

def calc_freqs(audio, sample_rate, min_freq, max_freq, steps):
    freqs_to_check = np.linspace(min_freq, max_freq, steps, endpoint=False)
    widths = sample_rate / freqs_to_check

    a, d = pywt.cwt(audio, widths, 'cmor1.5-1.0')

    # plt.imshow(abs(a), extent=[0, audio_length, analysis_range[1], analysis_range[0]], cmap='Spectral', aspect='auto',
    #            vmax=abs(a).max(), vmin=abs(a).min())
    # plt.show()

    maxes = abs(a).argmax(axis=0)

    return lerp(min_freq, max_freq, maxes / len(widths))

def calc_amp(audio):
    return abs(signal.hilbert(audio))

sample_rate, nontremmed = load_audio_mono("./069-A-nt.wav")
sample_rate, tremmed = load_audio_mono("./069-A.wav")

nt_freq = calc_freqs(nontremmed, sample_rate, analysis_range[0], analysis_range[1], 20)
nt_freq_norm = ((nt_freq - np.min(nt_freq)) / (np.max(nt_freq) - np.min(nt_freq))) - 0.5
nt_freq_smoothed = butter_lowpass_filter(nt_freq_norm, assumed_max_trem_speed, sample_rate)
nt_freq_final = (nt_freq_smoothed * (np.max(nt_freq) - np.min(nt_freq))) + np.min(nt_freq)

nt_amp = calc_amp(nontremmed)
nt_amp_norm = ((nt_amp - np.min(nt_amp)) / (np.max(nt_amp) - np.min(nt_amp))) - 0.5
nt_amp_smoothed = butter_lowpass_filter(nt_amp_norm, assumed_max_trem_speed, sample_rate)
nt_amp_final = (nt_amp_smoothed * (np.max(nt_amp) - np.min(nt_amp))) + np.min(nt_amp)


t_freq = calc_freqs(tremmed, sample_rate, analysis_range[0], analysis_range[1], 20)
t_freq_norm = ((t_freq - np.min(t_freq)) / (np.max(t_freq) - np.min(t_freq))) - 0.5
t_freq_smoothed = butter_lowpass_filter(t_freq_norm, assumed_max_trem_speed, sample_rate)
t_freq_final = (t_freq_smoothed * (np.max(t_freq) - np.min(t_freq))) + np.min(t_freq)

t_amp = calc_amp(tremmed)
t_amp_norm = ((t_amp - np.min(t_amp)) / (np.max(t_amp) - np.min(t_amp))) - 0.5
t_amp_smoothed = butter_lowpass_filter(t_amp_norm, assumed_max_trem_speed, sample_rate)
t_amp_final = (t_amp_smoothed * (np.max(t_amp) - np.min(t_amp))) + np.min(t_amp)


min_len = min(len(nt_freq), len(t_freq))
nt_freq = nt_freq[0:min_len]
t_freq = t_freq[0:min_len]

plt.show()

reference_sig = t_freq_smoothed[5000:100000]
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
