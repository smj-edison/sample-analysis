from math import floor, ceil, pi, sin

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.io import wavfile

from helpers import hermite_interp, lerp, load_audio_mono, norm_sig, apply_filter, butter_lowpass
from analysis import get_sig_freq_and_amp, calc_trem_shape

two_pi = pi * 2

# tuning parameters
assumed_max_trem_speed = 9
assumed_min_trem_speed = 1

trem_steps = 512

analysis_range = [420, 460]


sample_rate, nontremmed = load_audio_mono("./069-A-nt.wav")
sample_rate, tremmed = load_audio_mono("./069-A.wav")

nt_freq, nt_amp = get_sig_freq_and_amp(nontremmed, sample_rate, analysis_range[0], analysis_range[1])
base_freq = np.mean(nt_freq)
base_amp = np.mean(nt_amp)

t_freq, t_amp = get_sig_freq_and_amp(tremmed, sample_rate, analysis_range[0], analysis_range[1])

min_len = min(len(nt_freq), len(t_freq))
nt_freq = nt_freq[0:min_len]
t_freq = t_freq[0:min_len]

freq_trem, freq_trem_offset = calc_trem_shape(t_freq, sample_rate, plot_results=False)
amp_trem, amp_trem_offset = calc_trem_shape(t_amp, sample_rate, plot_results=False)

freq_trem_table = np.interp(np.linspace(0, 1, trem_steps), np.linspace(0, 1, len(freq_trem)), freq_trem)
amp_trem_table = np.interp(np.linspace(0, 1, trem_steps), np.linspace(0, 1, len(amp_trem)), amp_trem)

freq_trem_norm = norm_sig(freq_trem_table)[0]
amp_trem_norm = norm_sig(amp_trem_table)[0]

def lookup(table, table_freq, pos):
    pos_in_table = pos * len(table) * table_freq

    return lerp(table[floor(pos_in_table) % len(table)], table[ceil(pos_in_table) % len(table)], pos_in_table % 1)

def val_or_zero(array, index):
    if index < 0:
        return 0

    try:
        return array[index]
    except IndexError:
        return 0

plt.plot(freq_trem_norm)
plt.plot(amp_trem_norm)
plt.show()

audio_out = []
audio_loc = 0
global_loc = 0
table_freq = 6.3
detune = 1.1

while audio_loc < (len(nontremmed) - detune * 3) / sample_rate:
    pos_in_audio = audio_loc * sample_rate

    target_freq = lookup(freq_trem_table, table_freq, global_loc)
    #detune = target_freq / base_freq

    target_amp = lookup(amp_trem_table, table_freq, global_loc)
    gain = target_amp / base_amp

    sample0 = nontremmed[floor(pos_in_audio - 1)]
    sample1 = nontremmed[floor(pos_in_audio + 0)]
    sample2 = nontremmed[floor(pos_in_audio + 1)]
    sample3 = nontremmed[floor(pos_in_audio + 2)]

    sample = hermite_interp(sample0, sample1, sample2, sample3, pos_in_audio % 1) * gain

    # if detune > 1:
    #     # better filter out so we don't hit nyquist
    #     cutoff = (sample_rate / 2) / detune
    #     b, a = butter_lowpass(cutoff, sample_rate)

    #     sample = apply_filter(
    #         b, a,
    #         sample,
    #         val_or_zero(audio_out, floor(pos_in_audio - 1)),
    #         val_or_zero(audio_out, floor(pos_in_audio - 2))
    #     )

    audio_out.append(sample)


    audio_loc += detune / sample_rate
    global_loc += 1 / sample_rate

wavfile.write("out.wav", sample_rate, np.array(audio_out))

