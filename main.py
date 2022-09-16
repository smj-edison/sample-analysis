from math import floor, ceil, pi, sin

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.io import wavfile

from helpers import hermite_interp, load_audio_mono, norm_sig, resample_to, table_lookup
from analysis import get_sig_freq_and_amp, calc_trem_shape, get_best_roll

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


freq_trem, freq_trem_offset = calc_trem_shape(t_freq, sample_rate, plot_results=False)
amp_trem, amp_trem_offset = calc_trem_shape(t_amp, sample_rate, plot_results=False)

freq_trem_table = resample_to(freq_trem, trem_steps)
amp_trem_table = resample_to(amp_trem, trem_steps)

freq_trem_norm = norm_sig(freq_trem_table)[0]
amp_trem_norm = norm_sig(amp_trem_table)[0]


audio_out = []
audio_loc = 0
global_loc = 0
table_freq = 6.3
detune = 1

while audio_loc < (len(nontremmed) - 4) / sample_rate:
    pos_in_audio = audio_loc * sample_rate

    target_freq = table_lookup(freq_trem_table, table_freq, global_loc)
    detune = target_freq / base_freq

    target_amp = table_lookup(amp_trem_table, table_freq, global_loc)
    gain = target_amp / base_amp

    sample0 = nontremmed[floor(pos_in_audio - 1)]
    sample1 = nontremmed[floor(pos_in_audio + 0)]
    sample2 = nontremmed[floor(pos_in_audio + 1)]
    sample3 = nontremmed[floor(pos_in_audio + 2)]

    sample = hermite_interp(sample0, sample1, sample2, sample3, pos_in_audio % 1) * gain

    audio_out.append(sample)


    audio_loc += detune / sample_rate
    global_loc += 1 / sample_rate

wavfile.write("out.wav", sample_rate, np.array(audio_out))

