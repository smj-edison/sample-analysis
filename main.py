from math import floor, ceil, pi, sin

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.io import wavfile

from helpers import hermite_interp, load_audio_mono, norm_sig, resample_to, table_lookup
from analysis import get_sig_freq_and_amp, calc_trem_table

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

freq_trem_table, amp_trem_table = calc_trem_table(tremmed, 440, sample_rate, plot_results=True)

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

