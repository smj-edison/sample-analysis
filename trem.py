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

analysis_range = [120, 140]
freq = 130


sample_rate, nontremmed = load_audio_mono("./test-samples/048-C-nt.wav")
sample_rate, tremmed = load_audio_mono("./test-samples/048-C.wav")

nt_freq, nt_amp = get_sig_freq_and_amp(nontremmed, sample_rate, analysis_range[0], analysis_range[1])
base_freq = np.mean(nt_freq)
base_amp = np.mean(nt_amp)

freq_trem_table, amp_trem_table, trem_freq = calc_trem_table(tremmed, freq, sample_rate, plot_results=True)

detune_trem_table = freq_trem_table / base_freq
gain_trem_table = amp_trem_table / base_amp

def audio_lookup(pos, audio):
    sample0 = audio[floor(pos_in_audio - 1)]
    sample1 = audio[floor(pos_in_audio + 0)]
    sample2 = audio[floor(pos_in_audio + 1)]
    sample3 = audio[floor(pos_in_audio + 2)]

    return hermite_interp(sample0, sample1, sample2, sample3, pos % 1) * gain

audio_out = []
audio_loc = 1/sample_rate
global_loc = 0
detune = 1

output_audio_len = len(nontremmed) - 4

while audio_loc < output_audio_len / sample_rate:
    pos_in_audio = audio_loc * sample_rate

    detune = table_lookup(detune_trem_table, trem_freq, global_loc)
    gain = table_lookup(gain_trem_table, trem_freq, global_loc) * 0.3

    sample = audio_lookup(pos_in_audio, nontremmed) * gain

    audio_out.append(sample)


    audio_loc += detune / sample_rate
    global_loc += 1 / sample_rate

wavfile.write("out.wav", sample_rate, np.array(audio_out))
