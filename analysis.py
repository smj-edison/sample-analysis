import numpy as np
import pywt

from math import pi, floor
import cmath

from scipy.interpolate import interp1d
from scipy.signal import zoom_fft, savgol_filter, hilbert
import matplotlib.pyplot as plt

from helpers import butter_lowpass_filter, denorm_sig, norm_sig, resample_to, hl_envelopes_idx

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

def calc_amp(audio, dmin=300, dmax=300):
    _, low_idx = hl_envelopes_idx(audio, dmin=dmin, dmax=dmax)
    low_idx = np.append(np.insert(low_idx, 0, 0), len(audio) - 1)

    env_points = audio[low_idx]
    env_points_smoothed = savgol_filter(env_points, 3, 2)

    # to prevent oscillations when doing the cubic interpolation
    interp = interp1d(low_idx, env_points_smoothed, kind='linear')(range(0, len(audio), 10))

    return resample_to(savgol_filter(interp, 1000, 3), len(audio), kind='cubic')

def calc_amp_hilbert(audio):
    return abs(hilbert(audio))

def get_sig_freq_and_amp(sig, sample_rate, min_freq, max_freq, lp_freq=20, freq_steps=20):
    # smooth down signals, then move them back to their original range
    freq = calc_freqs(sig, sample_rate, min_freq, max_freq, freq_steps)
    freq_norm, freq_min, freq_max = norm_sig(freq)
    freq_smoothed = butter_lowpass_filter(freq_norm, lp_freq, sample_rate)
    freq_final = denorm_sig(freq_smoothed, freq_min, freq_max)

    amp = calc_amp(sig, dmin=100, dmax=100)
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

    # average up the chunks
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

def get_best_roll(sig_to_roll, sig_ref):
    roll_by = 0
    largest = 0

    for i in range(0, len(sig_to_roll)):
        res = sig_to_roll.dot(sig_ref)

        if res > largest:
            largest = res
            roll_by = i

        sig_to_roll = np.roll(sig_to_roll, 1)

    return roll_by

def calc_trem_table(
    audio,
    note_freq,
    sample_rate,
    freq_spread=100,
    freq_steps=20,
    trem_steps=512,
    lp_freq=20,
    min_trem_speed=1,
    max_trem_speed=9,
    plot_results=False,
):
    lower_test_bound = note_freq * (2 ** (-freq_spread / 1200))
    higher_test_bound = note_freq * (2 ** (freq_spread / 1200))

    t_freq, t_amp = get_sig_freq_and_amp(
        audio,
        sample_rate,
        lower_test_bound,
        higher_test_bound,
        lp_freq=lp_freq,
        freq_steps=freq_steps
    )

    freq_trem, freq_trem_offset = calc_trem_shape(
        t_freq,
        sample_rate,
        min_trem_speed=min_trem_speed,
        max_trem_speed=max_trem_speed
    )
    amp_trem, amp_trem_offset = calc_trem_shape(
        t_amp,
        sample_rate,
        min_trem_speed=min_trem_speed,
        max_trem_speed=max_trem_speed
    )

    trem_freq = sample_rate / len(amp_trem)

    freq_trem_table = resample_to(freq_trem, trem_steps)
    amp_trem_table = resample_to(amp_trem, trem_steps)

    freq_trem_norm = norm_sig(freq_trem_table)[0]
    amp_trem_norm = norm_sig(amp_trem_table)[0]

    # line up phase in trem tables
    roll_by = get_best_roll(amp_trem_norm, freq_trem_norm)
    amp_trem_table = np.roll(amp_trem_table, roll_by)

    if plot_results:
        amp_trem_norm = norm_sig(amp_trem_table)[0]

        plt.plot(amp_trem_norm)
        plt.plot(freq_trem_norm)
        plt.show()

    return freq_trem_table, amp_trem_table, trem_freq
