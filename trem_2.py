import math
from math import floor, pi, e

import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up

from scipy.io import wavfile
from scipy.signal import zoom_fft
from scipy.interpolate import interp1d

from helpers import (
    load_audio_mono,
    norm_sig,
    denorm_sig,
    butter_lowpass_filter,
    resample_to,
    gain_to_db,
    detune_to_cents,
    lerp,
)

fs, tremmed = load_audio_mono("./samples/069-A.wav")
fs, nontremmed = load_audio_mono("./samples/069-A-nt.wav")
freq = 440


def viz(Tx, Wx):
    plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo")
    plt.show()
    plt.imshow(np.abs(Tx), aspect="auto", vmin=0, vmax=0.2, cmap="turbo")
    plt.show()


def ranged_cwt(sig, fmin, fmax, fs, nv=256):
    N = len(sig)

    wavelet = Wavelet("gmw", N=p2up(N)[0])

    min_scale, max_scale = cwt_scalebounds(wavelet, N)
    scales = make_scales(
        N,
        (fs / 2) / fmax * min_scale,
        (fs / 2) / fmin * min_scale,
        wavelet=wavelet,
        nv=nv,
    )

    return ssq_cwt(sig, wavelet, scales)


# find tremulant period
def find_period(sig, fmin, fmax, fs):
    freq_amps = np.abs(zoom_fft(sig, [fmin, fmax], fs=fs))

    # find peak amplitude
    peak = np.max(freq_amps)

    # find lowest frequency above at least 6/10ths of peak
    lowest_matching = np.where(freq_amps > peak * 0.6)[0][0]

    # localized maximum
    lower_bounds = int(max(lowest_matching - len(sig) / 20, 0))
    upper_bounds = int(min(lowest_matching + len(sig) / 20, len(sig)))

    local_max = np.argmax(freq_amps[lower_bounds:upper_bounds]) + lower_bounds

    frequency = lerp(fmin, fmax, local_max / len(sig))

    return fs / frequency


def calc_trem_shape(sig, fs, min_trem_speed=0.5, max_trem_speed=15):
    period = find_period(sig, min_trem_speed, max_trem_speed, fs)

    slices = np.array_split(sig, round(period))

    slices = []
    for i in np.arange(0, len(sig), period):
        slices.append(sig[floor(i) : (floor(period) + floor(i))])

    # discard the last one if it's too short
    if len(slices[-1]) != len(slices[0]):
        slices.pop()

    trem_chunks = np.stack(slices)
    avg_trem = np.mean(trem_chunks, axis=0)

    return avg_trem, fs / period


def calc_freqs_amps(sig, note_freq, fs):
    Tx, Wx, ssq_freqs, *_ = ranged_cwt(sig, note_freq * 0.8, note_freq * 1.3, fs)
    peak_freqs = ssq_freqs[np.argmax(np.abs(Tx), axis=0)]

    amps = np.max(np.abs(Wx), axis=0)

    # trim off padding
    return peak_freqs[1000:-1000] * fs, amps[1000:-1000]


def calc_trem_table(sig, note_freq, fs, table_len=512):
    freqs_unnorm, amps_unnorm = calc_freqs_amps(sig, note_freq, fs)

    # normalize and smooth signal
    freqs, freqs_min, freqs_max = norm_sig(np.log2(freqs_unnorm))
    amps, amps_min, amps_max = norm_sig(amps_unnorm)

    amps = butter_lowpass_filter(amps, 20, fs, order=1)
    freqs = butter_lowpass_filter(freqs, 20, fs, order=1)

    amp_trem, amp_speed = calc_trem_shape(amps, fs)
    # amp trem speed is much more reliable
    freq_trem, freq_speed = calc_trem_shape(
        freqs, fs, min_trem_speed=amp_speed - 0.2, max_trem_speed=amp_speed + 0.2
    )

    freq_trem_unnorm = 2 ** denorm_sig(freq_trem, freqs_min, freqs_max)
    amp_trem_unnorm = denorm_sig(amp_trem, amps_min, amps_max)

    return (
        resample_to(amp_trem_unnorm, table_len),
        resample_to(freq_trem_unnorm, table_len),
        (freq_speed + amp_speed) / 2.0,
    )


def calculate_shelf_table(nontremmed, tremmed, note_freq, fs, table_len=512):
    # look at third harmonic
    Tx, Wx, ssq_freqs, *_ = ranged_cwt(
        nontremmed, (note_freq * 3) * 0.7, (note_freq * 3) * 1.5, fs
    )

    nontrem_amp = np.mean(np.max(np.abs(Tx), axis=0))

    # now look at third harmonic of tremmed version
    Tx, Wx, ssq_freqs, *_ = ranged_cwt(
        tremmed, (note_freq * 3) * 0.7, (note_freq * 3) * 1.5, fs
    )

    trem_amps = np.max(np.abs(Tx), axis=0)
    trem_amps = butter_lowpass_filter(trem_amps, 50, fs, order=1)

    harmonic_trem, _ = calc_trem_shape(trem_amps, fs)

    harmonic_trem = harmonic_trem / nontrem_amp

    return resample_to(harmonic_trem, table_len)


nontremmed_freqs, nontremmed_amps = calc_freqs_amps(nontremmed[100000:300000], freq, fs)
nontremmed_freq = np.mean(nontremmed_freqs)
nontremmed_amp = np.mean(nontremmed_amps)

tremmed_start = 10000
tremmed_end = len(tremmed) - tremmed_start
amps, freqs, speed = calc_trem_table(tremmed[tremmed_start:tremmed_end], freq, fs)

detune_trem_table = detune_to_cents(freqs / nontremmed_freq)
gain_trem_table = gain_to_db(amps / nontremmed_amp)

shelf_trem_table = calculate_shelf_table(
    nontremmed[tremmed_start:tremmed_end], tremmed[tremmed_start:tremmed_end], freq, fs
)

# how much are the upper harmonics affected, _in addition_ to what changed due to
# overall gain changes?
shelf_trem_table = gain_to_db(shelf_trem_table) - gain_trem_table

plt.scatter(gain_trem_table, detune_trem_table, c=shelf_trem_table)
plt.title("Pipe gain vs frequency")
plt.xlabel("Gain (dB)")
plt.ylabel("Detune from nontremmed (cents)")
plt.colorbar()
plt.show()

wavfile.write("detune-table.wav", fs, detune_trem_table)
wavfile.write("gain-table.wav", fs, gain_trem_table)
wavfile.write("shelf-table.wav", fs, shelf_trem_table)
