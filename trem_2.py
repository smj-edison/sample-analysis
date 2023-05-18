from math import floor

import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up
from ssqueezepy.experimental import scale_to_freq, freq_to_scale

from scipy.io import wavfile
from scipy.signal import kaiserord, lfilter, firwin, zoom_fft
from scipy.interpolate import interp1d

from helpers import (
    load_audio_mono,
    norm_sig,
    denorm_sig,
    butter_lowpass_filter,
    resample_to,
)

fs, tremmed = load_audio_mono("./test-samples/072-C.wav")
fs, nontremmed = load_audio_mono("./test-samples/072-C-nt.wav")
freq = 523.25


def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo")
    plt.show()
    plt.imshow(np.abs(Tx), aspect="auto", vmin=0, vmax=0.2, cmap="turbo")
    plt.show()


def ranged_cwt(sig, fmin, fmax, fs, nv=128):
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

    return ssq_cwt(tremmed, wavelet, scales)


# find tremulant period
def find_period(sig, fmin, fmax, fs):
    freq_amps = np.abs(zoom_fft(sig, [fmin, fmax], fs=fs))

    # find peak amplitude
    peak = np.max(freq_amps)

    # find lowest frequency above at least half of peak
    lowest = np.where(freq_amps > peak * 0.5)[0][0]

    # localized maximum
    lower_bounds = int(max(lowest - len(sig) / 20, 0))
    upper_bounds = int(min(lowest + len(sig) / 20, len(sig)))

    local_max = np.argmax(freq_amps[lower_bounds:upper_bounds]) + lower_bounds

    frequency = interp1d([0, len(sig)], [fmin, fmax])(local_max)

    return fs / frequency


def calc_trem_shape(sig, fs):
    period = find_period(sig, 0.5, 20, fs)

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


def calc_freqs(sig, note_freq, fs):
    Tx, Wx, ssq_freqs, *_ = ranged_cwt(sig, note_freq * 0.7, note_freq * 1.5, fs)
    peak_freqs = ssq_freqs[np.argmax(np.abs(Tx), axis=0)]

    return peak_freqs * fs


def calc_amps(sig, note_freq, fs):
    window_size = int(fs / note_freq * 1.1)
    sliding_rms = np.sqrt(
        np.convolve(np.square(sig), np.ones(window_size), "valid") / window_size
    )[(window_size // 2) :]

    return sliding_rms


def calc_trem_table(sig, note_freq, fs, table_len=512):
    freqs_unnorm = calc_freqs(sig, note_freq, fs)
    amps_unnorm = calc_amps(sig, note_freq, fs)

    # normalize and smooth signal
    freqs, freqs_min, freqs_max = norm_sig(np.log2(freqs_unnorm))
    amps, amps_min, amps_max = norm_sig(amps_unnorm)

    amps = butter_lowpass_filter(amps, 50, fs, order=1)
    freqs = butter_lowpass_filter(freqs, 50, fs, order=1)

    freq_trem, _ = calc_trem_shape(freqs, fs)
    amp_trem, _ = calc_trem_shape(amps, fs)

    freq_trem_unnorm = 2 ** denorm_sig(freq_trem, freqs_min, freqs_max)
    amp_trem_unnorm = denorm_sig(amp_trem, amps_min, amps_max)

    return resample_to(amp_trem_unnorm, table_len), resample_to(
        freq_trem_unnorm, table_len
    )


def calculate_shelf_table(nontremmed, tremmed, note_freq, fs, table_len=512):
    # look at third harmonic
    Tx, Wx, ssq_freqs, *_ = ranged_cwt(
        nontremmed, (note_freq * 3) * 0.7, (note_freq * 3) * 1.5, fs
    )

    nontrem_amp = np.mean(np.max(np.abs(Wx), axis=0))

    # now look at third harmonic of tremmed version
    Tx, Wx, ssq_freqs, *_ = ranged_cwt(
        tremmed, (note_freq * 3) * 0.7, (note_freq * 3) * 1.5, fs
    )

    trem_amps = np.max(np.abs(Wx), axis=0)
    harmonic_trem, _ = calc_trem_shape(trem_amps, fs)

    harmonic_trem = harmonic_trem / nontrem_amp

    return resample_to(harmonic_trem, table_len)


nontremmed_freq = np.mean(calc_freqs(nontremmed[10000:200000], freq, fs))
nontremmed_amp = np.mean(calc_amps(nontremmed[10000:200000], freq, fs))


amps, freqs = calc_trem_table(tremmed[10000:200000], freq, fs)

detune_trem_table = freqs / nontremmed_freq
gain_trem_table = amps / nontremmed_amp
shelf_trem_table = calculate_shelf_table(nontremmed, tremmed, freq, fs)

wavfile.write("detune-table.wav", fs, detune_trem_table)
wavfile.write("gain-table.wav", fs, gain_trem_table)
wavfile.write("shelf-table.wav", fs, shelf_trem_table)
