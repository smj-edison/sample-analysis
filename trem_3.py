import math
from math import floor, pi, e

import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up

from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import zoom_fft, iirfilter, butter, filtfilt
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
    window_rms,
)

two_pi = pi * 2.0

fs, tremmed = load_audio_mono("./test-samples/072-C.wav")
fs, nontremmed = load_audio_mono("./test-samples/072-C-nt.wav")
freq = 523


# find tremulant period
def find_period(sig, fmin, fmax, fs):
    sig, _, _ = norm_sig(sig)

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


def calc_freqs(sig, freq, fs, avg_size=200):
    period_width = int(fs / freq)
    half_period_width = int(period_width / 2)

    kernel = np.exp(np.linspace(0, two_pi, period_width) * 1j)
    complex_rep = np.convolve(sig, kernel, mode="valid")
    phases = np.angle(complex_rep)

    advance_by = np.gradient(np.unwrap(phases))
    # sliding average to clean up
    advance_by = np.convolve(advance_by, np.ones(avg_size) / avg_size, mode="valid")

    freqs = advance_by * fs / two_pi

    padding = np.zeros(half_period_width)
    return np.concatenate((padding, freqs[half_period_width:]))


def calc_amps(sig, window_size=1000):
    half = int(window_size / 2)
    padding = np.zeros(half)

    return np.concatenate((padding, window_rms(sig, window_size)[half:]))


def calc_all(nontremmed, tremmed, fs, note_freq, table_size=512):
    third_harmonic_filt = butter(
        1, (note_freq * 2.8, note_freq * 3.2), fs=fs, btype="bandpass", analog=False
    )

    base_freq = np.median(calc_freqs(nontremmed, note_freq, fs))
    base_amp = np.median(calc_amps(nontremmed))
    base_amp_third = np.median(
        calc_amps(filtfilt(third_harmonic_filt[0], third_harmonic_filt[1], nontremmed))
    )

    freqs = calc_freqs(tremmed, note_freq, fs)
    amps = calc_amps(tremmed)
    amps_third = calc_amps(
        filtfilt(third_harmonic_filt[0], third_harmonic_filt[1], tremmed)
    )

    freqs = butter_lowpass_filter(freqs, 50, fs, order=1)[20000:-20000]
    amps = butter_lowpass_filter(amps, 50, fs, order=1)[20000:-20000]
    amps_third = butter_lowpass_filter(amps_third, 50, fs, order=1)[20000:-20000]

    # normalize all
    freqs_norm = freqs / base_freq
    amps_norm = amps / base_amp
    amps_third_norm = amps_third / base_amp_third

    amp_table, trem_speed = calc_trem_shape(amps_norm, fs)
    freq_table, _ = calc_trem_shape(freqs_norm, fs, trem_speed - 0.2, trem_speed + 0.2)
    amp_third_table, _ = calc_trem_shape(
        amps_third_norm, fs, trem_speed - 0.2, trem_speed + 0.2
    )

    amp_sized = resample_to(amp_table, table_size)
    freq_sized = resample_to(freq_table, table_size)
    amp_third_sized = resample_to(amp_third_table, table_size) / amp_sized

    amp_db = gain_to_db(amp_sized)
    freq_cents = detune_to_cents(freq_sized)
    amp_third_db = gain_to_db(amp_third_sized)

    return amp_db, freq_cents, amp_third_db, trem_speed


amp_db, freq_cents, amp_third_db, speed = calc_all(nontremmed, tremmed, fs, freq)

wavfile.write("amp-db.wav", fs, amp_db)
wavfile.write("detune-cents.wav", fs, freq_cents)
wavfile.write("amp-third-db.wav", fs, amp_third_db)

plt.scatter(amp_db, freq_cents, c=amp_third_db)
plt.title("Pipe gain vs frequency")
plt.xlabel("Gain (dB)")
plt.ylabel("Detune from nontremmed (cents)")
plt.colorbar()
plt.show()
