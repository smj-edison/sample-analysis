import numpy as np
from math import floor, ceil, factorial

from scipy.signal import butter, lfilter, firwin
from scipy.interpolate import interp1d
from scipy.io import wavfile


def lerp(a, b, t):
    return (1 - t) * a + t * b


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def fir_lowpass(sig, cutoff, fs, taps=10):
    filter = firwin(taps, cutoff / fs)

    return np.convolve(sig, filter)


def apply_filter(b, a, x, xm1, xm2):
    return b[0] * x + b[1] * xm1 + b[2] * xm2 - a[1] * xm1 + a[2] * xm2


def load_audio_mono(filename):
    sample_rate, data = wavfile.read(filename)

    if len(data.shape) > 1:
        audio_raw = np.array(list(zip(*data))[0]) / 32768
    else:
        audio_raw = np.array(data) / 32768

    return (sample_rate, audio_raw)


def norm_sig(signal):
    signal_min, signal_max = np.min(signal), np.max(signal)

    signal_norm = (((signal - signal_min) / (signal_max - signal_min)) * 2) - 1

    return (signal_norm, signal_min, signal_max)


def denorm_sig(signal, signal_min, signal_max):
    signal_denorm = (((signal + 1) / 2) * (signal_max - signal_min)) + signal_min

    return signal_denorm


def calc_rms(signal):
    return np.sqrt(np.mean(signal**2))


def gain_to_db(gain):
    return 20 * np.log10(gain)


def detune_to_cents(detune):
    return np.log2(detune) * 1200


def resample_to(signal, output_length, kind="linear"):
    return interp1d(np.linspace(0, 1, len(signal)), signal, kind=kind)(
        np.linspace(0, 1, output_length)
    )


def table_lookup(table, table_freq, pos):
    pos_in_table = pos * len(table) * table_freq

    return lerp(
        table[floor(pos_in_table) % len(table)],
        table[ceil(pos_in_table) % len(table)],
        pos_in_table % 1,
    )


# https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global min of dmin-chunks of locals min
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    return lmin, lmax


# https://stackoverflow.com/questions/1125666/how-do-you-do-bicubic-or-other-non-linear-interpolation-of-re-sampled-audio-da


def hermite_interp(x0, x1, x2, x3, t):
    c0 = x1
    c1 = 0.5 * (x2 - x0)
    c2 = x0 - (2.5 * x1) + (2 * x2) - (0.5 * x3)
    c3 = (0.5 * (x3 - x0)) + (1.5 * (x1 - x2))

    return (((((c3 * t) + c2) * t) + c1) * t) + c0
