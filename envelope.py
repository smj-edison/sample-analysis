import math
from math import floor, log2, ceil
import numpy as np
import matplotlib.pyplot as plt

from analysis import calc_amp
from scipy.signal import zoom_fft
from scipy.signal.windows import hann
from scipy.fft import rfft
from scipy.io import wavfile
from helpers import load_audio_mono, norm_sig, resample_to

# /home/mason/rust/mjuo/vpo-backend/060-C.wav
sample_rate, nontremmed = load_audio_mono("./test-samples/069-A-nt.wav")
nontremmed, source_min, source_max = norm_sig(nontremmed)
freq = 440

# calculate the envelope and convert to logarithmic scale
envelope = calc_amp(nontremmed, dmin=(sample_rate // 300), dmax=(sample_rate // 300)) + 0.2
envelope_db = 20 * np.log10(envelope)

# take the derivative of the envelope. A decrease or increase in amplitude
# will corrispond with a large value in the derivative, where a roughly uniform
# amplitude will barely register in the derivative.
# This is useful, as it lets us figure out where the attack and release are
# (which both corrispond with a strong change in volume)
envelope_deriv = np.gradient(envelope_db, 1)

# calculate median and standard deviation of the envelope derivative
median = np.median(envelope_deriv)
std = np.std(envelope_deriv)

# PART ONE: Find attack and release locations

# 1. Find the peaks (these are the maximum bounds to start searching for our attack and release).
# The peaks (max/min) in the derivative are the biggest swings in amplitude, so the loop most
# definitely needs to be inside.
peak_attack = np.argmax(envelope_deriv[0:(len(envelope_deriv) // 2)])
peak_release = np.argmin(envelope_deriv[peak_attack:]) + peak_attack

# find start of attack
attack_index = -1

# 2. Search through the envelope to find the amplitude that's closest to the mean
# also, incentivize it to use a loop point closer to the beginning or ending
search_width = 20000
search_step = 100

# used to tune what region attack and release points can be in
# attack region is from `0` to `len(nontremmed) * too_far_in_percentage_attack``
# release region is from `len(nontremmed) * too_far_in_percentage_release` to peak_release
too_far_in_percentage_attack = 0.15
too_far_in_percentage_release = 0.7

search_start = peak_attack
search_end = floor(min(peak_release - search_width, len(nontremmed) * too_far_in_percentage_attack))
search_span = search_end - search_start

lowest_score = math.inf

# we're looking for when the signal amplitude stabilizes, by looking where the derivative is within normal
# fluctuation
for i in range(search_start, search_end, search_step):
    env_slice = envelope_deriv[i:(i + search_width)] * hann(search_width)
    env_slice_mean = np.mean(env_slice)
    median_dist = abs(env_slice_mean - median)

    # incentivize the loop start to be at the beginning of the sample
    end_penalty = 0  # 0.1 * std * ((i - search_start) / search_span)**2

    # lower score = better
    score = median_dist + end_penalty

    if score < lowest_score:
        lowest_score = score
        attack_index = i + search_width / 2

# same approach for finding release
lowest_score = math.inf

search_start = floor(min(peak_release - search_width, len(nontremmed) * too_far_in_percentage_release))
search_end = peak_release - search_width
search_span = search_end - search_start

for i in range(search_start, search_end, search_step):
    env_slice = envelope_db[i:(i + search_width)] * hann(search_width)
    env_slice_mean = np.mean(env_slice)
    median_dist = abs(env_slice_mean - median)

    # incentivize the loop end to be at the end of the sample
    beginning_penalty = 0.3 * std * (1 - (i - search_start)**2 / search_span)

    # lower score = better
    score = median_dist + beginning_penalty

    if score < lowest_score:
        lowest_score = score
        release_index = i + search_width / 2

attack_index = floor(attack_index)
release_index = floor(release_index)

# search in the area around attack index for where it hits zero
search_area = nontremmed[attack_index:(attack_index + 1000)]
attack_index += np.argmin(abs(search_area))

# PART TWO: find loop point

# VV alternate approach (WIP) VV
slice_width = max((sample_rate / freq), 512) // 1

# look for a spot of equal amplitude as attack
attack_amp = envelope_db[attack_index]

loop_end_search_start = floor(len(nontremmed) * 0.6)
loop_end_area = np.argmin(abs(envelope_db[loop_end_search_start:release_index] - attack_amp)) + loop_end_search_start

res = np.convolve(nontremmed[(loop_end_area - slice_width * 2):(loop_end_area + slice_width * 2)],
                  nontremmed[(attack_index - slice_width):(attack_index + slice_width)], mode="valid")
loop_end = np.argmax(res) + loop_end_area - slice_width * 2

plt.plot(np.concatenate((nontremmed[(loop_end - slice_width):loop_end],
                         nontremmed[attack_index:(attack_index + slice_width)])))
plt.show()

# 274641


def cosine_fadeout(x):
    """ x: 0 - 1 """
    return np.cos(x * math.pi / 2)


def cosine_fadein(x):
    return np.cos((x * math.pi / 2) - math.pi / 2)


# loop test
loop = np.copy(nontremmed[attack_index:loop_end])

crossfade_length = 512
crossfade = np.linspace(0, 1, crossfade_length)
fadeout = cosine_fadeout(crossfade)
fadein = cosine_fadein(crossfade)

crossed = (fadeout * loop[-crossfade_length:]) + (fadein * loop[0:crossfade_length])

ang = np.angle(zoom_fft(res, [440, 440], fs=48000, m=1))[0]
ang += (440*(1/48000)*crossfade_length)
after_ang = np.angle(zoom_fft(loop[0:crossfade_length], [440, 440], fs=48000, m=1))[0]
delta_ang = after_ang - ang+math.pi
delta_ang_mapped = delta_ang/440*48000

loop_sans_crossed = loop[crossfade_length:-crossfade_length]

plt.plot(np.concatenate((loop_sans_crossed[-crossfade_length:], crossed, loop_sans_crossed[0:crossfade_length])))
plt.axvline(x=crossfade_length, color="black")
plt.axvline(x=crossfade_length * 2, color="black")
plt.show()

out = np.concatenate((loop_sans_crossed, crossed, loop_sans_crossed,
                      crossed, loop_sans_crossed, crossed, loop_sans_crossed))
wavfile.write("loop.wav", sample_rate, out)

plt.plot(envelope_db)
plt.axvline(x=peak_attack, color="blue")
plt.axvline(x=attack_index, color="green")
plt.axvline(x=loop_end, color="cyan")
plt.axvline(x=release_index, color="red")
plt.axvline(x=peak_release, color="purple")
plt.axhline(y=median, color="black")
plt.axhline(y=median + std, color="black")
plt.axhline(y=median - std, color="black")
plt.show()
