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
# set slice size (align for FFT)
# slice_width = 2 ** ceil(log2(max((sample_rate / freq), 512)))

# # look for a spot of equal amplitude as attack
# attack_amp = envelope_db[attack_index]

# loop_end_search_start = floor(len(nontremmed) * 0.6)
# loop_end_area = np.argmin(abs(envelope_db[loop_end_search_start:release_index] - attack_amp)) + loop_end_search_start

# ref_sample = nontremmed[attack_index:(attack_index + slice_width)]
# ref_fft = zoom_fft(ref_sample, [freq / 2, freq * 16.5], m=512)
# ref_phases = np.angle(ref_fft)

# loop_end_sample = nontremmed[loop_end_area:(loop_end_area + slice_width)]
# loop_end_fft = zoom_fft(loop_end_sample, [freq / 2, freq * 16], m=512)
# loop_end_phases = np.angle(loop_end_fft)

# phase_diff = (loop_end_phases[16] - ref_phases[16]) * freq / math.pi
# loop_end = floor(loop_end_area + phase_diff)

# plt.plot(np.concatenate((nontremmed[(loop_end - slice_width):loop_end],
#                          nontremmed[attack_index:(attack_index + slice_width)])))
# plt.show()

# PART TWO: find loop point
# I use a pretty unique solution here. I was inspired when I was finding loops manually.
# The way I determined whether a loop was good or not was whether it made a "click" sound
# when it looped. But, what is a click sound but harmonic distortion?

# My algorithm tries every possible loop, and calculates the harmonic distortion
# for that potential loop. It does this by concatinating the end of the loop with the
# beginning of the loop. I take the FFT of this, normalize it (based on loop start FFT),
# and calculate the distortion. I also make the lower frequencies more important using
# the table `harmonic_bias`. This table punishes lower distortion more than higher distortion,
# as the lower distortion is much more audible.

# slice off the beginning and end of the sample. We don't want to deal with
# the randomness of attack and release parts, because next we're looking for a
# loop


def calc_harmonics(x):
    return abs(rfft(x))


loop_search_slice = nontremmed[floor(attack_index):floor(release_index)]

# slice_width is how wide of a slice to take from the beginning and end of the loop
# align it to a power of 2 for FFT
slice_width = 2 ** ceil(log2(max((sample_rate / freq), 512)))

# ref_sample is the reference sample (for normalizing `test_loop`)
ref_sample = loop_search_slice[0:slice_width]
ref_sample_amps = resample_to(calc_harmonics(ref_sample), slice_width + 1)

# this is a curve that biases the lower frequencies, making them more punishing
harmonic_bias = (1 - (np.linspace(0.0, 1.0, slice_width + 1) ** 3)) * 2

lowest_score = math.inf
lowest_index = -1

for i in np.arange(len(loop_search_slice) * 0.6, len(loop_search_slice) - slice_width, 2):
    pos = floor(i)

    # calculate score based on what provides the least harmonic distortion
    potential_end = loop_search_slice[(pos - slice_width):pos]

    # loop to test distortion
    test_loop = np.concatenate((potential_end, ref_sample))
    test_loop_amps = calc_harmonics(test_loop)

    # normalize the amplitudes by the reference sample
    test_loop_amps_norm = test_loop_amps / ref_sample_amps

    # bias the lower frequencies, they are more audible
    test_loop_amps_biased = np.maximum(test_loop_amps_norm, ref_sample_amps) * harmonic_bias

    # calculate distortion (sqrt isn't necessary)
    score = np.sum(np.abs(test_loop_amps_biased)**2)

    if score < lowest_score:
        lowest_score = score
        lowest_index = pos

loop_end = attack_index + lowest_index

plt.plot(np.concatenate((nontremmed[(loop_end - slice_width):loop_end],
                         nontremmed[attack_index:(attack_index + slice_width)])))
plt.show()

# loop test
loop = nontremmed[attack_index:loop_end]
out = np.concatenate((loop, loop, loop, loop, loop))
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
