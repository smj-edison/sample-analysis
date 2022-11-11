import math
from math import floor
import numpy as np
import matplotlib.pyplot as plt

from analysis import calc_amp
from scipy.signal.windows import hann
from scipy.fft import rfft
from scipy.io import wavfile
from helpers import load_audio_mono, calc_rms, norm_sig, resample_to


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

# calculate mean and standard deviation of the envelope. I use this to check whether
# a segment of signal is within normal variation (take RMS of part of signal, then
# see whether it's below standard deviation, or some multiple of)
mean = np.mean(envelope_deriv)
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
search_width = 5000
search_step = 100

# used to tune what region attack and release points can be in
# attack region is from `0` to `len(nontremmed) * too_far_in_percentage_attack``
# release region is from `len(nontremmed) * too_far_in_percentage_release` to peak_release
too_far_in_percentage_attack = 0.15
too_far_in_percentage_release = 0.8

search_start = peak_attack
search_end = floor(min(peak_release - search_width, len(nontremmed) * too_far_in_percentage_attack))
search_span = search_end - search_start

lowest_score = math.inf

for i in range(search_start, search_end, search_step):
    env_slice = envelope_db[i:(i + search_width)]
    rms = calc_rms(env_slice)
    rms_dist = abs(rms - mean)

    # incentivize the loop start to be at the beginning of the sample
    end_bias = std * (1 - (i - search_start) / search_span)

    # lower score = better
    score = rms + end_bias

    if score < lowest_score:
        lowest_score = score
        attack_index = i + search_width / 2

lowest_score = math.inf

search_start = floor(min(peak_release - search_width, len(nontremmed) * too_far_in_percentage_release))
search_end = peak_release - search_width
search_span = search_end - search_start

for i in range(search_start, search_end, search_step):
    env_slice = envelope_db[i:(i + search_width)]
    rms = calc_rms(env_slice)
    rms_dist = abs(rms - mean)

    # incentivize the loop end to be at the end of the sample
    end_bias = 0  # std * (i - search_start) / search_span

    # lower score = better
    score = rms + end_bias

    if score < lowest_score:
        lowest_score = score
        release_index = i + search_width / 2

attack_index = floor(attack_index)
release_index = floor(release_index)

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
loop_search_slice = nontremmed[floor(attack_index):floor(release_index)]

# slice_width is how wide of a slice to take from the beginning and end of the loop
slice_width = max((sample_rate / freq) * 8, 512)
slice_width_int = floor(slice_width)

# ref_sample is the reference sample (for normalizing `test_loop`)
ref_sample = loop_search_slice[0:slice_width_int]
ref_sample_amps = resample_to(abs(rfft(ref_sample, norm="ortho")[1:]), slice_width_int)

# this is a curve that biases the lower frequencies, making them more punishing
harmonic_bias = (1 - (np.linspace(0.0, 1.0, slice_width_int) ** 2)) * 2

highest_score = -math.inf
highest_index = -1

for i in np.arange(len(loop_search_slice) * 0.6, len(loop_search_slice) - slice_width_int, 2):
    pos = floor(i)

    # calculate score based on what provides the least harmonic distortion
    potential_end = loop_search_slice[(pos - slice_width_int):pos]

    # loop to test distortion
    test_loop = np.concatenate((potential_end, ref_sample))
    test_loop_amps = abs(rfft(test_loop, norm="ortho")[1:])

    # normalize the amplitudes by the reference sample
    test_loop_amps_norm = test_loop_amps / ref_sample_amps

    # bias the lower frequencies, they are more audible
    test_loop_amps_biased = test_loop_amps_norm * harmonic_bias

    # calculate distortion
    #score = -math.sqrt(np.sum(np.abs(test_loop_amps_biased)**2))
    score = -np.sum(np.abs(test_loop_amps_biased)**2)

    if score > highest_score:
        highest_score = score
        highest_index = pos

loop_end = attack_index + highest_index

plt.plot(np.concatenate((nontremmed[(loop_end - slice_width_int):loop_end],
                         nontremmed[attack_index:(attack_index + slice_width_int)])))
plt.show()

# loop test
loop = nontremmed[attack_index:loop_end]
out = np.concatenate((loop, loop, loop, loop, loop))
wavfile.write("loop.wav", sample_rate, out)

plt.plot(envelope_deriv)
plt.axvline(x=peak_attack, color="blue")
plt.axvline(x=attack_index, color="green")
plt.axvline(x=loop_end, color="cyan")
plt.axvline(x=release_index, color="red")
plt.axvline(x=peak_release, color="purple")
plt.axhline(y=mean, color="black")
plt.axhline(y=mean + std*2, color="black")
plt.axhline(y=mean - std*2, color="black")
plt.show()
