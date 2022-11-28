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
sample_rate, nontremmed = load_audio_mono("./060-C.wav")
nontremmed, source_min, source_max = norm_sig(nontremmed)
midi_note = 120
freq = (440 / 32) * (2 ** ((midi_note - 9) / 12))

nontremmed_deriv = np.gradient(nontremmed, 1)

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
    env_slice = envelope_deriv[i:(i + search_width)] * hann(search_width)
    env_slice_mean = np.mean(env_slice)
    median_dist = abs(env_slice_mean - median)

    # incentivize the loop end to be at the end of the sample
    beginning_penalty = 0.3 * std * (1 - (i - search_start) / search_span)**2

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


#############################
# PART TWO: find loop point #
#############################

# HUGE thanks to https://sourceforge.net/p/loopauditioneer/code/HEAD/tree/trunk/src/AutoLooping.cpp
# for determining the loop point
DERIVATIVE_THRESHOLD = 0.02
MIN_LOOP_LENGTH = 1.0  # seconds
DISTANCE_BETWEEN_LOOPS = 0.3  # seconds
QUALITY_FACTOR = 8  # value (8) /32767 (0.00008) for float)
FINAL_PASS_COUNT = 1000

max_derivative = np.std(nontremmed_deriv)
derivative_threshold = max_derivative * DERIVATIVE_THRESHOLD
passed = abs(nontremmed_deriv) < derivative_threshold

indicies_passed = np.array([i for i, x in enumerate(passed) if x])
in_range = np.logical_and(indicies_passed > attack_index, indicies_passed < release_index)
indicies_passed = indicies_passed[in_range]
# TODO: remove superfluous values in indexes_passed

slice_width = max((sample_rate / freq) * 2, 512) // 2 * 2
min_loop_length = MIN_LOOP_LENGTH * sample_rate
distance_between_loops = DISTANCE_BETWEEN_LOOPS * sample_rate

found_loops = []

for from_index in indicies_passed:
    for to_index in indicies_passed:
        if to_index < from_index + min_loop_length:
            continue

        if from_index + slice_width >= len(nontremmed) or to_index + slice_width >= len(nontremmed):
            continue

        if len(found_loops) > 0 and from_index - found_loops[-1][0][0] < distance_between_loops:
            continue

        # cross correlation (squared error)
        cross = (nontremmed[(from_index - 2):(from_index + 3)] - nontremmed[(to_index - 2):(to_index + 3)]) ** 2
        correlation_value = np.mean(cross)

        if correlation_value < QUALITY_FACTOR * QUALITY_FACTOR / 32767.0:
            found_loops.append(((from_index, to_index), correlation_value))

found_loops.sort(key=lambda x: x[1])

top_loops = found_loops[0:FINAL_PASS_COUNT]
top_pick = top_loops[0]

end_loop_sample = nontremmed[floor(top_pick[0][1] - slice_width):top_pick[0][1]]
start_loop_sample = nontremmed[top_pick[0][0]:(top_pick[0][0] + slice_width)]
plt.plot(np.concatenate((end_loop_sample, start_loop_sample)))
plt.show()

loop_start = top_pick[0][0]
loop_end = top_pick[0][1]

# out of the loops, which one lines up best spectral-wise?
# spectral_match = []

# for top_pick in top_loops:
#     end = nontremmed[(top_pick[0][1] - slice_width):top_pick[0][1]]
#     start = nontremmed[top_pick[0][0]:(top_pick[0][0] + slice_width)]

#     ref = resample_to(zoom_fft(end, [freq, min(freq * 16, sample_rate / 2 - 1)]), slice_width * 2)

#     together = np.concatenate((end, start))
#     res = zoom_fft(together, [freq, freq * 16])

#     distortion = np.sum((np.abs(res) / np.abs(ref)) ** 2)

#     spectral_match.append(((top_pick[0][0], top_pick[0][1]), distortion))

# spectral_match.sort(key=lambda x: x[1])

# loop_start = spectral_match[0][0][0]
# loop_end = spectral_match[0][0][1]


def cosine_fadeout(x):
    """ x: 0 - 1 """
    return 1 - x
    # return np.cos(x * math.pi / 2)


def cosine_fadein(x):
    return x
    # return np.cos((x * math.pi / 2) - math.pi / 2)


# loop test
crossfade_length = 256

loop = np.copy(nontremmed[loop_start:floor(loop_end + crossfade_length)])

crossfade = np.linspace(0, 1, crossfade_length)
fadeout = cosine_fadeout(crossfade)
fadein = cosine_fadein(crossfade)

crossed = (fadeout * loop[-crossfade_length:]) + (fadein * loop[0:crossfade_length])

loop_sans_crossed = loop[crossfade_length:-crossfade_length]

plt.plot(np.concatenate((loop_sans_crossed[-crossfade_length:], crossed, loop_sans_crossed[0:crossfade_length])))
plt.axvline(x=crossfade_length, color="black")
plt.axvline(x=crossfade_length * 2, color="black")
plt.show()

out = np.concatenate((loop_sans_crossed, crossed, loop_sans_crossed,
                      crossed, loop_sans_crossed, crossed, loop_sans_crossed))
# out = np.concatenate((loop, loop, loop))
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
