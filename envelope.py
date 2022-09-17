import math
from math import floor
import numpy as np
import matplotlib.pyplot as plt

from analysis import calc_amp
from scipy.signal import zoom_fft
from scipy.io import wavfile
from helpers import butter_lowpass_filter, load_audio_mono, calc_rms, norm_sig


sample_rate, nontremmed = load_audio_mono("./069-A-nt.wav")
nontremmed, source_min, source_max = norm_sig(nontremmed)
freq = 440

# use the hilbert transform to get the envelope
envelope = calc_amp(nontremmed)
envelope_norm, _, _ = norm_sig(envelope)

# smooth the envelope signal down. The hilbert transform seems to have residue
# audio within the signal, but a simple lowpass filter clears that up
envelope_smoothed = butter_lowpass_filter(envelope, 10, sample_rate)[2000:]

# take the derivative of the envelope. A decrease or increase in amplitude
# will corrispond with a large value in the derivative, where a roughly uniform
# amplitude will barely register in the derivative.
# This is useful, as it lets us figure out where the attack and release are
# (which both corrispond with a strong change in volume)
envelope_deriv = np.gradient(envelope_smoothed, 1)

# calculate mean and standard deviation of the envelope. I use this to check whether
# a segment of signal is within normal variation (take RMS of part of signal, then
# see whether it's below standard deviation, or some multiple of)
mean = np.mean(envelope_deriv)
std = np.std(envelope_deriv)

# find the peaks
peak_attack = np.argmax(envelope_deriv)
peak_release = np.argmin(envelope_deriv)

# find start of attack
attack_index = -1
release_index = -1


# search for spots to stay within for picking a loop
search_width = 40000
search_step = 100
strictness_start = 0.5
strictness_relax_factor = 0.1
too_far_in_percentage = 0.15

# it starts very strict, and will keep trying and relaxing until it gets a good hit
strictness = strictness_start
while attack_index == -1:
    # search through the envelope to find a part that is within the standard deviation.
    # pretty much find a part of the signal that is not doing anything crazy, which
    # is what we want for a loop point.
    for i in range(peak_attack, peak_release, search_step):
        rms = calc_rms(envelope_deriv[i:(i + search_width)])

        if rms < (std * strictness) + mean:
            attack_index = i + search_width / 2
            break

    # if the loop point seems too far in the sample, keep iterating
    if attack_index > len(nontremmed) * too_far_in_percentage:
        attack_index = -1

    strictness += strictness_relax_factor

strictness = strictness_start
while release_index == -1:
    for i in range(peak_release, max(peak_attack, search_width), -search_step):
        rms = calc_rms(envelope_deriv[(i - search_width):i])

        if rms < (std * strictness) + mean:
            release_index = i - search_width / 2
            break

    # if the loop point seems too far in the sample, keep iterating
    if release_index < len(nontremmed) * (1 - too_far_in_percentage):
        release_index = -1

    strictness += strictness_relax_factor

attack_index = floor(attack_index)

# To make sure phases line up, we'll take the fft surrounding the inputted frequency
# this way we can zero in on the exact frequency that we should be checking for loop
# points along
potential_loop = nontremmed[floor(attack_index):floor(release_index)]

spread = 100
min_freq = freq * (2 ** (-spread / 1200))
max_freq = freq * (2 ** (spread / 1200))

analysis = zoom_fft(potential_loop, [min_freq, max_freq], fs=sample_rate)
analysis_freqs = np.linspace(min_freq, max_freq, len(potential_loop))
detected_freq = analysis_freqs[np.argmax(abs(analysis))]

# Now we have where to look along for loop points, we'll proceed to do so
increment_by = (sample_rate / detected_freq) * 8
increment_by_int = floor(increment_by)

highest_score = -math.inf
highest_index = -1

ref_sample = potential_loop[0:increment_by_int]

for i in np.arange(len(potential_loop) * 0.6, len(potential_loop) - increment_by_int, 1):
    pos = floor(i)

    score = 0

    cross_corr = (ref_sample.dot(potential_loop[pos:(pos + increment_by_int)]) / increment_by) * 10
    score += cross_corr

    # encourage first samples to line up more than any of the others
    point_diff = 0.5 - abs(ref_sample[0] - potential_loop[pos])
    similarity_bias = ((point_diff * 50) ** 3 / 48000)
    score += similarity_bias

    amp_closeness = np.dot(
        envelope[attack_index:(attack_index + increment_by_int)],
        envelope[(attack_index + pos):(attack_index + pos + increment_by_int)]
    ) / increment_by * 5
    score += amp_closeness

    if score > highest_score:
        highest_score = score
        highest_index = pos

        print(f"cross correlation: {cross_corr}")
        print(f"similarity: {similarity_bias} (diff {point_diff})")
        print(f"amplitude closeness: {amp_closeness}")
        print("---")

plt.plot(potential_loop[0:increment_by_int])
plt.plot(potential_loop[highest_index:(highest_index + increment_by_int)])
plt.show()


loop_end = highest_index + floor(attack_index)

# loop test
loop = nontremmed[floor(attack_index):loop_end]
out = np.concatenate((loop, loop, loop, loop, loop))
wavfile.write("loop.wav", sample_rate, out)

plt.plot(envelope_smoothed)
plt.axvline(x=peak_attack, color="blue")
plt.axvline(x=attack_index, color="green")
plt.axvline(x=loop_end, color="cyan")
plt.axvline(x=release_index, color="red")
plt.axvline(x=peak_release, color="purple")
# plt.axhline(y=mean, color="black")
# plt.axhline(y=mean + std*2, color="black")
# plt.axhline(y=mean - std*2, color="black")
plt.show()
