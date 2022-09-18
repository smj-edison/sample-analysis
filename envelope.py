import math
from math import floor
import numpy as np
import matplotlib.pyplot as plt

from analysis import calc_amp
from scipy.signal import zoom_fft
from scipy.fft import rfft, fftfreq
from scipy.io import wavfile
from helpers import butter_lowpass_filter, load_audio_mono, calc_rms, norm_sig, resample_to


sample_rate, nontremmed = load_audio_mono("./test-samples/076-E.wav")
nontremmed, source_min, source_max = norm_sig(nontremmed)
freq = 659

# use the hilbert transform to get the envelope
envelope = calc_amp(nontremmed)
envelope_norm = (norm_sig(envelope)[0] + 1.1) / 2

# smooth the envelope signal down. The hilbert transform seems to have residue
# audio within the signal, but a simple lowpass filter clears that up
envelope_smoothed = np.log10(butter_lowpass_filter(envelope_norm, 10, sample_rate)[2000:])

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
peak_release = np.argmin(envelope_deriv[peak_attack:]) + peak_attack

# find start of attack
attack_index = -1
release_index = -1


# search for spots to stay within for picking a loop
search_width = 40000
search_step = 100
strictness_start = 0.3
strictness_relax_factor = 1.2
too_far_in_percentage_attack = 0.15
too_far_in_percentage_release = 0.2

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
    if attack_index > len(nontremmed) * too_far_in_percentage_attack:
        attack_index = -1

    strictness *= strictness_relax_factor

    if strictness > 50:
        raise Exception("signal is too noisy")

strictness = strictness_start
while release_index == -1:
    for i in range(peak_release, max(peak_attack, search_width), -search_step):
        rms = calc_rms(envelope_deriv[(i - search_width):i])

        if rms < (std * strictness) + mean:
            release_index = i
            break

    # if the loop point seems too far in the sample, keep iterating
    if release_index < len(nontremmed) * (1 - too_far_in_percentage_release):
        release_index = -1

    strictness += strictness_relax_factor

    if strictness > 50:
        raise Exception("signal is too noisy")

attack_index = floor(attack_index)

# To make sure phases line up, we'll take the fft surrounding the inputted frequency
# this way we can zero in on the exact frequency that we should be checking for loop
# points along
potential_loop = nontremmed[floor(attack_index):floor(release_index)]

# Now we have where to look along for loop points, we'll proceed to do so
increment_by = (sample_rate / freq) * 8
increment_by_int = floor(increment_by)

highest_score = -math.inf
highest_index = -1

ref_sample = potential_loop[0:increment_by_int]
ref_sample_amps = resample_to(abs(rfft(ref_sample)[1:]), increment_by)

last_plot = -math.inf

for i in np.arange(len(potential_loop) * 0.6, len(potential_loop) - increment_by_int, 1):
    pos = floor(i)

    # calculate score based on what provides the least harmonic distortion
    potential_end = potential_loop[(pos - increment_by_int):pos]
    analysis = abs(rfft(np.concatenate((potential_end, ref_sample)))[1:]) / ref_sample_amps
    bias = (1 - (np.linspace(0.0, 1.0, len(analysis)) ** 2)) * 2
    
    score = -math.sqrt(np.sum(np.abs(analysis / ref_sample_amps * bias)**2))

    if score > highest_score:
        if score - last_plot > 20:
            plt.plot(np.concatenate((potential_end, ref_sample)))
            plt.show()

            last_plot = score

        highest_score = score
        highest_index = pos

        print(f"harmonic distortion: {-score}")

plt.plot(np.concatenate((potential_loop[(highest_index - increment_by_int):highest_index], ref_sample)))
plt.show()

loop_end = attack_index + highest_index


plt.plot(np.concatenate((nontremmed[(loop_end - 512):loop_end], nontremmed[attack_index:(attack_index + 512)])))
plt.show()

# loop test
loop = nontremmed[attack_index:loop_end]
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
