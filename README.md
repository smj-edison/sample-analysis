# Sample analysis

This is a repository with various attempts at working with organ samples. So far the scripts I have are:
1. trem.py - Tremulant modelling, based on amplitude and fundamental frequency shifts
2. envelope.py - Calculate attack, release, and loop positions in a sample. The technique I use for looping is concatinating two pieces of audio, and see how much spectral distortion is introduced. I have an optimization loop that looks for what concatination causes the least audible distortion (which at the end of the day is what matters). It is decently slow, but quite accurate (~5-20 seconds per sample, depending on the frequency)
