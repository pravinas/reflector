import numpy as np

bits = np.random.choice([0,1],size=10)
fft = np.fft.fft(bits)
fft_mags = np.abs(fft)
fft_phases = 