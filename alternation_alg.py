import numpy as np
import math
import matplotlib
import matplotlib.pyplot as mpl

def dft(l):
    return np.fft.fft(l)*math.sqrt(len(l))

def ifft(l):
    return np.fft.ifft(l)/math.sqrt(len(l))

def error(fft_vectors, magnitudes):
    x = np.sum((np.abs(fft_vectors) - magnitudes)**2)
    print(x)
    return x

def adjust_fft(fft_vectors, magitudes):
    return fft_vectors * magitudes / np.abs(fft_vectors)

def adjust_bin(sequence):
    return (np.abs(sequence) > 0.5).astype(int)

def alternation_algorithm(mags, tolerance = 10000):

    # initialize a random binary sequence
    bin_seq = (np.random.rand(len(mags)) > 0.5).astype(int)
    # convert the binary to fft vectors
    fft_vectors = dft(bin_seq)

    while error(fft_vectors, magnitudes) > tolerance: # while you are sad
    #for i in range(5000):
        print("~~~~")
        print(np.sum(bin_seq))
        # change the fft vectors to match the given magitudes while maintaining phase angles
        adjusted_fft = adjust_fft(fft_vectors, mags)
        # convert back to primal domain
        sequence = ifft(adjusted_fft)
        # adjust to binary sequence
        bin_seq = adjust_bin(sequence)
        # convert the binary to fft vectors
        fft_vectors = dft(bin_seq)

    return bin_seq

if __name__=='__main__':
    # source binary array in the primal domain
    # this is the "unknown" array which we are trying to find.
    primal = (np.random.rand(10000) > 0.5).astype(int)

    # input magnitude array
    magnitudes = abs(dft(primal))

    output = alternation_algorithm(magnitudes)

    print(np.sum(np.abs(primal-output)))