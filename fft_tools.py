# fft_tools.py
import numpy as np

def windowed_centered(samples):
    x = samples.astype(float)
    x -= np.mean(x)
    win = np.hanning(len(x))
    return x * win

def compute_fft_spectrum(samples, fs):
    x_win = windowed_centered(samples)
    N = len(x_win)
    fft_vals = np.fft.rfft(x_win)
    fft_mag = np.abs(fft_vals) * 2.0 / N
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    return freqs, fft_mag, fft_vals

def fft_mag_at_freq(samples, fs, target_freq):
    freqs, fft_mag, _ = compute_fft_spectrum(samples, fs)
    idx = np.argmin(np.abs(freqs - target_freq))
    return fft_mag[idx], freqs[idx]
