import numpy as np
import matplotlib.pyplot as plt

def plot_fft(samples, fs):
    x = samples.astype(float)
    x -= np.mean(x)
    x *= np.hanning(len(x))

    N = len(x)
    fft_vals = np.fft.rfft(x)
    fft_mag = np.abs(fft_vals) * 2 / N
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    plt.figure(figsize=(8,4))
    plt.plot(freqs, fft_mag)
    plt.title("FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

    idx = np.argmax(fft_mag[1:]) + 1
    print(f"Dominant freq: {freqs[idx]:.2f} Hz (amp {fft_mag[idx]:.2f})")

def fft_bin_at_freq(samples, fs, target_freq):
    x = samples.astype(float)
    x -= np.mean(x)
    x *= np.hanning(len(x))

    N = len(x)
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    idx = np.argmin(np.abs(freqs - target_freq))
    return fft_vals[idx], freqs[idx]
