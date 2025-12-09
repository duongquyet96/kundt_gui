import numpy as np

# Funktion: Zentrieren des Signals + Hanning-Fenster
def windowed_centered(samples):
    x = samples.astype(float)
    x -= np.mean(x)                 # DC-Offset entfernen
    win = np.hanning(len(x))        # Hanning-Fenster
    return x * win                  # Fenster auf jedes Sample anwenden → reduziert Spektralverzerrung

# Funktion: FFT berechnen und Frequenz- sowie Magnitudenspektrum zurückgeben
def compute_fft_spectrum(samples, fs):
    x_win = windowed_centered(samples)     # Samples vorbereiten (zentrieren + fenstern)
    N = len(x_win)                         # Anzahl der FFT-Punkte = Länge des Signals
    fft_vals = np.fft.rfft(x_win)          # Reelle FFT berechnen (nur positive Frequenzen)
    fft_mag = np.abs(fft_vals) * 2.0 / N   # Betrag der komplexen FFT + Normierung auf Amplitude
    freqs = np.fft.rfftfreq(N, 1.0 / fs)   # Frequenzachse für rFFT berechnen (0 bis Nyquist)
    return freqs, fft_mag, fft_vals        # Frequenzen, Magnituden und komplexe FFT zurückgeben

# Funktion: Magnitude der FFT an einer bestimmten Ziel-Frequenz
def fft_mag_at_freq(samples, fs, target_freq):
    freqs, fft_mag, _ = compute_fft_spectrum(samples, fs)  # FFT-Spektrum berechnen
    idx = np.argmin(np.abs(freqs - target_freq))           # Index des Frequenz-Bins finden, der der Ziel-Frequenz am nächsten liegt
    return fft_mag[idx], freqs[idx]                        # Magnitude und tatsächliche Bin-Frequenz zurückgeben
