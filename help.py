import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
Fs = 10000  # Sampling frequency in Hz
T = 1 / Fs  # Sampling period
t = np.arange(0, 1, T)  # Time vector for 1 second
f_sine = np.pi  # Frequency of the sine wave in Hz

# Generate the sine wave
y = np.sin(2 * np.pi * f_sine * t)

# Perform the FFT
n = len(t)  # Length of the signal
Y = np.fft.fft(y)  # FFT of the signal
frequencies = np.fft.fftfreq(n, T)  # Frequency bins

# Get the magnitude of the FFT (real part)
Y_magnitude = np.abs(Y)

# Plot the FFT result
plt.plot(frequencies[:n // 2], Y_magnitude[:n // 2])  # Only positive frequencies
plt.title("FFT of Sine Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# Find the dominant frequency
dominant_frequency = frequencies[np.argmax(Y_magnitude[:n // 2])]
print(f"The dominant frequency is: {dominant_frequency} Hz")
