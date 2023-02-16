import numpy
import scipy
import matplotlib.pyplot as plt
from scipy import signal, fft

a = 0
b = 10
n = 500
Fs = 1000
F_max = 5
width = 21/2.54
height = 14/2.54

random = numpy.random.normal(a, b, n)
x = numpy.arange(n) / Fs
w = F_max / (Fs / 2)
parameter = scipy.signal.butter(3, w, 'low', output='sos')
y = scipy.signal.sosfiltfilt(parameter, random)

fig, ax = plt.subplots(figsize=(width, height))
ax.plot(x, y, linewidth=1)
ax.set_xlabel('Час', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Cигнал з максимальною частотою F_max = 15Гц', fontsize=14)
fig.savefig("./figures/Сигнал з максимальною частотою 5Гц.png")

spectrum = fft.fft(y)
spectrum_abs = numpy.abs(fft.fftshift(spectrum))
frequency_counts = fft.fftfreq(500, 1/500)
frequency_counts_symetry = fft.fftshift(frequency_counts)

fig, ax = plt.subplots(figsize=(width, height))
ax.plot(frequency_counts_symetry, spectrum_abs, linewidth="1")
ax.set_xlabel("Частота", fontsize=14)
ax.set_ylabel("Амплітуда спектру", fontsize=14)
fig.savefig("./figures/Спектр сигналу з максимальною частотою 5Гц.png")