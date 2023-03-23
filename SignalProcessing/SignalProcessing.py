import numpy
import scipy
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt

n = 500
fs = 1000
f_max = 5
f_filter = 12

random = numpy.random.normal(0, 10, n)
time = numpy.arange(n) / fs
w = f_max / (fs / 2)

filter = scipy.signal.butter(3, w, "low", output="sos")
filter_signal = scipy.signal.sosfiltfilt(filter, random)

x1 = time
y1 = filter_signal
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(x1, y1, linewidth=1)
ax.set_xlabel("Час(секунди)", fontsize=14)
ax.set_ylabel("Амплітуда сигналу", fontsize=14)
plt.title("Cигнал з максимальною частотою F_max = 5Гц", fontsize=14)
title = "График 1"
fig.savefig("./figures/" + title + ".png", dpi=600)

spectrum = fft.fft(filter_signal)
spectrum_abs = numpy.abs(fft.fftshift(spectrum))
frequency_counts = fft.fftfreq(n, 1 / n)
frequency_counts_shifted = fft.fftshift(frequency_counts)

x2 = frequency_counts_shifted
y2 = spectrum_abs
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(x2, y2, linewidth=1)
ax.set_xlabel("Частота (Гц)", fontsize=14)
ax.set_ylabel("Амплітуда сигналу", fontsize=14)
plt.title("Спектр сигналу з максимальною частотою 5Гц")
title2 = "График 2"
fig.savefig("./figures/" + title2 + ".png", dpi=600)

discrete_signals = []
discrete_spectrums = []
w2 = f_filter/(fs/2)
filter_discrete = scipy.signal.butter(3, w2, "low", output="sos")
discrete_signals_filtred = []

dispersion_s = []
snr_s = []
for d_t in [2, 4, 8, 16]:
    discrete_signal = numpy.zeros(n)

    for i in range(0, round(n / d_t)):
        discrete_signal[i * d_t] = filter_signal[i * d_t]
    discrete_signals.append(discrete_signal)

    discrete_spectrum = fft.fft(discrete_signal)
    module_s = numpy.abs(fft.fftshift(discrete_spectrum))
    discrete_spectrums.append(module_s)

    discrete_signals_filtred.append(scipy.signal.sosfiltfilt(filter_discrete, discrete_signal))

    el = discrete_signals_filtred[-1] - filter_signal
    dispersion = numpy.var(el)
    snr = numpy.var(filter_signal) / dispersion
    dispersion_s.append(dispersion)
    snr_s.append(snr)

x3 = time
y3 = discrete_signals
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(x3, y3[s], linewidth=1)
        s += 1
fig.suptitle("Сигнал з кроком дискретизації Dt = [2, 4, 8, 16]", fontsize=14)
ax[1, 0].set_xlabel("Час (секунди)", fontsize=14)
ax[1, 1].set_xlabel("Час (секунди)", fontsize=14)
ax[0, 0].set_ylabel("Амплітуда сигналу", fontsize=14)
ax[1, 0].set_ylabel("Амплітуда сигналу", fontsize=14)
title3 = "Графік 3"
fig.savefig("./figures/" + title3 + ".png", dpi=600)

x4 = frequency_counts_shifted
y4 = discrete_spectrums
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s2 = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(x4, y4[s2], linewidth=1)
        s2 += 1
fig.suptitle("Спектри сигналів з кроком дискретизації Dt = [2, 4, 8, 16]", fontsize=14)
ax[1, 0].set_xlabel("Частота (Гц)", fontsize=14)
ax[1, 1].set_xlabel("Частота (Гц)", fontsize=14)
ax[0, 0].set_ylabel("Амплітуда спектру", fontsize=14)
ax[1, 0].set_ylabel("Амплітуда спектру", fontsize=14)
title4 = "Графік 4"
fig.savefig("./figures/" + title4 + ".png", dpi=600)

x5 = time
y5 = discrete_signals_filtred
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s3 = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(x5, y5[s3], linewidth=1)
        s3 += 1
fig.suptitle("Спектр сигналів з кроком дискретизації Dt = [2, 4, 8, 16]", fontsize=14)
ax[1, 0].set_xlabel("Час (секунди)", fontsize=14)
ax[1, 1].set_xlabel("Час (секунди)", fontsize=14)
ax[0, 0].set_ylabel("Амплітуда сигналу", fontsize=14)
ax[1, 0].set_ylabel("Амплітуда сигналу", fontsize=14)
title5 = "Графік 5"
fig.savefig("./figures/" + title5 + ".png", dpi=600)

x6 = [2, 4, 8, 16]
y6 = dispersion_s
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x6, y6, linewidth=1)
ax.set_xlabel("Крок дистанції", fontsize=14)
ax.set_ylabel("Дисперсія", fontsize=14)
plt.title("Залежність дисперсії від кроку дистанції", fontsize=14)
title6 = "Графік 6"
fig.savefig("./figures/" + title6 + ".png", dpi=600)

x7 = [2, 4, 8, 16]
y7 = snr_s
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x7, y7, linewidth=1)
ax.set_xlabel("Крок дискретизації", fontsize=14)
ax.set_ylabel("ССШ", fontsize=14)
plt.title("Залежність співвідношення сигналу-шуму від кроку дискретизації", fontsize=14)
title7 = "Графік 7"
fig.savefig("./figures/" + title7 + ".png", dpi=600)