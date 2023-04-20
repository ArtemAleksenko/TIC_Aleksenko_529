import numpy
import numpy as np
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
fig.suptitle("Відтворення аналогових сигналів з кроком дискретизації Dt = [2, 4, 8, 16]", fontsize=14)
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
ax.set_xlabel("Крок дискретизації", fontsize=14)
ax.set_ylabel("Дисперсія", fontsize=14)
plt.title("Залежність дисперсії від кроку дискретизації", fontsize=14)
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

quantize_signals = []
dispersion_ss = []
snr_ss = []

for M in [4, 16, 64, 256]:
    bits = []
    signal_from_bits = []
    delta = (numpy.max(filter_signal) - numpy.min(filter_signal)) / (M - 1)
    quantize_signal = delta * np.round(filter_signal / delta)

    quantize_levels = numpy.arange(numpy.min(quantize_signal), numpy.max(quantize_signal)+1, delta)
    quantize_bit = numpy.arange(0, M)
    quantize_bits = [format(bits, '0' + str(int(numpy.log(M) / numpy.log(2))) + 'b') for bits in quantize_bit]

    quantize_table = numpy.c_[quantize_levels[:M], quantize_bits[:M]]
    title = f'Таблиця квантування для {M} рівнів'
    fig, ax = plt.subplots(figsize=(14/2.54, M/2.54))
    table = ax.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fig.savefig('./figures/' + title + '.png', dpi=600)

    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if numpy.round(numpy.abs(signal_value-value), 0) == 0:
                bits.append(quantize_bits[index])
                break

    quantize_signals.append(quantize_signal)
    bits = [int(item) for item in list(''.join(bits))]

    e2 = quantize_signal - filter_signal
    dispersion_a = numpy.var(e2)
    snr_a = numpy.var(filter_signal) / dispersion_a
    dispersion_ss += [dispersion_a]
    snr_ss += [snr_a]

    x8 = numpy.arange(0, len(bits))
    y8 = bits
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.step(x8, y8, linewidth=0.1)
    ax.set_xlabel('Біти', fontsize=14)
    ax.set_ylabel('Амлітуда сигналу', fontsize=14)
    plt.title(f'Кодова послідовність сигналупри кількості рівнів квантування {M}', fontsize=14)
    title = f'Кодова послідовність сигналупри кількості рівнів квантування {M}'
    fig.savefig('./figures/' + title + '.png', dpi=600)

x9 = time
y9 = quantize_signals
fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
s4 = 0
for i in range(0, 2):
    for j in range(0, 2):

        ax[i][j].plot(x9, y9[s4], linewidth=1)

        s4 += 1

fig.supxlabel('Час (секунди)', fontsize=14)
fig.supylabel('Амлітуда сигналу', fontsize=14)
fig.suptitle('Цифрові сигнали з рівнями квантування (4, 16, 64, 256)', fontsize=14)
title = 'Цифрові сигнали з рівнями квантування (4, 16, 64, 256)'
fig.savefig('./figures/' + title + '.png', dpi=600)

x10 = [4, 16, 64, 256]
y10 = dispersion_ss
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(x10, y10, linewidth=1)
ax.set_xlabel('Кількість рівнів квантування', fontsize=14)
ax.set_ylabel('Дисперсія', fontsize=14)
plt.title('Залежність дисперсії від кількості рівнів квантування', fontsize=14)
title = 'Графік 10'
fig.savefig('./figures/' + title + '.png', dpi=600)

x11 = [4, 16, 64, 256]
y11 = snr_ss
fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
ax.plot(x11, y11, linewidth=1)
ax.set_xlabel('Кількість рівнів квантування', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title('Залежність співвідношення сигнал-шум від кількості рівнів квантування', fontsize=14)
title = 'Графік 11'
fig.savefig('./figures/' + title + '.png', dpi=600)
