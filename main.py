from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
from common_values import *
from data_loader import DataLoader


def dummy_plot(fc, fadhd):
    ax_fc = plt.subplot(211)
    ax_fc.set_title("Woman control gorup")
    ax_fadhd = plt.subplot(212)
    ax_fadhd.set_title("Woman ADHD")

    x = fc.N
    y0 = fc.get_single_signal(patient_id=0, task_id=0, electrode_id=0)
    y1 = fc.get_single_signal(patient_id=0, task_id=0, electrode_id=1)

    y_adhd0 = fadhd.get_single_signal(patient_id=0, task_id=0, electrode_id=0)
    y_adhd1 = fadhd.get_single_signal(patient_id=0, task_id=0, electrode_id=1)

    plt.subplots_adjust(hspace=0.5)

    ax_fc.plot(x, y0, linewidth=1)
    ax_fadhd.plot(x, y_adhd0, linewidth=1)
    ax_fc.plot(x, y1, color='red', linewidth=1)
    ax_fadhd.plot(x, y_adhd1, color='red', linewidth=1)

    min_ampl = min(min(y_adhd0), min(y0), min(y_adhd1), min(y1))
    max_ampl = max(max(y_adhd0), max(y0), max(y_adhd1), max(y1))

    ax_fc.set_ylim(min_ampl, max_ampl)
    ax_fadhd.set_ylim(min_ampl, max_ampl)

    plt.show()


def plot_fft(signal: list[float], task_id: int):
    Fs = TASK_MEASUREMENT_FREQUENCY[task_id]
    print(f"sampling freq: {Fs}")

    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)

    frequencies = np.fft.fftfreq(N, d=1/Fs)

    plt.figure(figsize=(8, 4))
    plt.plot(frequencies[:N//2], fft_magnitude[:N//2])
    plt.title('FFT of EEG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    # Nyquist frequency constraints
    plt.xlim(0, Fs/2)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    fc = DataLoader("FC")
    fadhd = DataLoader("FADHD")

    sig = fc.get_single_signal(patient_id=0, task_id=0, electrode_id=0)
    plot_fft(signal=sig, task_id=0)
