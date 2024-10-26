from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
from data_loader import DataLoader, Signal


def plot_fft(signal: Signal):
    N = len(signal.samples)
    Fs = signal.frequency

    fft_result = np.fft.fft(signal.samples)
    fft_magnitude = np.abs(fft_result)

    frequencies = np.fft.fftfreq(N, d=1/Fs)

    plt.figure(figsize=(8, 4))
    plt.plot(frequencies[:N//2], fft_magnitude[:N//2])
    plt.title(f'FFT - database: {signal.db_name}, task: {signal.task_idx+1},\
patient: {signal.patient_idx+1}, electrode: {signal.electode_type}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    # Nyquist frequency constraints
    plt.xlim(0, Fs/2)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    fc = DataLoader("FC")
    fadhd = DataLoader("FADHD")

    sig_chann_1 = fc.get_single_signal(
        task_idx=0, patient_idx=0, electrode_idx=0
    )
    sig_chann_2 = fc.get_single_signal(
        task_idx=0, patient_idx=0, electrode_idx=1
    )

    sig = fc.get_single_signal(task_idx=0, patient_idx=0, electrode_idx=0)
    plot_fft(signal=sig)
