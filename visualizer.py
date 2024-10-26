from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
from data_loader import DataLoader, Signal


def plot_fft(signal: Signal):
    """Plots two fft graphs on one figure. The data originates from a single task related to one patient, collected through two different channels."""
    # both channels have the same number of samples and the same freq
    N = sig.num_of_samples
    Fs = sig.frequency

    # calculate fft of both channels
    fft_sig = np.fft.fft(sig.ch1_data)
    fft_magnitude_ch1 = np.abs(fft_sig)

    fft_ch2 = np.fft.fft(sig.ch2_data)
    fft_magnitude_ch2 = np.abs(fft_ch2)

    # x axis generation
    frequencies = np.fft.fftfreq(N, d=1/Fs)

    ax_ch1 = plt.subplot(211)
    ax_ch1.plot(frequencies[:N//2], fft_magnitude_ch1[:N//2])
    ax_ch1.set_title(f'FFT - database: {sig.db_name}, task: {sig.task_idx+1}, \
patient: {sig.patient_idx+1}, channel: {sig.ch1_type}')
    ax_ch1.set_xlabel('Frequency (Hz)')
    ax_ch1.set_ylabel('Magnitude')
    # Nyquist frequency constraints
    ax_ch1.set_xlim(0, Fs/2)
    ax_ch1.grid()

    plt.subplots_adjust(hspace=0.5)

    ax_ch2 = plt.subplot(212)
    ax_ch2.plot(frequencies[:N//2], fft_magnitude_ch2[:N//2], color='red')
    ax_ch2.set_title(f'FFT - database: {sig.db_name}, task: {sig.task_idx+1}, \
patient: {sig.patient_idx+1}, channel: {sig.ch2_type}')
    ax_ch2.set_xlabel('Frequency (Hz)')
    ax_ch2.set_ylabel('Magnitude')
    # Nyquist frequency constraints
    ax_ch2.set_xlim(0, Fs/2)
    ax_ch2.grid()

    # standarize y axis
    min_mag = min(min(fft_magnitude_ch1), min(fft_magnitude_ch2))
    max_mag = max(max(fft_magnitude_ch1), max(fft_magnitude_ch2))
    ax_ch1.set_ylim(min_mag, max_mag)
    ax_ch2.set_ylim(min_mag, max_mag)

    plt.show()


if __name__ == "__main__":
    fc = DataLoader("FC")
    sig = fc.get_signal(task_idx=0, patient_idx=0)
    plot_fft(signal=sig)
