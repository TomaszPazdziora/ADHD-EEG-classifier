from sig import Signal
from children_db_loader import ChildrenDBLoader
from adult_db_loader import AdultDBLoader
from pre_processing import standardize_signals, filter_signals

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_sig_in_time(sig: Signal, save_to_file=False):
    plt.plot(sig.data)
    plt.title(
        f"Time sig - "
        f"{sig.meta.db_name},"
        f"group: {sig.meta.group}, "
        f"patient: {sig.meta.patient_idx}, "
        f"electrode: {sig.meta.electrode}, "
        f"task: {sig.meta.task}, "
    )

    if save_to_file == True:
        if sig.meta.task != -1:
            dir_path = f".{os.sep}plots{os.sep}time{os.sep}DB_{sig.meta.db_name}{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}time{os.sep}DB_{sig.meta.db_name}{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        plt.savefig(file_path)
    else:
        plt.show()


def plot_fft(sig: Signal, save_to_file=False):
    fft_sig = np.fft.fft(sig.data)
    fft_magnitude = np.abs(fft_sig)

    # x axis generation
    frequencies = np.fft.fftfreq(sig.num_of_samples, d=1/sig.fs)

    plt.plot(frequencies[:sig.num_of_samples//2], fft_magnitude[:sig.num_of_samples//2])
    plt.title(
        f"FFT - "
        f"{sig.meta.db_name},"
        f"group: {sig.meta.group}, "
        f"patient: {sig.meta.patient_idx}, "
        f"electrode: {sig.meta.electrode}, "
        f"task: {sig.meta.task}, "
    )
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Nyquist frequency constraints
    plt.xlim(0, sig.fs/2)

    if save_to_file == True:
        if sig.meta.task != -1:
            dir_path = f".{os.sep}plots{os.sep}fft{os.sep}DB_{sig.meta.db_name}{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}fft{os.sep}DB_{sig.meta.db_name}{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        plt.savefig(file_path)
    else:
        plt.show()


def save_signals_to_files(singals: list[Signal]):
    # saves all plots as png files in dedicated folders
    for sig in singals:
        plot_fft(sig=sig, save_to_file=True)
        plt.clf()
        plot_sig_in_time(sig=sig, save_to_file=True)
        plt.clf()


if __name__ == "__main__":
    c = ChildrenDBLoader()
    signals = c.load_all_patients_signals_for_single_electrode("ADHD", "F4")
    filter_signals(signals)
    standardize_signals(signals)
    save_signals_to_files(signals[:10])
