from sig import Signal
from children_db_loader import ChildrenDBLoader
from adult_db_loader import AdultDBLoader
from pre_processing import standarize_all_db_signals, filter_all_db_signals, iterate_over_whole_db_signals

import matplotlib.pyplot as plt
import numpy as np
import os
from logger_config import setup_logger
_logger = setup_logger(__name__)


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
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}time{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}time{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")        
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
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}fft{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}fft{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")
        plt.savefig(file_path)
    else:
        plt.show()


def save_signals_to_files(singals: list[Signal]):
    # saves all plots as png files in dedicated folders
    cnt = 0
    for sig in singals:
        plot_fft(sig=sig, save_to_file=True)
        plt.clf()
        plot_sig_in_time(sig=sig, save_to_file=True)
        plt.clf()
        cnt += 2
    _logger.info(f"Generated {cnt} plots")

def save_all_db_signals_to_file(data_loader):
    iterate_over_whole_db_signals(data_loader, save_signals_to_files)

if __name__ == "__main__":
    # loader = ChildrenDBLoader()
    loader = AdultDBLoader()
    filter_all_db_signals(loader)
    standarize_all_db_signals(loader)
    save_all_db_signals_to_file(loader)

    # signals = c.load_all_patients_signals_for_single_electrode("ADHD", "F4")
    # filter_signals(signals)
    # standardize_signals(signals)
    # save_signals_to_files(signals[:10])
