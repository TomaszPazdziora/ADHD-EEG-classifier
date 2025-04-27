from sig import Signal, PatientMeasurement
from adult_db_loader import AdultDBLoader, _TASK_CHANNELS
from pre_processing import iterate_over_whole_db_signals
from features import feature_names, load_features_for_model
from scipy.signal import spectrogram
from pre_processing import filter_all_db_signals, standarize_all_db_signals
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from logger_config import setup_logger
_logger = setup_logger(__name__)

use_filters = False


def plot_sig_in_time(sig: Signal, save_to_file=False):
    global use_filters
    if use_filters == True:
        filt_str = "filtered"
    else:
        filt_str = "not_filtered"

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
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}{filt_str}{os.sep}time{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}{filt_str}{os.sep}time{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + \
            f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")
        plt.savefig(file_path)
    else:
        plt.show()


def plot_fft(sig: Signal, save_to_file=False):
    global use_filters
    if use_filters == True:
        filt_str = "filtered"
    else:
        filt_str = "not_filtered"

    fft_sig = np.fft.fft(sig.data)
    fft_magnitude = np.abs(fft_sig)
    # x axis generation
    frequencies = np.fft.fftfreq(sig.num_of_samples, d=1/sig.fs)

    plt.plot(frequencies[:sig.num_of_samples//2],
             fft_magnitude[:sig.num_of_samples//2])
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
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}{filt_str}{os.sep}fft{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{filt_str}{os.sep}{os.sep}fft{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + \
            f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")
        plt.savefig(file_path)
    else:
        plt.show()


def plot_spect(sig: Signal, save_to_file=False):
    global use_filters
    if use_filters == True:
        filt_str = "filtered"
    else:
        filt_str = "not_filtered"

    sig_arr = np.array(sig.data)
    f, t_spec, Sxx = spectrogram(sig_arr, sig.fs)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Częstotliwość [Hz]')
    plt.xlabel('Czas [s]')
    plt.title('Spektrogram sygnału')
    plt.colorbar(label='Moc [dB]')

    if save_to_file == True:
        if sig.meta.task != -1:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}{filt_str}{os.sep}spect{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}{filt_str}{os.sep}spect{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + \
            f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")
        plt.savefig(file_path)
    else:
        plt.show()


def save_raw_signals_to_files(singals: list[Signal]):
    # saves all plots as png files in dedicated folders
    for sig in singals:
        plot_sig_in_time(sig=sig, save_to_file=True)
        plt.clf()


def save_all_db_raw_signals_to_file(data_loader):
    iterate_over_whole_db_signals(data_loader, save_raw_signals_to_files)


def save_fft_to_files(singals: list[Signal]):
    # saves all fft signals as png files in dedicated folders
    for sig in singals:
        plot_fft(sig=sig, save_to_file=True)
        plt.clf()


def save_all_db_fft_to_file(data_loader):
    iterate_over_whole_db_signals(data_loader, save_fft_to_files)


def save_spect_to_files(singals: list[Signal]):
    # saves all signals spect as png files in dedicated folders
    for sig in singals:
        plot_spect(sig=sig, save_to_file=True)
        plt.clf()


def save_all_db_spect_to_file(data_loader):
    iterate_over_whole_db_signals(data_loader, save_spect_to_files)


def save_features_histograms(adhd: list[PatientMeasurement], control: list[PatientMeasurement]):
    for task_electrode_idx in range(0, 22):
        for i, name in enumerate(feature_names):
            adhd_hist = []
            control_hist = []

            for meas in adhd:
                adhd_hist.append(
                    meas.features[task_electrode_idx * len(feature_names) + i])
            for meas in control:
                control_hist.append(
                    meas.features[task_electrode_idx * len(feature_names) + i])

            _logger.info(f"number of hist values (adhd): {len(adhd_hist)}")
            _logger.info(
                f"number of hist values (control): {len(control_hist)}")

            # 79 * 22 taski
            task_idx = int(task_electrode_idx/2)
            electrode_idx = task_electrode_idx % 2
            electrode = _TASK_CHANNELS[task_idx][electrode_idx]
            save_dir = f"plots{os.sep}features_histograms{os.sep}task_{task_idx}"

            plt.hist(adhd_hist, histtype='stepfilled', alpha=0.3,
                     bins=25, edgecolor='black', label='adhd')
            plt.hist(control_hist, histtype='stepfilled', alpha=0.3,
                     bins=25, edgecolor='black', label='control')
            plt.title('Histogram ' + name + ", zadanie: " +
                      str(task_idx) + ", elektroda: " + electrode)
            plt.xlabel('Values')
            plt.ylabel('Number of occurences')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.legend()
            plt.savefig(save_dir + os.sep + name + "_" + electrode)
            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signals visualization script. To run properly use one or more args from the given list: [--raw, --fft, --hist, --spect] with or without --filt option.\n")
    parser.add_argument("--filt", action="store_true",
                        help="apply 50Hz filter and standarization for loaded signals (for the --hist option filtration is allways applied)")
    parser.add_argument("--raw", action="store_true",
                        help="saves to files raw signals visualization")
    parser.add_argument("--fft", action="store_true",
                        help="saves to files signals fft visualization")
    parser.add_argument("--hist", action="store_true",
                        help="saves to files feature histograms")
    parser.add_argument("--spect", action="store_true",
                        help="saves to files signals spectrogram visualization")

    args = parser.parse_args()

    if args.raw == False and args.fft == False and args.hist == False and args.spect == False:
        raise ValueError(
            "User provided incorrect args! \nPlease execute `python3 visualizer.py -h` to check how to properly use the script!")

    loader = AdultDBLoader()

    if args.hist == True:
        adhd_features, control_features = load_features_for_model(
            loader=loader, features_type="cwt")
        save_features_histograms(adhd_features, control_features)
    if args.filt == True and args.hist == False:
        use_filters = True
        filter_all_db_signals(loader)
        standarize_all_db_signals(loader)
    if args.raw == True:
        save_all_db_raw_signals_to_file(loader)
    if args.fft == True:
        save_all_db_fft_to_file(loader)
    if args.spect == True:
        save_all_db_spect_to_file(loader)

    # plot_spect(loader.measurements["FADHD"]["patient_0"].signals[0])
