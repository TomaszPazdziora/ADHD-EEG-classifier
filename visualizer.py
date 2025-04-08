from sig import Signal, PatientMeasurement
from adult_db_loader import AdultDBLoader, _TASK_CHANNELS
from pre_processing import iterate_over_whole_db_signals
from features import feature_names
from scipy.signal import spectrogram

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
        file_path = dir_path + \
            f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")
        plt.savefig(file_path)
    else:
        plt.show()


def plot_fft(sig: Signal, save_to_file=False):
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
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}fft{os.sep}task{sig.meta.task}"
        else:
            dir_path = f".{os.sep}plots{os.sep}DB_{sig.meta.db_name}{os.sep}fft{os.sep}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + \
            f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
        _logger.info(f"saving file: {file_path}")
        plt.savefig(file_path)
    else:
        plt.show()


def plot_spect(sig: Signal, save_to_file=False):
    sig_arr = np.array(sig.data)
    f, t_spec, Sxx = spectrogram(sig_arr, sig.fs)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Częstotliwość [Hz]')
    plt.xlabel('Czas [s]')
    plt.title('Spektrogram sygnału')
    plt.colorbar(label='Moc [dB]')
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
    loader = AdultDBLoader()
    # adhd_features, control_features = load_features_for_model(loader=loader, features_type="cwt")
    # save_features_histograms(adhd_features, control_features)

    plot_spect(loader.measurements["FADHD"]["patient_0"].signals[0])

    # signals = c.load_all_patients_signals_for_single_electrode("ADHD", "F4")
    # filter_signals(signals)
    # standardize_signals(signals)
    # save_signals_to_files(signals[:10])
