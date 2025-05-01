import matplotlib.pyplot as plt
import numpy as np
from sig import Signal, PatientMeasurement
from scipy.stats import norm, kurtosis, skew
from pywt import wavedec
from logger_config import setup_logger
from pre_processing import iterate_over_whole_db_signals, iterate_over_whole_db, filter_all_db_signals, standarize_all_db_signals
from adult_db_loader import AdultDBLoader
import scipy.signal as signal
from scipy.signal import spectrogram
import os

_logger = setup_logger(__name__)
SEP_NUM = 60
BRAIN_WAVES = ["delta", "theta", "alfa", "beta", "gamma"]
STATISTICAL_FEATURES = ["średnia", "mediana", "wariancja",
                        "odchylenie_std.", "sekwens", "kurtoza", "śr._energia"]
SHORTEST_ADULT_DB_SIG = 3840

feature_names = []
for wave in BRAIN_WAVES:
    for feature in STATISTICAL_FEATURES:
        feature_names.append(wave + "_" + feature)

feature_title_names = [s.replace('_', ' ') for s in feature_names]
feature_file_names = [s.replace('.', '') for s in feature_names]


def get_statistical_features(dwt_sig: list) -> list:
    """calculates statistical features for given dwt signal

    Args:
        dwt_sig (list): single dwt signal - for example Alpha wave
    """
    dwt_mean = np.mean(dwt_sig)
    dwt_median = np.median(dwt_sig)
    dwt_variance = np.var(dwt_sig)
    dwt_std_dev = np.std(dwt_sig)
    dwt_skew = skew(dwt_sig)
    dwt_kurtosis = kurtosis(dwt_sig)
    mean_energy = np.mean(dwt_sig**2)
    return [dwt_mean, dwt_median, dwt_variance, dwt_std_dev, dwt_skew, dwt_kurtosis, mean_energy]


def get_all_waves_statistical_features(waves: dict) -> list:
    """format list of statistical waves for single dictonary with dwt output

    Args:
        waves (dict): dictonary of brain waves names and dwt output signals

    Returns:
        list: list of statistical features
    """
    all_waves_features = []
    for b in BRAIN_WAVES:
        all_waves_features.extend(get_statistical_features(waves[b]))
    return all_waves_features


def plot_waves(coefs: list, sig_path: str) -> None:
    delta = plt.subplot(3, 2, 1)
    delta.set_title("Fale delta")
    theta = plt.subplot(3, 2, 2)
    theta.set_title("Fale theta")
    alfa = plt.subplot(3, 2, 3)
    alfa.set_title("Fale alfa")
    beta = plt.subplot(3, 2, 4)
    beta.set_title("Fale beta")
    gamma = plt.subplot(3, 2, 5)
    gamma.set_title("Fale gamma do 64 Hz")
    gamma_hf = plt.subplot(3, 2, 6)
    gamma_hf.set_title("Fale gamma 64-128 Hz")
    plt.subplots_adjust(hspace=0.8, wspace=0.5)

    delta.plot(coefs[0])
    theta.plot(coefs[1])
    alfa.plot(coefs[2])
    beta.plot(coefs[3])
    gamma.plot(coefs[4])
    gamma_hf.plot(coefs[5])
    plt.savefig(sig_path)
    plt.clf()


def get_signal_features(signals: list[Signal]) -> list:
    """format list of features for model

    Args:
        signal_list (list): list of signals - each list element is a single signal from single channel

    Returns:
        list: list of features for training model
    """

    for sig in signals:
        waves = {}
        # BRAIN_WAVES = ["delta", "theta", "alfa", "beta", "gamma"]
        # THETA_IDX = 1
        # BETA_IDX = 3
        if sig.fs == 128:
            coefs = wavedec(sig.data, 'db4', level=4)
            for i in range(len(coefs)):
                waves[BRAIN_WAVES[i]] = coefs[i]

        elif sig.fs == 256:
            coefs = wavedec(sig.data, 'db4', level=5)

            # dir_path = f".{os.sep}plots{os.sep}waves{os.sep}{sig.meta.group}{os.sep}task{sig.meta.task}"
            # os.makedirs(dir_path, exist_ok=True)
            # file_path = dir_path + f"{os.sep}{sig.meta.group}_patient_{sig.meta.patient_idx}_electrode_{sig.meta.electrode}.png"
            # plot_waves(coefs=coefs, sig_path=file_path)

            for i in range(len(coefs)-1):
                waves[BRAIN_WAVES[i]] = coefs[i+1]

        sig.features = get_all_waves_statistical_features(waves)
        # frequencies, psd_values = signal.welch(sig.data, sig.fs, nperseg=256)
        # sig.features.extend(psd_values)


def get_all_db_signal_features(loader):
    _logger.info("Feature extraction...")
    iterate_over_whole_db_signals(loader, get_signal_features)


def get_measurement_features(measurement: PatientMeasurement):
    for sig in measurement.signals:
        measurement.features.extend(sig.features)


def get_add_db_measurement_features(loader):
    iterate_over_whole_db(loader, get_measurement_features)


def extract_all_db_features(loader):
    filter_all_db_signals(loader)
    standarize_all_db_signals(loader)
    get_all_db_signal_features(loader)
    get_add_db_measurement_features(loader)


def get_signals_as_measurement_features(measurement: PatientMeasurement):
    for sig in measurement.signals:
        measurement.features.append(sig.data[:SHORTEST_ADULT_DB_SIG])


def get_all_db_signals_as_measurement_features(loader):
    iterate_over_whole_db(loader, get_signals_as_measurement_features)


def load_all_raw_db_signals_to_measurement_features(loader):
    filter_all_db_signals(loader)
    standarize_all_db_signals(loader)
    get_all_db_signals_as_measurement_features(loader)


def load_features_for_model(loader: AdultDBLoader, features_type: str):
    adhd_set = []  # list of patient measurements
    control_set = []

    if features_type == "raw":
        load_all_raw_db_signals_to_measurement_features(loader)
    elif features_type == "cwt":
        extract_all_db_features(loader)
    else:
        raise ValueError("Unknown feature_type parameter!")

    if type(loader) == AdultDBLoader:
        for p_name in loader.measurements["FADHD"]:
            adhd_set.append(loader.measurements["FADHD"][p_name])
        for p_name in loader.measurements["MADHD"]:
            adhd_set.append(loader.measurements["MADHD"][p_name])

        for p_name in loader.measurements["FC"]:
            control_set.append(loader.measurements["FC"][p_name])
        for p_name in loader.measurements["MC"]:
            control_set.append(loader.measurements["MC"][p_name])
    else:
        raise ValueError("Incorrect loader type!")

    return adhd_set, control_set


if __name__ == "__main__":
    loader = AdultDBLoader()
    adhd, contorl = load_features_for_model(loader=loader, features_type="cwt")
