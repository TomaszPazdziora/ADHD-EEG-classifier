import matplotlib.pyplot as plt
import numpy as np
from sig import Signal, PatientMeasurement
from scipy.stats import norm, kurtosis, skew
from pywt import wavedec
from logger_config import setup_logger
from pre_processing import iterate_over_whole_db_signals, iterate_over_whole_db, filter_all_db_signals, standarize_all_db_signals
from adult_db_loader import AdultDBLoader
from children_db_loader import ChildrenDBLoader
import scipy.signal as signal

_logger = setup_logger(__name__)
SEP_NUM = 60
BRAIN_WAVES = ["delta", "theta", "alfa", "beta", "gamma"]
STATISTICAL_FEATURES = ["mean", "median", "variance", "std_dev", "skew", "kurtosis"]

feature_names = []
for wave in BRAIN_WAVES:
    for feature in STATISTICAL_FEATURES:
        feature_names.append(wave + " - " + feature)

adhd_label = [0]
control_label = [1]


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
    mean_energy = np.mean(np.sum(dwt_sig**2))
    # max_val = max(dwt_sig)
    # min_val = min(dwt_sig)
    return [dwt_mean, dwt_median, dwt_variance, dwt_std_dev, dwt_skew, dwt_kurtosis, mean_energy] #, max_val, min_val]


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
            for i in range(len(coefs)-1):
                waves[BRAIN_WAVES[i]] = coefs[i+1]

        frequencies, psd_values = signal.welch(sig.data, sig.fs, nperseg=256)
        sig.features = (get_all_waves_statistical_features(waves))
        sig.features.extend(psd_values)

def get_all_db_signal_features(loader):
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
        measurement.features.extend(sig.data)

def get_all_db_signals_as_measurement_features(loader):
    iterate_over_whole_db(loader, get_signals_as_measurement_features)

def load_all_raw_db_signals_to_measurement_features(loader):
    filter_all_db_signals(loader)
    standarize_all_db_signals(loader)
    get_all_db_signals_as_measurement_features(loader)


if __name__ == "__main__":
    # loader = AdultDBLoader()
    loader = ChildrenDBLoader()

