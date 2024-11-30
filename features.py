from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
from data_loader import DataLoader, Signal, DB_NAMES
from scipy.stats import norm, kurtosis, skew
import os
from pywt import wavedec
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from logger_config import setup_logger


_logger = setup_logger(__name__)
SEP_NUM = 60
BRAIN_WAVES = ["delta", "theta", "alfa", "beta", "gamma"]
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
    return [dwt_mean, dwt_median, dwt_variance, dwt_std_dev, dwt_skew, dwt_kurtosis]


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


def get_features_for_model(signal_list: list) -> list:
    """format list of features for model

    Args:
        signal_list (list): list of signals - each list element is a single signal from single channel

    Returns:
        list: list of features for training model
    """
    features = []

    for feature in signal_list:
        coefs = wavedec(feature, 'db1', level=5)
        waves = {}
        for i in range(len(coefs)-1):
            waves[BRAIN_WAVES[i]] = coefs[i+1]
        features.append(get_all_waves_statistical_features(waves))
    return features


def plot_waves(waves: dict) -> None:
    delta = plt.subplot(3,2,1)
    theta = plt.subplot(3,2,2)
    alfa = plt.subplot(3,2,3)
    beta = plt.subplot(3,2,4)
    gamma = plt.subplot(3,2,5)

    delta_x = [i for i in range(len(waves["delta"]))]
    delta.plot(delta_x, waves["delta"])

    theta_x = [i for i in range(len(waves["theta"]))]
    theta.plot(theta_x, waves["theta"])

    alfa_x = [i for i in range(len(waves["alfa"]))]
    alfa.plot(alfa_x, waves["alfa"])

    beta_x = [i for i in range(len(waves["beta"]))]
    beta.plot(beta_x, waves["beta"])

    gamma_x = [i for i in range(len(waves["gamma"]))]
    gamma.plot(gamma_x, waves["gamma"])

    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    # TASK_SET = [1, 2, 7, 8]
    TASK_SET = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # definition of structures for iterating through them
    WOMEN_ADHD_DICT = {"db": DataLoader(DB_NAMES[0]), "number_of_patients": 11}
    WOMEN_CONTROL_DICT = {"db": DataLoader(DB_NAMES[1]), "number_of_patients": 13}
    MEN_ADHD_DICT = {"db": DataLoader(DB_NAMES[2]), "number_of_patients": 27}
    MEN_CONTROL_DICT = {"db": DataLoader(DB_NAMES[3]), "number_of_patients": 29}


    ADHD_DB_DICT = {
        "WOMEN": WOMEN_ADHD_DICT, 
        "MEN": MEN_ADHD_DICT, 
    }

    CONTROL_DB_DICT = {
        "WOMEN": WOMEN_CONTROL_DICT, 
        "MEN": MEN_CONTROL_DICT,
    }

    _logger.info("Loading signals...")
    
    # adhd singals loading
    adhd_signals = []
    for key, value in ADHD_DB_DICT.items():
        for p in range(value["number_of_patients"]):
            for t in TASK_SET:
                adhd_signals.append(value["db"].get_signal(task_idx=t, patient_idx=p).ch1_data)
                adhd_signals.append(value["db"].get_signal(task_idx=t, patient_idx=p).ch2_data)

    # control singals loading
    control_signals = []
    for key, value in CONTROL_DB_DICT.items():
        for p in range(value["number_of_patients"]):        
            for t in TASK_SET:
                control_signals.append(value["db"].get_signal(task_idx=t, patient_idx=p).ch1_data)
                control_signals.append(value["db"].get_signal(task_idx=t, patient_idx=p).ch2_data)


    _logger.info(f"Signals loaded in: {time.time() - start_time} s")
    _logger.info("=" * SEP_NUM)
    start_time = time.time()

    # feature extraction
    _logger.info("Feature extraction...")

    adhd_features = get_features_for_model(adhd_signals)
    adhd_features_len = len(adhd_features)
    _logger.info(f"ahdh size: {adhd_features_len}")

    control_features = get_features_for_model(control_signals)
    control_features_len = len(control_features)
    _logger.info(f"control size: {control_features_len}")

    all_features = adhd_features
    all_features.extend(control_features)
    _logger.info(f"all features size {len(all_features)}")

    _logger.info(f"Features extracted in: {time.time() - start_time} s")
    _logger.info("=" * SEP_NUM)

    # split to test and train groups 
    labels = adhd_label * adhd_features_len + control_label * control_features_len
    X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.3)

    start_time = time.time()
    _logger.info(f"Training set size: {len(X_train)}")
    _logger.info(f"Testing set size: {len(X_test)}")  

    clf = RandomForestClassifier()

    # Model training
    _logger.info("=" * SEP_NUM)
    _logger.info("Training model...")

    clf.fit(X_train, y_train)

    _logger.info(f"Model trained in: {time.time() - start_time}")
    _logger.info("=" * SEP_NUM)

    # Test group predict
    y_pred = clf.predict(X_test)
    _logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
