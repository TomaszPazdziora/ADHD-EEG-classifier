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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from logger_config import setup_logger
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd 
import random


_logger = setup_logger(__name__)
SEP_NUM = 60
BRAIN_WAVES = ["delta", "theta", "alfa", "beta", "gamma"]
STATISTICAL_FEATURES = ["mean", "median", "variance", "std_dev"] #, "skew", "kurtosis"]

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
    # dwt_skew = skew(dwt_sig)
    # dwt_kurtosis = kurtosis(dwt_sig)
    return [dwt_mean, dwt_median, dwt_variance , dwt_std_dev] #, dwt_skew, dwt_kurtosis]



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

def save_features_to_csv(features: list, filename: str) -> None:
    data = {"feature names": feature_names}
    i = 0
    for single_signal_features in features:
        data[str(i)] =  single_signal_features
        i += 1
    df = pd.DataFrame(data)
    df.to_csv(filename+".csv", index=False)

def closest_index(arr, value):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - value))

def _get_feat_dict_for_hist(features: list) -> dict:
    num_of_patients = len(features) 
    # number of features for each patient - single electrode counted for single raw signal
    num_of_features = len(features[0])

    feature_dict = {}
    for i in range(num_of_features):
        feature_buff = []
        for j in range(num_of_patients):
            feature_buff.append(features[j][i])

        wave_name_idx = int(i / len(STATISTICAL_FEATURES)) 
        feature_name_idx = i % len(STATISTICAL_FEATURES)
        full_feature_name = BRAIN_WAVES[wave_name_idx] + '_' + STATISTICAL_FEATURES[feature_name_idx]
        feature_dict[full_feature_name] = feature_buff
    return feature_dict

def filter_outliers(data, threshold=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std_dev]

def save_features_hist(adhd_features: list, control_features: list) -> None:
    adhd_feat_dict = _get_feat_dict_for_hist(adhd_features)
    control_feat_dict = _get_feat_dict_for_hist(control_features)

    for (adhd_key, adhd_value), (control_key, control_value) in zip(adhd_feat_dict.items(), control_feat_dict.items()):
        # feat_min = min(min(adhd_value), min(control_value))
        # feat_max = max(max(adhd_value), max(control_value))

        # samples = np.linspace(feat_min, feat_max, num=10)
        # hist_adhd_values = []
        # hist_control_values = []
        
        # for f in adhd_value:
        #     hist_adhd_values.append(samples[closest_index(samples, f)])

        # for f in control_value:
        #     hist_control_values.append(samples[closest_index(samples, f)])
        adhd_feat_cnt = len(adhd_value)
        adhd_value = filter_outliers(adhd_value)
        after_filter_cnt = len(adhd_value)
        _logger.info(f"Filtered {adhd_feat_cnt - after_filter_cnt} adhd values. Feature set: {adhd_key}")

        plt.hist(adhd_value, histtype='stepfilled', alpha=0.3, bins=25, edgecolor='black', label='adhd')
        plt.hist(control_value, histtype='stepfilled', alpha=0.3, bins=25, edgecolor='black', label='control')
        plt.title('Histogram ' + adhd_key)
        plt.xlabel('Values')
        plt.ylabel('Number of occurences')
        save_dir = 'data' + os.sep + "features_histograms"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.legend()
        plt.savefig(save_dir + os.sep + adhd_key)
        plt.clf()


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

def count_accurancy(labels, predict):
    assert len(labels) == len(predict)  

    match_cnt = 0
    for i in range(len(labels)):
        if labels[i] == predict[i]:
            match_cnt += 1
    return match_cnt / len(labels)


if __name__ == "__main__":
    start_time = time.time()
    # TASK_SET = [1, 2, 7, 8]
    TASK_SET = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]

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

    # ====================================================================
    # feature extraction

    _logger.info("Feature extraction...")

    adhd_features = get_features_for_model(adhd_signals)
    # save_features_to_csv(adhd_features, "adhd_features")
    adhd_features_len = len(adhd_features)
    _logger.info(f"ahdh size: {adhd_features_len}")

    control_features = get_features_for_model(control_signals)
    # save_features_to_csv(control_features, "control_features")
    control_features_len = len(control_features)
    _logger.info(f"control size: {control_features_len}")

    save_features_hist(adhd_features=adhd_features, control_features=control_features)

    _logger.info(f"Features extracted in: {time.time() - start_time} s")
    _logger.info("=" * SEP_NUM)

    # ====================================================================
    # splitting into classes

    SPLIT_PERCENTAGE = 0.8
    random.shuffle(adhd_features)
    random.shuffle(control_features)

    # split to test and train groups 
    adhd_len = int(adhd_features_len * SPLIT_PERCENTAGE)
    _logger.info(f"ADHD train set len: {adhd_len}")
    _logger.info(f"All ADHD features len: {adhd_features_len}")
    _logger.info("-" * SEP_NUM)

    control_len = int(control_features_len * SPLIT_PERCENTAGE)
    _logger.info(f"Control train set len: {control_len}")
    _logger.info(f"All control features len: {control_features_len}")
    _logger.info("-" * SEP_NUM)

    train_set = adhd_features[:adhd_len]
    train_set.extend(control_features[:control_len])
    _logger.info(f"Train set len: {len(train_set)}")
    _logger.info("-" * SEP_NUM)

    labels = adhd_label * adhd_len + control_label * control_len
    str_labels = "adhd" * adhd_len + "control" * control_len


    start_time = time.time()
    # ====================================================================
    # Model selection and training

    # clf = RandomForestClassifier()
    clf = DecisionTreeClassifier(max_depth=5)    

    # Model training
    _logger.info("=" * SEP_NUM)
    _logger.info("Training model...")

    # clf.fit(X_train, y_train)
    clf.fit(train_set, labels)

    _logger.info(f"Model trained in: {time.time() - start_time}")
    _logger.info("=" * SEP_NUM)

    # Predictions
    adhd_test_group = adhd_features[adhd_len:]
    adhd_test_group_labels = (adhd_features_len - adhd_len) * adhd_label

    control_test_group = control_features[control_len:]
    control_test_group_labels = (control_features_len - control_len) * control_label

    adhd_predict = clf.predict(adhd_test_group)
    _logger.info(f"ADHD classification accuracy: {count_accurancy(adhd_test_group_labels, adhd_predict)}")

    control_predict = clf.predict(control_test_group)
    _logger.info(f"Control group classification accuracy: {count_accurancy(control_test_group_labels, control_predict)}")

    # plt.figure(figsize=(24, 16))
    # plot_tree(clf, filled=True, feature_names=feature_names, class_names=["ADHD", "Control"])
    # plt.title("Decision Tree")
    # plt.show()

    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(train_set)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000)
    reduced_data = tsne.fit_transform(np.array(train_set))    

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Klasy")
    plt.title('t-SNE - Wizualizacja cech')
    plt.xlabel('Pierwszy komponent główny')
    plt.ylabel('Drugi komponent główny')
    plt.grid(True)  

    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        train_set,
        labels,
        display_labels=["adhd","control"],
        cmap=plt.cm.Blues,
        normalize=None,
    )

    plt.title("Macierz pomyłek")
    plt.show()

    # averages = [sum(column) / len(column) for column in zip(*adhd_features)]

    # # stat_features = ["mean", "median", "variance", "std_dev", "skew", "kurtois"]
    # stat_features = ["mean", "median", "variance", "std_dev"]

    # i = 0
    # pr_adhd = []
    # for b in BRAIN_WAVES:
    #     for s in stat_features:
    #         pr_adhd.append(f"ADHD   - Wave: {b}, Feature: {s}, average: {averages[i]}")
    #         i += 1


    # averages = [sum(column) / len(column) for column in zip(*control_features)]

    # i = 0
    # pr_control = []

    # for b in BRAIN_WAVES:
    #     for s in stat_features:
    #         pr_control.append(f"CONTROL - Wave: {b}, Feature: {s}, average: {averages[i]}")
    #         i += 1


    # for i in range(len(pr_control)):
    #     print(pr_adhd[i])
    #     print(pr_control[i])
    #     print('-' * 60)
