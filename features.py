from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import fft
import numpy as np
from data_loader import DataLoader, Signal, DB_NAMES
from scipy.stats import norm, kurtosis, skew
import os
from pywt import wavedec
import time
from sklearn.model_selection import  cross_val_score, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from logger_config import setup_logger
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import export_text
import pandas as pd 
import random
import argparse


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


def load_signal_from_dict(signal_dict: dict) -> list:
    signals = []
    for key, value in signal_dict.items():
        for p in range(value["number_of_patients"]):
            # ADHD woman - patient idx = 6 - 2 chanel signals are corrupted 
            if key == 'ADHD_WOMEN' and p == 6:
                continue
            else:
                for t in TASK_SET:
                    # signals.append(value["db"].get_signal(task_idx=t, patient_idx=p).ch1_data)
                    signals.append(value["db"].get_signal(task_idx=t, patient_idx=p).ch2_data)
    return signals


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
    return [dwt_mean, dwt_median, dwt_variance , dwt_std_dev, dwt_skew, dwt_kurtosis]


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


def filter_outliers(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    return [x for x in data if abs(x - mean) <= threshold * std_dev]


def save_features_hist(adhd_features: list, control_features: list) -> None:
    adhd_feat_dict = _get_feat_dict_for_hist(adhd_features)
    control_feat_dict = _get_feat_dict_for_hist(control_features)

    for (adhd_key, adhd_value), (control_key, control_value) in zip(adhd_feat_dict.items(), control_feat_dict.items()):
        # adhd_feat_cnt = len(adhd_value)
        # adhd_value = filter_outliers(adhd_value)
        # control_value = filter_outliers(control_value)
        # after_filter_cnt = len(adhd_value)
        # _logger.info(f"Filtered {adhd_feat_cnt - after_filter_cnt} adhd values. Feature set: {adhd_key}")

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
    parser=argparse.ArgumentParser(description="Model training parser")
    parser.add_argument("--method", type=str, required=True, help="tree, forest, knn and nn are allowed")
    parser.add_argument("--opt", action="store_true", help="Arg tell if script should optimize parameter for model - may took a lot of time")
    parser.add_argument("--hist", action="store_true", help="Generate figures with features histograms. By default set to False.")
    args=parser.parse_args()

    start_time = time.time()
    TASK_SET = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # definition of structures for iterating through them
    WOMEN_ADHD_DICT = {"db": DataLoader(DB_NAMES[0]), "number_of_patients": 11}
    WOMEN_CONTROL_DICT = {"db": DataLoader(DB_NAMES[1]), "number_of_patients": 13}
    MEN_ADHD_DICT = {"db": DataLoader(DB_NAMES[2]), "number_of_patients": 27}
    MEN_CONTROL_DICT = {"db": DataLoader(DB_NAMES[3]), "number_of_patients": 29}


    ADHD_DB_DICT = {
        "ADHD_WOMEN": WOMEN_ADHD_DICT, 
        "ADHD_MEN": MEN_ADHD_DICT, 
    }

    CONTROL_DB_DICT = {
        "CONTROL_WOMEN": WOMEN_CONTROL_DICT, 
        "CONTROL_MEN": MEN_CONTROL_DICT,
    }

    _logger.info("Loading signals...")
    
    adhd_signals = load_signal_from_dict(ADHD_DB_DICT)
    control_signals = load_signal_from_dict(CONTROL_DB_DICT)

    _logger.info(f"Signals loaded in: {time.time() - start_time} s")
    _logger.info("=" * SEP_NUM)

    # ====================================================================
    # extract features for different ML methods

    start_time = time.time()
    _logger.info("Feature extraction...")
    _logger.info("Filtering outliers features (only for historam visualisation). Training is not affected")

    adhd_features = get_features_for_model(adhd_signals)
    control_features = get_features_for_model(control_signals)
    if args.hist == True:
        save_features_hist(adhd_features=adhd_features, control_features=control_features)

    _logger.info(f"Features extracted in: {time.time() - start_time} s")
    _logger.info("=" * SEP_NUM)

    adhd_features_len = len(adhd_features)
    control_features_len = len(control_features)

    # ====================================================================
    # splitting into classes and groups

    random.shuffle(adhd_features)
    random.shuffle(control_features)

    cross_val_set = adhd_features
    cross_val_set.extend(control_features)
    cross_val_labels = adhd_label * adhd_features_len + control_label * control_features_len


    _logger.info(f"ADHD test set len: {adhd_features_len}")
    _logger.info(f"All ADHD features len: {adhd_features_len}")
    _logger.info("-" * SEP_NUM)

    _logger.info(f"Control test set len: {control_features_len}")
    _logger.info(f"All control features len: {control_features_len}")
    _logger.info("-" * SEP_NUM)

    _logger.info(f"All signals len (adhd + control): {adhd_features_len + control_features_len}")
    _logger.info("-" * SEP_NUM)
    
    
    # ====================================================================
    # Model selection and training
    clf_list = []
    param_list = []

    if args.method == "nn":
        if args.opt == True:
            param_list = [i for i in range(1, 200, 5)]
            for param in param_list:
                clf_list.append(MLPClassifier(hidden_layer_sizes=(param,), max_iter=1000, random_state=42))
        elif args.opt == False:
            clf = MLPClassifier(hidden_layer_sizes=(85,), max_iter=1000, random_state=42)

    elif args.method == "forest":
        if args.opt == True:
            param_list = [i for i in range(1, 100, 5)]
            for param in param_list:
                clf_list.append(RandomForestClassifier(n_estimators=param, random_state=42))
        elif args.opt == False:
            clf = RandomForestClassifier(n_estimators=80, random_state=42)

    elif args.method == "tree":
        if args.opt == True:
            param_list = [i for i in range(1, 100, 5)]
            for param in param_list:
                clf_list.append(DecisionTreeClassifier(random_state=param))
        elif args.opt == False:
            clf = DecisionTreeClassifier(random_state=42)


    elif args.method == "knn":
        if args.opt == True:
            param_list = [i for i in range(1, 150)]
            for param in param_list:
                clf_list.append(KNeighborsClassifier(n_neighbors=param))
        elif args.opt == False:
            clf = KNeighborsClassifier(n_neighbors=1)

    start_time = time.time()
    cv = StratifiedKFold(n_splits=10)
    
    if args.opt == True:
        max_acc = 0
        best_parameter = 0
        acc_list = []

        for param, clf in zip(param_list, clf_list):
            scores = cross_val_score(clf, cross_val_set, cross_val_labels, cv=cv)
            print("Cross-validation scores:", scores)
            
            mean = sum(scores) / len(scores)
            acc_list.append(mean)
            _logger.info(f"Cross-validation mean: {mean}, parameter: {param}")

            if mean > max_acc:
                best_parameter = param
                max_acc = mean

        _logger.info(80*'=')
        _logger.info(f"Max accurancy: {max_acc}, best parameter: {best_parameter}")
        _logger.info(80*'=')

        plt.plot(param_list, acc_list)
        plt.title(args.method)
        plt.xlabel('Wartość parametru')
        plt.ylabel('Skuteczność')
        plt.grid()
        plt.show()

    if args.opt == False:
        y_pred = cross_val_predict(clf, cross_val_set, cross_val_labels, cv=cv)
        
        # Compute confusion matrix
        cm = confusion_matrix(cross_val_labels, y_pred)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=["ADHD", "control"], yticklabels=["ADHD", "control"], cmap=plt.cm.Blues)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title("Macierz pomyłek - Tree")
        
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000)
        reduced_data = tsne.fit_transform(np.array(cross_val_set)) 

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cross_val_labels, cmap='viridis')
        plt.legend(*scatter.legend_elements(), title="Klasy")
        plt.title('t-SNE - Wizualizacja cech')
        plt.xlabel('Pierwszy komponent główny')
        plt.ylabel('Drugi komponent główny')
        plt.grid(True)  

        plt.show()
