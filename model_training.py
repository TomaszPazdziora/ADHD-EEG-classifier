from sklearn.ensemble import RandomForestClassifier
from adult_db_loader import AdultDBLoader
from children_db_loader import ChildrenDBLoader
from features import extract_all_db_features, load_all_raw_db_signals_to_measurement_features, load_features_for_model
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from logger_config import setup_logger
import time
from sklearn.neighbors import KNeighborsClassifier
import random

_logger = setup_logger(__name__)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Model training parser")
    parser.add_argument("--method", type=str, required=True, help="tree, forest, knn and nn are allowed")
    parser.add_argument("--opt", action="store_true", help="Arg tell if script should optimize parameter for model - may took a lot of time")
    args=parser.parse_args()

    loader = AdultDBLoader()
    # loader = ChildrenDBLoader()
    adhd_set, control_set = load_features_for_model(loader=loader, features_type="cwt")
    clf_list = []
    param_list = []

    if args.method == "nn":
        if args.opt == True:
            param_list = [i for i in range(1, 400, 20)]
            for param in param_list:
                clf_list.append(MLPClassifier(hidden_layer_sizes=(param,), max_iter=1000, random_state=42))
        elif args.opt == False:
            clf = MLPClassifier(hidden_layer_sizes=(85,), max_iter=1000, random_state=42)

    elif args.method == "forest":
        if args.opt == True:
            param_list = [i for i in range(1, 400, 20)]
            for param in param_list:
                clf_list.append(RandomForestClassifier(n_estimators=param, random_state=42))
        elif args.opt == False:
            clf = RandomForestClassifier(n_estimators=80, random_state=42)

    elif args.method == "knn":
        if args.opt == True:
            param_list = [i for i in range(1, 50)]
            for param in param_list:
                clf_list.append(KNeighborsClassifier(n_neighbors=param))
        elif args.opt == False:
            clf = KNeighborsClassifier(n_neighbors=1)


    cross_val_set = adhd_set
    cross_val_set.extend(control_set)
    random.shuffle(cross_val_set)

    cross_val_features = []
    cross_val_labels = []
    adhd_features = 0
    control_features = 0
    ADHD_LABEL = 0
    CONTROL_LABEL = 1

    for measurement in cross_val_set:
        if "ADHD" in measurement.signals[0].meta.group:
            adhd_features += 1
            cross_val_labels.append(ADHD_LABEL)
        else:
            control_features += 1
            cross_val_labels.append(CONTROL_LABEL)
        cross_val_features.append(measurement.features)

    _logger.info(f"ADHD features len: {adhd_features}")
    _logger.info(f"Control features len: {control_features}")

    before = time.time()
    cv = StratifiedKFold(n_splits=10)

    if args.opt == True:
        max_acc = 0
        best_parameter = 0
        acc_list = []

        for param, clf in zip(param_list, clf_list):
            scores = cross_val_score(clf, cross_val_features, cross_val_labels, cv=cv)
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
        y_pred = cross_val_predict(clf, cross_val_features, cross_val_labels, cv=cv)
        
        # Compute confusion matrix
        cm = confusion_matrix(cross_val_labels, y_pred)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=["ADHD", "control"], yticklabels=["ADHD", "control"], cmap=plt.cm.Blues)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title("Macierz pomyłek - Tree")
        
    # y_pred = cross_val_predict(clf, cross_val_set, cross_val_labels, cv=cv)
    # print(time.time() - before)
        
    # # Compute confusion matrix
    # cm = confusion_matrix(cross_val_labels, y_pred)

    # # Plot confusion matrix
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=["ADHD", "control"], yticklabels=["ADHD", "control"], cmap=plt.cm.Blues)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title("Macierz pomyłek - Tree")
    # plt.show()

    # print(cm)
    # print('aaa')
