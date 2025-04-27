from sklearn.ensemble import RandomForestClassifier
from adult_db_loader import AdultDBLoader
from features import load_features_for_model
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from logger_config import setup_logger
import time
from sklearn.neighbors import KNeighborsClassifier
from argparse import RawTextHelpFormatter
import random

ADHD_LABEL = 0
CONTROL_LABEL = 1

# Training parameters for no optimization option
K_FOLD_SPLITS = 10
NO_OPT_MPL_LAYERS = 85
NO_OPT_KNN_NEIGHBOURS = 10
NO_OPT_FOREST_TREES = 80

# Training parameters for --opt arg
OPT_PARAM_LIST_MPL = [i for i in range(1, 400, 20)]
OPT_PARAM_LIST_KNN = [i for i in range(1, 50)]
OPT_PARAM_LIST_FOREST = [i for i in range(1, 400, 20)]

_logger = setup_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training script.\n"
                                     "-------------------------------\n"
                                     "Examples:\n"
                                     "python3 model_training.py --method knn --opt\n"
                                     "          ^ performs model training for KNN method with parameter optimization\n\n"
                                     "python3 model_training.py --method mpl\n"
                                     "          ^ performs MPL model training for single parameter",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("--method", type=str, required=True,
                        help="forest, knn and mpl are allowed")
    parser.add_argument("--opt", action="store_true",
                        help="perform parameter optimization - may took some time")
    args = parser.parse_args()

    # Load signals and extract features
    loader = AdultDBLoader()
    adhd_set, control_set = load_features_for_model(
        loader=loader, features_type="cwt")
    clf_list = []
    param_list = []

    # Choose training model and optimization parameters
    if args.method == "mpl":
        if args.opt == True:
            param_list = OPT_PARAM_LIST_MPL
            _logger.info(
                f"Training mpl model for given parameter list: {param_list}")
            for param in param_list:
                clf_list.append(MLPClassifier(hidden_layer_sizes=(
                    param,), max_iter=1000, random_state=42))
        elif args.opt == False:
            clf = MLPClassifier(hidden_layer_sizes=(
                NO_OPT_MPL_LAYERS,), max_iter=1000, random_state=42)
            _logger.info(
                f"Training mpl model for given parameter: {NO_OPT_MPL_LAYERS}")

    elif args.method == "forest":
        if args.opt == True:
            param_list = OPT_PARAM_LIST_FOREST
            _logger.info(
                f"Training random forest model for given parameter list: {param_list}")
            for param in param_list:
                clf_list.append(RandomForestClassifier(
                    n_estimators=param, random_state=42))
        elif args.opt == False:
            clf = RandomForestClassifier(
                n_estimators=NO_OPT_FOREST_TREES, random_state=42)
            _logger.info(
                f"Training random forest model for given parameter: {NO_OPT_FOREST_TREES}")

    elif args.method == "knn":
        if args.opt == True:
            param_list = OPT_PARAM_LIST_KNN
            _logger.info(
                f"Training knn model for given parameter list: {param_list}")
            for param in param_list:
                clf_list.append(KNeighborsClassifier(n_neighbors=param))
        elif args.opt == False:
            clf = KNeighborsClassifier(n_neighbors=NO_OPT_KNN_NEIGHBOURS)
            _logger.info(
                f"Training knn model for given parameter: {NO_OPT_KNN_NEIGHBOURS}")

    # Format cross validation set, shuffle placement
    cross_val_set = adhd_set
    cross_val_set.extend(control_set)
    random.shuffle(cross_val_set)

    cross_val_features = []
    cross_val_labels = []
    adhd_features = 0
    control_features = 0

    # Label data using signal class meta information
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
    cv = StratifiedKFold(n_splits=K_FOLD_SPLITS)

    # Perform parameter oprimization loop
    if args.opt == True:
        max_acc = 0
        best_parameter = 0
        acc_list = []

        for param, clf in zip(param_list, clf_list):
            scores = cross_val_score(
                clf, cross_val_features, cross_val_labels, cv=cv)
            _logger.info("Cross-validation scores:", scores)

            mean = sum(scores) / len(scores)
            acc_list.append(mean)
            _logger.info(f"Cross-validation mean: {mean}, parameter: {param}")

            if mean > max_acc:
                best_parameter = param
                max_acc = mean

        _logger.info(80*'=')
        _logger.info(
            f"Max accurancy: {max_acc}, best parameter: {best_parameter}")
        _logger.info(80*'=')

        plt.plot(param_list, acc_list)
        plt.title(args.method)
        plt.xlabel('Wartość parametru')
        plt.ylabel('Skuteczność')
        plt.grid()
        plt.show()

    # Perform single training for given method
    if args.opt == False:
        y_pred = cross_val_predict(
            clf, cross_val_features, cross_val_labels, cv=cv)
        cm = confusion_matrix(cross_val_labels, y_pred)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=[
                    "ADHD", "control"], yticklabels=["ADHD", "control"], cmap=plt.cm.Blues)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f"Macierz pomyłek - {args.method}")
        plt.show()
