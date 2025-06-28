from sklearn.ensemble import RandomForestClassifier
from adult_db_loader import AdultDBLoader
from features import load_features_for_model
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
# import seaborn as sns
from logger_config import setup_logger
import time
from sklearn.neighbors import KNeighborsClassifier
from argparse import RawTextHelpFormatter
import random
import csv
import os

ADHD_LABEL = 0
CONTROL_LABEL = 1

# Training parameters for no optimization option
K_FOLD_SPLITS = 5
NO_OPT_MPL_LAYERS = 6
NO_OPT_KNN_NEIGHBOURS = 9
NO_OPT_FOREST_TREES = 30

# Training parameters for --opt arg
OPT_PARAM_LIST_MPL = [i for i in range(1, 40, 1)]
OPT_PARAM_LIST_KNN = [i for i in range(1, 45)]
OPT_PARAM_LIST_FOREST = [i for i in range(1, 200, 5)]

arg_names = {"knn": "knn", "forest": "las losowy", "mpl": "mpl"}
args_param = {"knn": "ilość sąsiadów",
              "forest": "ilość drzew", "mpl": "liczba warstw ukrytych"}
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
            best_parameter = NO_OPT_MPL_LAYERS
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
            best_parameter = NO_OPT_FOREST_TREES
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
            best_parameter = NO_OPT_KNN_NEIGHBOURS
            clf = KNeighborsClassifier(n_neighbors=NO_OPT_KNN_NEIGHBOURS)
            _logger.info(
                f"Training knn model for given parameter: {NO_OPT_KNN_NEIGHBOURS}")

    acc_task_list = []  # lista najlepszych dokładności dla tasków
    best_param_list = []
    # lista wszystkich dokładności dla modelu - wykorzystywane do zrobienia zbiorowego wykresu
    acc_plot_task_list = []

    if os.path.exists("acc.csv"):
        pass
    else:
        with open(f'acc.csv', mode='w') as acc_file:
            acc_writer = csv.writer(
                acc_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            acc_writer.writerow(
                ["Model", "Numer zadania", "Skuteczność", "Najlepszy parametr uczenia"])

    for i in range(11):
        task_list = [i]
        _logger.info("=" * 80)
        _logger.info(f"Chosen method: {args.method}")
        _logger.info(f"Used task list: {task_list}")
        # Load signals and extract features
        loader = AdultDBLoader(active_tasks_list=task_list)
        adhd_set, control_set = load_features_for_model(
            loader=loader, features_type="cwt")

        # Format cross validation set, shuffle placement
        cross_val_set = adhd_set
        cross_val_set.extend(control_set)
        random.seed(155)
        random.shuffle(cross_val_set)

        for x in cross_val_set:
            _logger.info(x.signals[0])
            _logger.info(x.signals[1])

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
                _logger.info(f"Cross-validation scores: {scores}")

                mean = sum(scores) / len(scores)
                acc_list.append(mean)
                _logger.info(
                    f"Cross-validation mean: {mean}, parameter: {param}")

                if mean > max_acc:
                    best_parameter = param
                    max_acc = mean

            _logger.info(80*'=')
            _logger.info(
                f"Max accurancy: {max_acc}, best parameter: {best_parameter}")
            _logger.info(80*'=')

            path = f"plots{os.sep}model_verification{os.sep}{args.method}{os.sep}"
            if not os.path.exists(path):
                os.makedirs(path)

            task_plot_path = path + f"task_{i+1}"
            plt.plot(param_list, acc_list)
            plt.title(
                f"{arg_names[args.method]} - dokładność w funkcji parametru, zadanie: {i+1}")
            plt.xlabel(f'Wartość parametru ({args_param[args.method]})')
            plt.ylabel('Dokładność [-]')
            plt.grid()
            plt.savefig(task_plot_path)
            plt.clf()
            acc_plot_task_list.append(acc_list)

            best_param_list.append(best_parameter)
            best_acc = round(max_acc*100, 4)
            acc_task_list.append(best_acc)

        # Perform single training for given method
        if args.opt == False:
            scores = cross_val_score(
                clf, cross_val_features, cross_val_labels, cv=cv)
            _logger.info(f"Cross-validation scores: {scores}")

            mean = sum(scores) / len(scores)
            _logger.info(
                f"Cross-validation mean: {mean}, parameter: {best_parameter}")

            best_acc = round(mean*100, 4)

        with open(f'acc.csv', mode='a') as acc_file:
            acc_writer = csv.writer(
                acc_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            acc_writer.writerow([args.method, i+1, best_acc, best_parameter])

    if args.opt == True:
        _logger.info(f"best parameters: {best_param_list}")
        _logger.info(f"best parameters len: {len(best_param_list)}")
        _logger.info(f"task acc: {acc_task_list}")
        _logger.info(f"task acc len: {len(acc_task_list)}")

        group_file = path + f"group"
        colors = plt.get_cmap('tab20').colors  # 20 wyraźnych kolorów

        for i, task in enumerate(acc_plot_task_list):
            plt.plot(param_list, task,
                     label=f'Zadanie {i+1}', color=colors[i % len(colors)])

        plt.title(
            f"{arg_names[args.method]} - Dokładność w funkcji parametru ({args_param[args.method]})")
        plt.xlabel(f'Wartość parametru ({args_param[args.method]})')
        plt.ylabel('Dokładność [-]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(path + f"group", bbox_inches='tight')
        plt.clf()
