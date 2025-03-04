from sklearn.ensemble import RandomForestClassifier
from adult_db_loader import AdultDBLoader
from children_db_loader import ChildrenDBLoader
from feat import extract_all_db_features
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

_logger = setup_logger(__name__)

def load_features_for_model(loader):
    adhd_features = []
    control_features = []
    extract_all_db_features(loader)

    if type(loader) == ChildrenDBLoader:
        for p_name in loader.measurements["ADHD"]:
            adhd_features.append(loader.measurements["ADHD"][p_name].features)

        for p_name in loader.measurements["Control"]:
            control_features.append(loader.measurements["Control"][p_name].features)

    elif type(loader) == AdultDBLoader:
        for p_name in loader.measurements["FADHD"]:
            adhd_features.append(loader.measurements["FADHD"][p_name].features)
        for p_name in loader.measurements["MADHD"]:
            adhd_features.append(loader.measurements["MADHD"][p_name].features)

        for p_name in loader.measurements["FC"]:
            control_features.append(loader.measurements["FC"][p_name].features)
        for p_name in loader.measurements["MC"]:
            control_features.append(loader.measurements["MC"][p_name].features)
    else:
        raise ValueError("Incorrect loader type!")
    
    return adhd_features, control_features


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Model training parser")
    parser.add_argument("--method", type=str, required=True, help="tree, forest, knn and nn are allowed")
    parser.add_argument("--opt", action="store_true", help="Arg tell if script should optimize parameter for model - may took a lot of time")
    args=parser.parse_args()

    # loader = AdultDBLoader()
    loader = ChildrenDBLoader()
    adhd_features, control_features = load_features_for_model(loader)
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



    print(f"ADHD features len: {len(adhd_features)}")
    print(f"Control features len: {len(control_features)}")
    cross_val_labels = len(adhd_features) * [0]
    cross_val_labels.extend(len(control_features) * [1])

    cross_val_set = adhd_features
    cross_val_set.extend(control_features)
    print(f"Cross val set len: {len(cross_val_set)}")

    before = time.time()
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
