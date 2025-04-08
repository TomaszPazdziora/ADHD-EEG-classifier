from adult_db_loader import AdultDBLoader
from features import extract_all_db_features,  load_all_raw_db_signals_to_measurement_features
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from logger_config import setup_logger
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import copy
import numpy as np

_logger = setup_logger(__name__)

def load_features_for_model(loader):
    adhd_features = []
    control_features = []
    # extract_all_db_features(loader)
    load_all_raw_db_signals_to_measurement_features(loader)

    if type(loader) == AdultDBLoader:
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
    loader = AdultDBLoader()
    adhd_features, control_features = load_features_for_model(loader)
    clf_list = []

    all_features = copy.deepcopy(adhd_features)
    all_features.extend(copy.deepcopy(control_features))

    all_labels = [0] * len(adhd_features)
    all_labels.extend([1] * len(control_features))
    X_train, X_test, Y_train, Y_test = train_test_split(all_features, all_labels, random_state=42, test_size=0.3, shuffle=True)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    Y_train  = np.array(Y_train)
    Y_test  = np.array(Y_test)

    _logger.info(f"Train x len: {len(X_train)}")
    _logger.info(f"Test x len: {len(X_test)}")
    _logger.info(f"Train y len: {len(Y_train)}")
    _logger.info(f"Test y len: {len(Y_test)}")

    model = keras.Sequential([
        layers.Conv2D(64, kernel_size=4, activation='relu', input_shape=(22, 3840, 1)),
        layers.Dropout(0.2),
        layers.Conv2D(32, kernel_size=4, activation='relu'),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    

    # Kompilacja
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=5, batch_size=5)   

    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"Dokładność modelu: {test_acc:.4f}")

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('MLP Model Accuracy vs. Epochs')
    plt.show()
