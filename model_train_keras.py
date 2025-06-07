from adult_db_loader import AdultDBLoader
from features import extract_all_db_features,  load_all_raw_db_signals_to_measurement_features
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from logger_config import setup_logger
from tensorflow import keras
from keras import layers
from keras import Sequential
from sklearn.model_selection import train_test_split
import copy
import numpy as np
from scipy.signal import spectrogram

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
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_features,
        all_labels,
        random_state=42,
        test_size=0.3,
        shuffle=True
    )

    TASK = 1
    ELECTRODE = 0
    X_train = [one_measure[TASK*2+ELECTRODE] for one_measure in X_train]
    X_test = [one_measure[TASK*2+ELECTRODE] for one_measure in X_test]

    X_spect_train = []
    X_spect_test = []

    for x in X_train:
        f, t, Sxx = spectrogram(x, fs=256, nperseg=256, noverlap=128)
        Sxx = np.log1p(Sxx)
        S_pad = np.zeros((128, 128))
        h, w = min(len(Sxx), 128), min(len(Sxx[0]), 128)
        S_pad[:h, :w] = Sxx[:h, :w]
        X_padded = S_pad[..., np.newaxis]
        X_spect_train.append(X_padded)

    for x in X_test:
        f, t, Sxx = spectrogram(x, fs=256, nperseg=256, noverlap=128)
        Sxx = np.log1p(Sxx)
        S_pad = np.zeros((128, 128))
        h, w = min(len(Sxx), 128), min(len(Sxx[0]), 128)
        S_pad[:h, :w] = Sxx[:h, :w]
        X_padded = S_pad[..., np.newaxis]
        X_spect_test.append(X_padded)

    X_train = np.array(X_spect_train)
    # len(X_train)
    # 55
    # len(X_train[0])
    # 22
    X_test = np.array(X_spect_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # zamiast 22 wziąć tylko 1 pomiar (1 task i 1 elektroda) zamiast 11 tasków i 2 elektrod
    # wyliczyć spektrogram
    # podać do modelu
    # sprawdzić skuteczność dla każdego obrazu

    _logger.info(f"Train x len: {len(X_train)}")
    _logger.info(f"Test x len: {len(X_test)}")
    _logger.info(f"Train y len: {len(Y_train)}")
    _logger.info(f"Test y len: {len(Y_test)}")

    for ep in range(10, 20, 2):
        for b in range(2, 6, 1):
            for kf in range(2, 6, 1):
                for ks in range(2, 6, 1):
                    # for pooling size
                    model = keras.Sequential([
                        layers.Conv2D(32, (kf, kf), activation='relu',
                                      input_shape=(128, 128, 1)),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(64, (ks, ks), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Flatten(),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(1, activation='sigmoid')
                    ])
                    model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    history = model.fit(
                        X_train, Y_train, epochs=ep, batch_size=b)
                    test_loss, test_acc = model.evaluate(X_test, Y_test)
                    _logger.info(
                        f"Dokładność modelu: {test_acc:.4f}, ep: {ep}, batch: {b}, kf {kf}, kl {ks}")

    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('MLP Model Accuracy vs. Epochs')
    # plt.show()
