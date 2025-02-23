from scipy.signal import iirnotch, filtfilt
from logger_config import setup_logger
from sig import Signal
import numpy as np


_logger = setup_logger(__name__)

f0 = 50   # freq to be cut
Q = 30
b_128, a_128 = iirnotch(f0, Q, 128)
b_256, a_256 = iirnotch(f0, Q, 256)


def standardize_signals(signals: list[Signal]) -> list[Signal]:
    for sig in signals:
        mean = np.mean(sig.data)
        std_dev = np.std(sig.data)
        
        if std_dev == 0:
            _logger.warning(f"Error - std_dev == 0 - DB: {sig.meta.db_name},"
                         f"Group: {sig.meta.group} "
                         f"Patient: {sig.meta.patient_idx}, "
                         f"Electrode: {sig.meta.electrode}, "
                         f"Task: {sig.meta.task}")
            sig.data = np.zeros_like(sig.data)
        sig.data = (sig.data - mean) / std_dev


def filter_signals(signals: list[Signal]):
    for sig in signals:
        if sig.fs == 128:
            sig.data = filtfilt(b_128, a_128, sig.data)
        elif sig.fs == 256:
            sig.data = filtfilt(b_256, a_256, sig.data) 
        else:
            raise ValueError("Unsuported sampling freq!")