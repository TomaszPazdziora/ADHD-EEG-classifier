from scipy.signal import iirnotch, filtfilt
from logger_config import setup_logger
from sig import Signal
import numpy as np


_logger = setup_logger(__name__)

f0 = 50   # freq to be cut
Q = 30
b_128, a_128 = iirnotch(f0, Q, 128)
b_256, a_256 = iirnotch(f0, Q, 256)

# Helpers for iterating over whole db

def iterate_over_whole_db(data_loader, func):
    for _, patients in data_loader.measurements.items():
        for _, measurement in patients.items():
            func(measurement)

def iterate_over_whole_db_signals(data_loader, func):
    for _, patients in data_loader.measurements.items():
        for _, measurement in patients.items():
            func(measurement.signals)

# Loggers

def log_signals_info(signals: list[Signal]):
    for sig in signals:
        _logger.info(sig)

def log_all_db_signals_info(data_loader):
    iterate_over_whole_db_signals(data_loader, log_signals_info)

# Preprocessing functions

def standardize_signals(signals: list[Signal]):
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

def standarize_all_db_signals(data_loader):
    iterate_over_whole_db_signals(data_loader, standardize_signals)

def filter_signals(signals: list[Signal]):
    for sig in signals:
        if sig.fs == 128:
            sig.data = filtfilt(b_128, a_128, sig.data)
        elif sig.fs == 256:
            sig.data = filtfilt(b_256, a_256, sig.data) 
        else:
            raise ValueError("Unsuported sampling freq!")
        
def filter_all_db_signals(data_loader):
    iterate_over_whole_db_signals(data_loader, filter_signals)