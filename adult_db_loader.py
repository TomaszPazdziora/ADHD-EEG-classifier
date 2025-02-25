import scipy.io
import os
import numpy as np
from logger_config import setup_logger
from sig import Signal, SignalMeta, PatientMeasurement

# 1- The data is a .mat file type
# 2- There are 4 files in .mat format, the first file called FC is related to the data of women in the control group, which
# contains a 1 x 11 cell, which in cells 1 to 11 is related to task, and inside each cell is the number of people, and there
# are samples of signal and EEG channel. The second file named MC is related to the data of men in the control group. The third
# file named FADHD is related to the data of women in the ADHD group. And the fourth file named MADHD is related to the data of
# men in the ADHD group.
# For example, each cell is as follows:
# 13*7680*2
# 13 is the number of subjects, for example, 7680 is the number of signal samples in 30 seconds, and 2 is the number of channels.
# Note: the EEG signal of subject number 7 of the group of women with ADHD is corrupted.

# cell 1:  Eyes open baseline, channels:  Cz, F4, duration: 30s
# cell 2: Eyes closed, channels: Cz, F4, duration : 20s
# cell 3: Eyes open, channels: Cz, F4, duration: 20s
# cell 4: Cognitive Challenge, channels: Cz, F4, duration: 45s
# cell 5: Pre-Omni harmonic baseline, channels: Cz, F4, duration: 15s
# cell 6: Omni harmonic assessment, channels: Cz, F4, duration: 30s
# cell 7: Eyes open baseline, channels: O1,F4, duration: 30s
# cell 8: Eyes closed, channels: O1,F4, duration: 30s / 20s
# cell 9: Eyes open, channels: O1,F4, duration: 30s / 20s
# cell 10: Eyes closed, channels: F3, F4, duration: 45s
# cell 11: Eyes closed, channels: Fz, F4, duration: 45s

# NUM_OF_TASKS - 11
# NUM_OF_PATIENTS - depends
# NUM_OF_CHANNELS - 2

_logger = setup_logger(__name__)
NUM_OF_TASKS = 11
TASK_DURATION = [30, 20, 20, 45, 15, 30, 30, 20, 20, 45, 45]  # in seconds
DB_NAMES = ["FADHD", "FC", "MADHD", "MC"]
_TASK_CHANNELS = {
    0: ["Cz", "F4"],
    1: ["Cz", "F4"],
    2: ["Cz", "F4"],
    3: ["Cz", "F4"],
    4: ["Cz", "F4"],
    5: ["Cz", "F4"],
    6: ["O1", "F4"],
    7: ["O1", "F4"],
    8: ["O1", "F4"],
    9: ["F3", "F4"],
    10: ["Fz", "F4"]
}

def get_channel_name(task_idx: int, channel_idx: int) -> str:
    return _TASK_CHANNELS[task_idx][channel_idx]


class ElectrodeData:
    def __init__(self, data: list):
        self.data = data
        self.num_of_samples = len(self.data)


class Patient:
    def __init__(self, channels: list[ElectrodeData]):
        self.channels = channels
        self.num_of_channels = len(self.channels)


class Task:
    def __init__(self, patients: list[Patient]):
        self.patients = patients
        self.num_of_patients = len(self.patients)


class AdultDBLoader:
    """Loads data from given file. Example input: db_name='FC'"""

    def __init__(self):
        self.load_all_measurements()

    def _load_single_electorode_data(self, task_idx: int, patient_idx: int, channel_idx: int, raw_data: list) -> list:
        samples = [sample[channel_idx]
                   for sample in raw_data[task_idx][patient_idx]]
        return ElectrodeData(data=samples)

    def _load_all_channel_data(self, task_idx: int, patient_idx: int, raw_data: list) -> list:
        return [
            self._load_single_electorode_data(task_idx, patient_idx, 0, raw_data),
            self._load_single_electorode_data(task_idx, patient_idx, 1, raw_data)
        ]

    def _load_all_patients(self, task_idx: int, raw_data: list) -> list:
        patient = [Patient(self._load_all_channel_data(
            task_idx, task, raw_data)) for task in range(len(raw_data[task_idx]))]
        return patient

    def _load_all_tasks(self, raw_data: list) -> list:
        tasks = [Task(self._load_all_patients(patient, raw_data))
                 for patient in range(len(raw_data))]
        return tasks
    
    def get_all_group_signals(self, group):
        signals = []
        mat = scipy.io.loadmat("adult_db" + os.sep + group + ".mat")
        db = mat[group][0]
        tasks = self._load_all_tasks(db)
        for task_idx, task in enumerate(tasks):
            for patient_idx, patient in enumerate(task.patients):
                for electrode_idx, electrode in enumerate(patient.channels):
                    meta = SignalMeta(
                        db_name="adult",
                        group=group,
                        patient_idx=patient_idx,
                        electrode=get_channel_name(task_idx, electrode_idx),
                        task=task_idx
                    )
                    signals.append(Signal(
                            sig=electrode.data,
                            meta=meta
                        )
                    )
        return signals

    def sort_signals_by_patient_idx(self, signals: list[Signal]):
        return sorted(signals, key=lambda sig: sig.meta.patient_idx)

    def load_all_measurements(self):
        self.measurements = {}
        sig_dict = {}
        for group in DB_NAMES:
            signals = self.get_all_group_signals(group=group)
            signals = self.sort_signals_by_patient_idx(signals)
            sig_dict[group] = signals

        for gr_name, sig_list in sig_dict.items():
            if len(sig_list) % NUM_OF_TASKS != 0:
                raise ValueError
            for i in range(0, len(sig_list), NUM_OF_TASKS*2):
                pat_str = f"patient_{sig_list[i].meta.patient_idx}"
                if gr_name not in self.measurements:
                    self.measurements[gr_name] = {}
                self.measurements[gr_name][pat_str] = PatientMeasurement(sig_list[i:i+NUM_OF_TASKS])


if __name__ == "__main__":
    data = AdultDBLoader()
    print(data.measurements['FC']['patient_0'].signals[0].data)
