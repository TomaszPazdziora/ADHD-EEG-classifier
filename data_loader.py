import scipy.io
import numpy as np
from enum import Enum
import os
from common_values import *

# 1- The data is a .mat file type
# 2- There are 4 files in .mat format, the first file called FC is related to the data of women in the control group, which
# contains a 1 x 11 cell, which in cells 1 to 11 is related to tasks, and inside each cell is the number of people, and there
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
# cell 8: Eyes closed, channels: O1,F4, duration: 30s
# cell 9: Eyes open, channels: O1,F4, duration: 30s
# cell 10: Eyes closed, channels: F3, F4, duration: 45s
# cell 11: Eyes closed, channels: Fz, F4, duration: 45s

# mat['FC'][people_id][task][electrode]


class ElectrodeData:
    def __init__(self, data: list):
        self.data = data


class Task:
    def __init__(self, electrodes: list[ElectrodeData]):
        self.electrodes = electrodes


class Patient:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks


class DataLoader:
    """Loads data from given file. Example input: db_name='FC'"""

    def __init__(self, db_name: str):
        mat = scipy.io.loadmat("data" + os.sep + db_name + ".mat")
        self.signal = mat[db_name][CONST_IDX]
        self.N = np.linspace(0, NUM_OF_SAMPLES, NUM_OF_SAMPLES)
        self.patients = self._load_all_patients()

    def get_single_signal(self, patient_id, task_id, electrode_id) -> list[float]:
        return self.patients[patient_id].tasks[task_id].electrodes[electrode_id].data

    def _load_single_electorode_data(self, patient_id: int, task_id: int, electrode_id: int) -> list:
        samples = [sample[electrode_id]
                   for sample in self.signal[patient_id][task_id]]
        return ElectrodeData(data=samples)

    def _load_all_electrode_data(self, patient_id: int, task_id: int) -> list:
        return [
            self._load_single_electorode_data(patient_id, task_id, 0),
            self._load_single_electorode_data(patient_id, task_id, 1)
        ]

    def _load_all_task_data(self, patient_id: int) -> list:
        tasks = [Task(self._load_all_electrode_data(
            patient_id, task)) for task in range(len(self.signal[patient_id]))]
        return tasks

    def _load_all_patients(self) -> list:
        patients = [Patient(self._load_all_task_data(patient))
                    for patient in range(len(self.signal))]
        return patients


if __name__ == "__main__":
    data = DataLoader('FC')
    print(
        f"Reading one singal sample: {data.patients[0].tasks[0].electrodes[0].data[0]}")
    x = data.patients[0].tasks[0].electrodes[0]
    print(data.get_single_signal(patient_id=0, task_id=0, electrode_id=0))
