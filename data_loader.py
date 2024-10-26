import scipy.io
import os

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
# cell 8: Eyes closed, channels: O1,F4, duration: 30s
# cell 9: Eyes open, channels: O1,F4, duration: 30s
# cell 10: Eyes closed, channels: F3, F4, duration: 45s
# cell 11: Eyes closed, channels: Fz, F4, duration: 45s

# NUM_OF_TASKS - 11
# NUM_OF_PATIENTS - 13
# NUM_OF_ELECTRODES - 2

CONST_IDX = 0  # const value used in database data loading
TASK_DURATION = [30, 20, 20, 45, 15, 30, 30, 30, 30, 30, 45]  # in seconds
_TASK_ELECTRODES = {
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


def get_task_frequency(num_of_samples: int, task_idx: int) -> float:
    return num_of_samples / TASK_DURATION[task_idx]


def get_electrode_name(task_idx: int, electrode_idx: int) -> str:
    return _TASK_ELECTRODES[task_idx][electrode_idx]


class Signal:
    def __init__(self, samples: list, db_name: str, task_idx: int, patient_idx: int, electrode_idx: int):
        self.samples = samples
        self.num_of_samples = len(samples)
        self.db_name = db_name
        self.task_idx = task_idx
        self.patient_idx = patient_idx
        self.electrode_idx = electrode_idx
        self.electode_type = get_electrode_name(
            task_idx=task_idx, electrode_idx=electrode_idx
        )
        self.frequency = get_task_frequency(
            num_of_samples=self.num_of_samples, task_idx=task_idx
        )


class ElectrodeData:
    def __init__(self, data: list):
        self.data = data
        self.num_of_samples = len(self.data)


class Patient:
    def __init__(self, electrodes: list[ElectrodeData]):
        self.electrodes = electrodes
        self.num_of_electrodes = len(self.electrodes)


class Task:
    def __init__(self, patients: list[Patient]):
        self.patients = patients
        self.num_of_patients = len(self.patients)


class DataLoader:
    """Loads data from given file. Example input: db_name='FC'"""

    def __init__(self, db_name: str):
        mat = scipy.io.loadmat("data" + os.sep + db_name + ".mat")
        self.db_name = db_name
        self.signal = mat[db_name][CONST_IDX]
        self.tasks = self._load_all_tasks()

    def get_single_signal(self, task_idx, patient_idx, electrode_idx) -> list[float]:
        return Signal(
            self.tasks[task_idx].patients[patient_idx].electrodes[electrode_idx].data,
            db_name=self.db_name,
            task_idx=task_idx,
            patient_idx=patient_idx,
            electrode_idx=electrode_idx
        )

    def get_number_of_samples(self, task_idx, patient_idx, electrode_idx) -> int:
        return self.tasks[task_idx].patients[patient_idx].electrodes[electrode_idx].num_of_samples

    def _load_single_electorode_data(self, task_idx: int, patient_idx: int, electrode_idx: int) -> list:
        samples = [sample[electrode_idx]
                   for sample in self.signal[task_idx][patient_idx]]
        return ElectrodeData(data=samples)

    def _load_all_electrode_data(self, task_idx: int, patient_idx: int) -> list:
        return [
            self._load_single_electorode_data(task_idx, patient_idx, 0),
            self._load_single_electorode_data(task_idx, patient_idx, 1)
        ]

    def _load_all_patients(self, task_idx: int) -> list:
        patient = [Patient(self._load_all_electrode_data(
            task_idx, task)) for task in range(len(self.signal[task_idx]))]
        return patient

    def _load_all_tasks(self) -> list:
        tasks = [Task(self._load_all_patients(patient))
                 for patient in range(len(self.signal))]
        return tasks


if __name__ == "__main__":
    # examle data loading
    data = DataLoader('FC')
    print(
        f"Reading one singal sample: {data.tasks[0].patients[0].electrodes[0].data[0]}")
    print(data.get_single_signal(task_idx=0, patient_idx=0, electrode_idx=0))
    print(data.get_number_of_samples(task_idx=0, patient_idx=0, electrode_idx=0))
