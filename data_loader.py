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
# NUM_OF_CHANNELS - 2

CONST_IDX = 0  # const value used in database data loading
TASK_DURATION = [30, 20, 20, 45, 15, 30, 30, 30, 30, 30, 45]  # in seconds
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


def get_task_frequency(num_of_samples: int, task_idx: int) -> float:
    return num_of_samples / TASK_DURATION[task_idx]


def get_channel_name(task_idx: int, channel_idx: int) -> str:
    return _TASK_CHANNELS[task_idx][channel_idx]


class Signal:
    def __init__(self, ch1_data: list, ch2_data: list, db_name: str, task_idx: int, patient_idx: int):
        self.ch1_data = ch1_data
        self.ch2_data = ch2_data
        if len(ch1_data) != len(ch2_data):
            raise RuntimeError(
                "Both signals should have the same length!"
            )
        self.num_of_samples = len(ch1_data)
        self.db_name = db_name
        self.task_idx = task_idx
        self.patient_idx = patient_idx
        self.ch1_type = get_channel_name(
            task_idx=task_idx, channel_idx=0
        )
        self.ch2_type = get_channel_name(
            task_idx=task_idx, channel_idx=1
        )
        self.frequency = get_task_frequency(
            num_of_samples=self.num_of_samples, task_idx=task_idx
        )


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


class DataLoader:
    """Loads data from given file. Example input: db_name='FC'"""

    def __init__(self, db_name: str):
        mat = scipy.io.loadmat("data" + os.sep + db_name + ".mat")
        self.db_name = db_name
        self.signal = mat[db_name][CONST_IDX]
        self.tasks = self._load_all_tasks()

    def get_signal(self, task_idx, patient_idx) -> list[float]:
        return Signal(
            ch1_data=self.tasks[task_idx].patients[patient_idx].channels[0].data,
            ch2_data=self.tasks[task_idx].patients[patient_idx].channels[1].data,
            db_name=self.db_name,
            task_idx=task_idx,
            patient_idx=patient_idx
        )

    def get_number_of_samples(self, task_idx, patient_idx, channel_idx) -> int:
        return self.tasks[task_idx].patients[patient_idx].channels[channel_idx].num_of_samples

    def _load_single_electorode_data(self, task_idx: int, patient_idx: int, channel_idx: int) -> list:
        samples = [sample[channel_idx]
                   for sample in self.signal[task_idx][patient_idx]]
        return ElectrodeData(data=samples)

    def _load_all_channel_data(self, task_idx: int, patient_idx: int) -> list:
        return [
            self._load_single_electorode_data(task_idx, patient_idx, 0),
            self._load_single_electorode_data(task_idx, patient_idx, 1)
        ]

    def _load_all_patients(self, task_idx: int) -> list:
        patient = [Patient(self._load_all_channel_data(
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
        f"Reading one singal sample: {data.tasks[0].patients[0].channels[0].data[0]}")
    print(data.get_signal(task_idx=0, patient_idx=0))
    print(data.get_number_of_samples(task_idx=0, patient_idx=0, channel_idx=0))
