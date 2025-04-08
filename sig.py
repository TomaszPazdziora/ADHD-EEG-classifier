from dataclasses import dataclass


@dataclass
class SignalMeta:
    db_name: str  # adult or children
    group: str  # adhd or control
    patient_idx: int
    electrode: str
    task: int = -1


class Signal:
    def __init__(self, sig: list, meta: SignalMeta):
        self.data = sig
        self.features = []
        self.num_of_samples = len(sig)
        self.meta = meta
        if self.meta.db_name == "adult":
            self.fs = 256
        elif self.meta.db_name == "children":
            self.fs = 128
        else:
            raise ValueError("Unrecognized database name!")

    def __str__(self):
        return f"""db_name: {self.meta.db_name}, group: {self.meta.group}, patient_idx: {self.meta.patient_idx}, electrode: {self.meta.electrode}, task: {self.meta.task}"""


class PatientMeasurement:
    def __init__(self, signals):
        self.signals: list[Signal] = signals
        self.features: list = []
        self.check_signals_integrity()

    def check_signals_integrity(self):
        """checks if every signal in object came from the same pacient"""
        ext_fs = self.signals[0].fs
        exp_db_name = self.signals[0].meta.db_name
        ext_group = self.signals[0].meta.group
        ext_patient_idx = self.signals[0].meta.patient_idx

        for sig in self.signals:
            if sig.fs != ext_fs or \
                    sig.meta.db_name != exp_db_name or \
                    sig.meta.group != ext_group or \
                    sig.meta.patient_idx != ext_patient_idx:
                raise ValueError("Signals loaded incorrectly!")
