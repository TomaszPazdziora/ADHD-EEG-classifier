from dataclasses import dataclass

@dataclass
class SignalMeta:
    db_name: str # adult or children
    group: str # adhd or control
    patient_idx: int 
    electrode: str
    task: int = -1

class Signal:
    def __init__(self, sig: list, meta: SignalMeta):
        self.data = sig
        self.num_of_samples = len(sig)
        self.meta = meta
        if self.meta.db_name == "adult":
            self.fs = 256
        elif self.meta.db_name == "children":
            self.fs = 128
        else: 
            raise ValueError("Unrecognized database name!")
