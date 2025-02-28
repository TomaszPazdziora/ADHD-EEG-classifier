import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from pathlib import Path
from logger_config import setup_logger
from sig import SignalMeta, Signal, PatientMeasurement
import re

# Participants were 61 children with ADHD and 60 healthy controls (boys and girls, ages 7-12). The ADHD children were diagnosed by an experienced psychiatrist to DSM-IV criteria, and have taken Ritalin for up to 6 months. None of the children in the control group had a history of psychiatric disorders, epilepsy, or any report of high-risk behaviors.

# EEG recording was performed based on 10-20 standard by 19 channels (Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2) at 128 Hz sampling frequency. The A1 and A2 electrodes were the references located on earlobes. 

# Since one of the deficits in ADHD children is visual attention, the EEG recording protocol was based on visual attention tasks. In the task, a set of pictures of cartoon characters was shown to the children and they were asked to count the characters. The number of characters in each image was randomly selected between 5 and 16, and the size of the pictures was large enough to be easily visible and countable by children. To have a continuous stimulus during the signal recording, each image was displayed immediately and uninterrupted after the child’s response. Thus, the duration of EEG recording throughout this cognitive visual task was dependent on the child’s performance (i.e. response speed).

_logger = setup_logger(__name__)

class ChildrenDBLoader:
    # scrap all children database filenames
    ADHD_path = Path.joinpath(Path("children_db"), "ADHD") 
    ADHD_FILES = [file for file in os.listdir(ADHD_path)]
    _logger.info(f'ADHD len: {len(ADHD_FILES)}')

    Control_path = Path.joinpath(Path("children_db"), "Control") 
    CONTROL_FILES = [file for file in os.listdir(Control_path)]
    _logger.info(f'Control len: {len(CONTROL_FILES)}')

    children_electrodes = {
        'Fp1': 1,
        'Fp2': 2,
        'F3' : 3,
        'F4' : 4,
        'C3' : 5,
        'C4' : 6,
        'P3' : 7,
        'P4' : 8, 
        'O1' : 9,
        'O2' : 10,
        'F7' : 11,
        'F8' : 12,
        'T7' : 13,
        'T8' : 14,
        'P7' : 15,
        'P8' : 16,
        'Fz' : 17,
        'Cz' : 18,
        'Pz' : 19
    }

    def __init__(self):
        self.load_all_measurements()

    def get_single_electrode_sig(self, db: dict, db_name: str, electrode: str):
        electrode_id = self.children_electrodes[electrode]
        return [db[db_name][i][electrode_id-1] for i in range(len(db[db_name]))]

    def get_all_patients_signals_for_single_electrode(self, group: str, electrode: str):
        """example: loads all signals for electrode F4 - of all patients in ADHD gorup"""
        signals = []
        if group == "ADHD":
            dir = "ADHD"
            filenames = self.ADHD_FILES
        elif group == "Control":
            dir = "Control"
            filenames = self.CONTROL_FILES
        else:
            raise ValueError("Unrecognized group name. Try ADHD or Control.")

        for name in filenames:
            filepath = "children_db" + os.sep + dir + os.sep + name
            mat = scipy.io.loadmat(filepath)

            meta = SignalMeta(
                db_name="children",
                group=dir,
                patient_idx=int(re.findall(r"\d+", name)[0]),
                electrode=electrode
            )
            signals.append(Signal(
                sig=self.get_single_electrode_sig(mat, name[:-4], electrode),
                meta=meta
                )
            ) 
        return signals
    
    def get_all_signals_for_patient(self, group, db_name):
        signals = []
        if group == "ADHD":
            dir = "ADHD"
        elif group == "Control":
            dir = "Control"
        else:
            raise ValueError("Unrecognized group name. Try ADHD or Control.")
        
        filepath = "children_db" + os.sep + dir + os.sep + db_name 
        mat = scipy.io.loadmat(filepath)

        for electrode, idx in self.children_electrodes.items():
            meta = SignalMeta(
                db_name="children",
                group=dir,
                patient_idx=int(re.findall(r"\d+", db_name)[0]),
                electrode=electrode
            )
            signals.append(Signal(
                sig=self.get_single_electrode_sig(mat, db_name[:-4], electrode),
                meta=meta
                )
            )
        return signals
    
    def load_all_measurements(self):
        self.measurements = {}
        self.measurements["ADHD"] = {}
        self.measurements["Control"] = {}

        for db_name in self.ADHD_FILES:
            signals = self.get_all_signals_for_patient(
                group="ADHD",
                db_name=db_name
            )
            self.measurements["ADHD"][db_name] = PatientMeasurement(signals)
            
        for db_name in self.CONTROL_FILES:
            signals = self.get_all_signals_for_patient(
                group="Control",
                db_name=db_name
            )
            self.measurements["Control"][db_name] = PatientMeasurement(signals)


if __name__ == "__main__":
    data_loader = ChildrenDBLoader()
    print(data_loader.measurements['ADHD']['v204.mat'].signals[0].data)
    print(data_loader.measurements['ADHD']['v204.mat'].signals[1].meta.electrode)
    print(data_loader.measurements['ADHD']['v204.mat'].signals[2].meta.electrode)
    print(data_loader.measurements['ADHD']['v204.mat'].signals[3].meta.electrode)

    i = 0
    for group, patients in data_loader.measurements.items(): #FADHD FC
        for patient, measurement in patients.items():
            for sig in measurement.signals:
                _logger.info(sig)
                i += 1
    _logger.info(f"Loaded {i} signals")