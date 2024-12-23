import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader, Signal, DB_NAMES
import os


def plot_sig_in_time(sig: Signal, save_to_file=False):
    """Plots two signals on one figure (sample idx in x axis). The data originates from a single task related to one patient, collected through two different channels."""
    ax_ch1 = plt.subplot(211)
    ax_ch2 = plt.subplot(212)

    ax_ch1.set_title(
        f"{sig.db_name}: Task: {sig.task_idx}, Patient: {sig.patient_idx}, Channel: {sig.ch1_type}"
    )
    ax_ch2.set_title(
        f"{sig.db_name}: Task: {sig.task_idx}, Patient: {sig.patient_idx}, Channel: {sig.ch2_type}"
    )

    plt.subplots_adjust(hspace=0.5)

    # both channels have the same length
    X = [i for i in range(sig.num_of_samples)]
    ax_ch1.plot(X, sig.ch1_data, linewidth=1)
    ax_ch2.plot(X, sig.ch2_data, color='red', linewidth=1)

    # standarize y axis
    min_ampl = min(min(sig.ch1_data), min(sig.ch2_data))
    max_ampl = max(max(sig.ch1_data), max(sig.ch2_data))

    ax_ch1.set_ylim(min_ampl, max_ampl)
    ax_ch2.set_ylim(min_ampl, max_ampl)

    if save_to_file == True:
        dir_path = f".{os.sep}data{os.sep}plots{os.sep}{sig.db_name}{os.sep}task_{sig.task_idx}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + f"{os.sep}time_patient_{sig.patient_idx}.png"
        plt.savefig(file_path)
    else:
        plt.show()


def plot_fft(sig: Signal, save_to_file=False):
    """Plots two fft graphs on one figure. The data originates from a single task related to one patient, collected through two different channels."""
    # both channels have the same number of samples and the same freq
    N = sig.num_of_samples
    Fs = sig.frequency

    # calculate fft of both channels
    fft_sig = np.fft.fft(sig.ch1_data)
    fft_magnitude_ch1 = np.abs(fft_sig)

    fft_ch2 = np.fft.fft(sig.ch2_data)
    fft_magnitude_ch2 = np.abs(fft_ch2)

    # x axis generation
    frequencies = np.fft.fftfreq(N, d=1/Fs)

    ax_ch1 = plt.subplot(211)
    ax_ch1.plot(frequencies[:N//2], fft_magnitude_ch1[:N//2])
    ax_ch1.set_title(f'FFT - database: {sig.db_name}, task: {sig.task_idx+1}, \
patient: {sig.patient_idx+1}, channel: {sig.ch1_type}')
    ax_ch1.set_xlabel('Frequency (Hz)')
    ax_ch1.set_ylabel('Magnitude')
    # Nyquist frequency constraints
    ax_ch1.set_xlim(0, Fs/2)
    ax_ch1.grid()

    plt.subplots_adjust(hspace=0.5)

    ax_ch2 = plt.subplot(212)
    ax_ch2.plot(frequencies[:N//2], fft_magnitude_ch2[:N//2], color='red')
    ax_ch2.set_title(f'FFT - database: {sig.db_name}, task: {sig.task_idx+1}, \
patient: {sig.patient_idx+1}, channel: {sig.ch2_type}')
    ax_ch2.set_xlabel('Frequency (Hz)')
    ax_ch2.set_ylabel('Magnitude')
    # Nyquist frequency constraints
    ax_ch2.set_xlim(0, Fs/2)
    ax_ch2.grid()

    # standarize y axis
    min_mag = min(min(fft_magnitude_ch1), min(fft_magnitude_ch2))
    max_mag = max(max(fft_magnitude_ch1), max(fft_magnitude_ch2))
    ax_ch1.set_ylim(min_mag, max_mag)
    ax_ch2.set_ylim(min_mag, max_mag)

    if save_to_file == True:
        dir_path = f".{os.sep}data{os.sep}plots{os.sep}{sig.db_name}{os.sep}task_{sig.task_idx}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + f"{os.sep}fft_patient_{sig.patient_idx}.png"
        plt.savefig(file_path)
    else:
        plt.show()


def save_all_plots():
    # saves all plots as png files in dedicated folders
    for db_n in DB_NAMES:
        db = DataLoader(db_n)
        for t in range(len(db.tasks)):
            for p in range(db.tasks[t].num_of_patients):
                sig = db.get_signal(task_idx=t, patient_idx=p)
                plot_fft(sig=sig, save_to_file=True)
                plt.clf()
                plot_sig_in_time(sig=sig, save_to_file=True)
                plt.clf()


def generate_raport():
    # generates db raport with number of patients and number of all signals
    file_path = f'data{os.sep}db_raport.txt'
    raport_content = ''
    adhd_cnt = 0
    control_cnt = 0

    for db_n in DB_NAMES:
        db = DataLoader(db_n)
        for t in range(len(db.tasks)):
            for p in range(db.tasks[t].num_of_patients):
                sig = db.get_signal(task_idx=t, patient_idx=p)
                if sig.frequency != 256:
                    print(sig)
                    break

        num_of_patients = len(db.tasks[0].patients)
        raport_content += f"{db_n}, Number of patients: {num_of_patients}\n"
        if "ADHD" in db_n:
            adhd_cnt += num_of_patients
        else:
            control_cnt += num_of_patients

    all_patients = adhd_cnt + control_cnt
    raport_content += f"----------------------------------\nNumber of patients with ADHD: \
{adhd_cnt}\nNumber of patients in control group: {control_cnt}\n"
    raport_content += f"All patients number {adhd_cnt + control_cnt} - 1\n"
    raport_content += f"EEG signal of subject number 7 of the group of women with ADHD is \
corrupted\n"
    raport_content += f"----------------------------------\nNumber of signals = patients *\
tasks * channels (electrodes) = {all_patients - 1} * 11 * 2 = {(all_patients-1)*11*2}"

    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'w') as file:
        file.write(raport_content)


if __name__ == "__main__":
    # save_all_plots()
    # generate_raport()

    # plot single signal
    fc = DataLoader(DB_NAMES[0])
    sig = fc.get_signal(task_idx=2, patient_idx=4)
    plot_fft(sig=sig, save_to_file=True)
    plt.clf()
    plot_sig_in_time(sig=sig)
