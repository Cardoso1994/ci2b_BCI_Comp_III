"""Learning ic2b preprocessing methodology."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

# global variables
DATA_DIR = os.path.join("..", "BCI_Comp_III_Wads_2004")
SUBJECT = "B"  # "A" or "B"
SET = "Train"  # "Train" or "Test"
DATA_FILE = f"Subject_{SUBJECT}_{SET}.mat"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# fmt: off
CHANNELS_NAMES = [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1",
        "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
        "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7",
        "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7",
        "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2",
        "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
        "Iz"
]
# fmt: on

CHANNEL_TYPES = ["eeg" for _ in range(64)]
SAMPLING_FREQ = 240
EVENT_DICT = {"P300": 1, "non-P300": 0}
info = mne.create_info(
    ch_names=CHANNELS_NAMES, sfreq=SAMPLING_FREQ, ch_types="eeg"
)
info.set_montage("standard_1020")


def main() -> None:
    """Preprocessing.

    1. Load data
    2. Bandpass filter [1-40] Hz
    3. ASR (artifact removal)
    4. Removal of artifacted channels
    5. Interpolation of eliminated channels
    6. Average Reference
    7. AAR (EMG removal)
    8. Epoch creation
    9. ICA

    From Dataset PDF:
        - Data bandpass filtered at [0.1-60] Hz
        - Digitized at 240 Hz
        - Epoch:
            - 2.5 s of no stimulus (matrix displayed without flickering)
            - Then, 15 blocks of:
                - Rows and columns intensify for 100ms
                  (12 stimuli, 6 rows and 6 columns)
                - 75 ms between intensifications
        - Variables:
            - Signal: Actual eeg signal (epochs, time_samples, channels)
            - target_char: only for training. Single string, each char in str
              is the target char of each epoch
            - flashing: 1 when a row/col is intensified, 0 otherwise
              (epochs, time_samples)
            - stimulus_code: 0 if no intensification
                             1-6 columns (left to right)
                             7-12 rows (top to bottom)
            - stimulus_type: 0 if no intensification or intensification doesn't
              contain desired character; 1 if intensification contains desired
              character
    """
    data = scipy.io.loadmat(DATA_PATH)
    signal = data["Signal"]
    if SET == "Train":
        target_char = data["TargetChar"][0]

    # signal starts with stimulus onset even at the start of the epochs, as per
    # `flashing`
    flashing = data["Flashing"]
    stimulus_code = data["StimulusCode"]
    stimulus_type = data["StimulusType"]

    print(signal.shape)
    signal = np.moveaxis(signal, 1, 2)
    print(signal.shape)
    epochs = mne.EpochsArray(signal, info)
    epochs.plot()
    plt.show()

    bad_epochs = []
    for epoch in range(signal.shape[0]):
        if epoch == 1:
            break

        for stim in range(15 * 12):
            print(f"\n\nepoch: {epoch}")
            print("Stimulus")
            print(
                flashing[epoch, :24],
                stimulus_code[epoch, :24],
                stimulus_type[epoch, :24],
            )
            print("non-Stimulus")
            print(
                flashing[epoch, 24:42],
                stimulus_code[epoch, 24:42],
                stimulus_code[epoch, 24:42],
            )
            if flashing[epoch, 0] != 1 or flashing[epoch, 24] != 0:
                bad_epochs.append(epoch)

    plt.plot(signal[0, -600:, 0])
    plt.show()


if __name__ == "__main__":
    main()
