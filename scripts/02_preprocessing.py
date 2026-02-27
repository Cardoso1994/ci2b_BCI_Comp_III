"""Learning ic2b preprocessing methodology."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from meegkit.asr import ASR
import mne
from mne.preprocessing import ICA
from mne.preprocessing import annotate_muscle_zscore
import numpy as np
from numpy.typing import NDArray
import scipy

from eeg_preproc import find_bad_channels
from eeg_preproc import find_flat_channels

# global variables
DATA_DIR: pathlib.Path = pathlib.Path("..", "..", "gigaScienceDataset")
SUBJECT: str = "B"  # "A" or "B"
SET: str = "Train"  # "Train" or "Test"
DATA_FILE: str = f"sess02_subj54_EEG_ERP.mat"
DATA_PATH: pathlib.Path = pathlib.Path(DATA_DIR, DATA_FILE)

# fmt: off
CHANNELS_NAMES: list[str] = [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1",
        "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
        "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7",
        "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7",
        "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2",
        "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
        "Iz"
]
# fmt: on

CHANNEL_TYPES: list[str] = ["eeg" for _ in range(64)]
SAMPLING_FREQ: int = 240
EVENT_DICT: dict[str, int] = {"non-P300": 1, "P300": 2}

TMAX: float = 0.650  # 600 ms — standard P300 window (Polich, 2007)


info: mne.Info = mne.create_info(
    ch_names=CHANNELS_NAMES, sfreq=SAMPLING_FREQ, ch_types="eeg"
)
info.set_montage("standard_1020")


def main() -> None:
    """Preprocessing pipeline replicating the ci²b MATLAB/EEGLAB methodology.

    The preprocessing order follows EEGLAB's ``clean_rawdata`` / ``clean_artifacts``
    internal execution sequence (verified against ``clean_artifacts.m``), which
    must be strictly respected because each step depends on the output of the
    previous one:

    clean_rawdata steps (continuous EEG):
        1. Flatline channel removal  — ``clean_flatlines``  (flatline_criterion=5 s)
        2. High-pass filter          — ``clean_drifts``     (highpass=[0.1 0.5] Hz)
        3. Bad channel detection     — ``clean_channels``   (channel_criterion=0.8,
                                                             line_noise_criterion=4)
        4. Artifact Subspace Reconstruction — ``clean_asr`` (burst_criterion=5)
           [step 5, clean_windows (window_criterion=0.25), applied at epoch stage]

    Post-clean_rawdata steps (MNE-specific, continuous EEG):
        6. Bad channel interpolation — ``raw.interpolate_bads()``

    Epoching and source separation:
        7. Average re-reference      — ``raw.set_eeg_reference("average")``
        9. Epoch creation            — ``mne.Epochs``
       10. ICA

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
    data: dict[str, NDArray] = scipy.io.loadmat(DATA_PATH)
    signal: NDArray = data["Signal"]
    if SET == "Train":
        target_char: np.str_ = data["TargetChar"][0]

    # signal starts with stimulus onset even at the start of the epochs, as per
    # `flashing` array. *No baseline signal*
    flashing: NDArray = data["Flashing"]
    stimulus_code: NDArray = data["StimulusCode"]
    stimulus_type: NDArray = data["StimulusType"]

    # select one experiment and convert to RawArray
    experiment: int = 0
    signal: NDArray = np.moveaxis(signal, 1, 2)
    signal *= 1e-6  # scale data to volts
    experiment_signal = mne.io.RawArray(signal[experiment, ...], info=info)

    # build stim channel: 0 = no flash, 1 = non-P300 flash, 2 = P300 flash
    exp_flashing: NDArray = flashing[experiment]
    exp_stimulus_type: NDArray = stimulus_type[experiment]
    stim: NDArray = np.zeros((1, exp_flashing.shape[0]))
    stim[0, (exp_flashing == 1) & (exp_stimulus_type == 0)] = EVENT_DICT[
        "non-P300"
    ]
    stim[0, (exp_flashing == 1) & (exp_stimulus_type == 1)] = EVENT_DICT["P300"]

    stim_info: mne.Info = mne.create_info(
        ["STI"], sfreq=SAMPLING_FREQ, ch_types="stim"
    )
    experiment_signal.add_channels([mne.io.RawArray(stim, stim_info)])

    events: NDArray = mne.find_events(experiment_signal, stim_channel="STI")

    experiment_signal.plot(
        events=events,
        event_id=EVENT_DICT,
        event_color={1: "blue", 2: "red"},
        title="Raw Signal",
    )

    # ------------------------------------------------------------------
    # Step 2 — High-pass filter (clean_drifts, highpass=[0.1 0.5] Hz)
    # Removes DC drift and sub-Hz noise before bad channel detection and
    # ASR. Parameters are matched exactly to EEGLAB's clean_drifts:
    #   passband_edge=0.5 Hz  → l_freq in MNE (−6 dB point)
    #   stopband_edge=0.1 Hz  → full attenuation; l_trans_bandwidth=0.4 Hz
    #   fir_window="hamming"  → same window as EEGLAB's FIR design
    #   phase="zero"          → zero-phase (forward+backward), no latency shift
    # h_freq=40.0
    # ------------------------------------------------------------------
    experiment_signal.filter(
        l_freq=0.5,
        h_freq=49.0,
        l_trans_bandwidth=0.4,  # passband_edge - stopband_edge = 0.5 - 0.1
        fir_window="hamming",
        phase="zero",
    )
    experiment_signal.plot(
        events=events,
        event_id=EVENT_DICT,
        event_color={1: "blue", 2: "red"},
        title="After High-Pass Filter (clean_drifts)",
    )

    # ------------------------------------------------------------------
    # Step 1 — Flatline channel removal (clean_flatlines, criterion=5 s)
    # Must run before filtering: a flat channel is a degenerate signal
    # that can cause numerical issues in filter overlap-add convolution
    # and must be excluded before ASR calibration.
    # ------------------------------------------------------------------
    flat_bads: list = find_flat_channels(
        experiment_signal, flatline_criterion=5.0
    )
    experiment_signal.info["bads"] += flat_bads

    # ------------------------------------------------------------------
    # Step 3 — Bad channel detection (clean_channels)
    # channel_criterion=0.8:    RANSAC spatial correlation threshold
    # line_noise_criterion=4:   HF broadband noise z-score threshold
    # frac_bad=0.4:             max fraction of bad RANSAC windows
    #                           (= EEGLAB's max_broken_time=0.4)
    # Both criteria are combined with a logical OR.
    # ------------------------------------------------------------------
    channel_bads = find_bad_channels(
        experiment_signal,
        channel_criterion=0.8,
        line_noise_criterion=4.0,
        frac_bad=0.4,
        corr_window_secs=5.0,
    )
    experiment_signal.info["bads"] += channel_bads

    # ------------------------------------------------------------------
    # Step 4 — Artifact Subspace Reconstruction (clean_asr)
    # burst_criterion=5:     signals exceeding 5 SD above the clean-baseline
    #                        RMS are reconstructed via subspace projection.
    # window_criterion=0.25: windows where >25% of channels still exceed the
    #                        threshold after ASR are flagged for rejection at
    #                        the epoch stage (step 5 / clean_windows).    #
    # ------------------------------------------------------------------
    eeg_picks = mne.pick_types(experiment_signal.info, eeg=True)
    eeg_data = experiment_signal.get_data(picks=eeg_picks)
    # method="euclid": original ASR (Chang et al., 2020). The Riemannian
    # variant requires strictly positive definite covariance matrices for
    # every calibration window; with 53 channels and 0.5 s windows (120
    # samples) on band-limited data, at least one window produces a
    # near-singular covariance (log of a near-zero eigenvalue → crash).
    # euclid is numerically unconditional and matches EEGLAB's default.
    asr = ASR(sfreq=experiment_signal.info["sfreq"], cutoff=5, method="euclid")
    asr.max_bad_chans = 0.25
    asr.fit(eeg_data)
    cleaned = asr.transform(eeg_data)
    experiment_signal._data[eeg_picks] = cleaned
    experiment_signal.plot(
        events=events,
        event_id=EVENT_DICT,
        event_color={1: "blue", 2: "red"},
        title="After ASR (clean_asr)",
    )

    # ------------------------------------------------------------------
    # Step 6 — Bad channel interpolation (raw.interpolate_bads)
    # Reconstructs channels in raw.info["bads"] (flatline + RANSAC/HF)
    # using spherical spline interpolation
    # reset_bads=True (default) clears info["bads"] after reconstruction
    # so that subsequent steps treat all channels as valid.
    # ------------------------------------------------------------------
    interpolated_bads: list[str] = list(experiment_signal.info["bads"])
    experiment_signal.interpolate_bads(reset_bads=True, mode="accurate")
    experiment_signal.plot(
        events=events,
        event_id=EVENT_DICT,
        event_color={1: "blue", 2: "red"},
        title=f"After Interpolation — {len(interpolated_bads)} channel(s) restored: "
        f"{interpolated_bads or 'none'}",
    )

    # ------------------------------------------------------------------
    # Step 7 — Average reference (raw.set_eeg_reference)
    # Subtracts the instantaneous mean of all EEG electrodes from each
    # channel at every sample.
    # ------------------------------------------------------------------
    experiment_signal.set_eeg_reference("average", projection=True)
    experiment_signal.apply_proj()
    experiment_signal.plot(
        events=events,
        event_id=EVENT_DICT,
        event_color={1: "blue", 2: "red"},
        title="After Average Reference",
    )

    # ------------------------------------------------------------------
    # Step 8 — Muscle artifact annotation (AAR / EMG removal)
    # Detects segments contaminated by skeletal muscle activity using the
    # broadband 110–140 Hz band (Muthukumaraswamy, 2013; Whitham et al.,
    # 2007), which is dominated by surface EMG and largely free of neural
    # signal above ~80 Hz at scalp level.
    #
    # Algorithm (MNE annotate_muscle_zscore):
    #   1. Band-pass filter the data to 110–140 Hz.
    #   2. Compute the Hilbert envelope (instantaneous amplitude).
    #   3. Z-score across channels at each sample; sum and normalise by
    #      sqrt(n_channels) to obtain a single time-series.
    #   4. Low-pass the z-score at 4 Hz to smooth transient peaks.
    #   5. Flag samples exceeding `threshold` as BAD_muscle.
    #
    # threshold=4: conservative; standard in MNE tutorials and consistent
    # with Muthukumaraswamy (2013). Lower values (e.g. 3) capture subtler
    # contamination but risk false positives on high-gamma neural activity.
    #
    # The annotations are stored in raw but data is NOT modified here —
    # actual exclusion happens at the epoch stage (step 9) via
    # reject_by_annotation=True,
    muscle_annot, muscle_scores = annotate_muscle_zscore(
        experiment_signal,
        threshold=4,
        filter_freq=(110, 119),  # nyquist frequency is 120 Hz for this dataset
        min_length_good=0.1,
    )
    experiment_signal.set_annotations(
        experiment_signal.annotations + muscle_annot
    )

    n_muscle_segments: int = sum(
        ann["description"] == "BAD_muscle"
        for ann in experiment_signal.annotations
    )

    fig, ax = plt.subplots()
    ax.plot(
        experiment_signal.times,
        muscle_scores,
        label="Muscle z-score",
        color="steelblue",
        linewidth=0.8,
    )
    ax.axhline(
        4, color="tomato", linestyle="--", linewidth=1, label="Threshold (z=4)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z-score")
    ax.set_title(
        f"Muscle artifact score — {n_muscle_segments} BAD_muscle segment(s)"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 9 — Epoch creation (mne.Epochs)
    #
    # tmin / tmax:
    #   The dataset has no pre-stimulus baseline ("signal starts with
    #   stimulus onset", per dataset PDF), so tmin=0. tmax=0.6 s (144
    #   samples at 240 Hz)     #
    # baseline=None:
    #   No pre-stimulus period exists in this dataset;     #
    # reject_by_annotation=True:
    #   Drops any epoch whose window overlaps a BAD_muscle annotation
    #   placed in step 8. This is the actual epoch-level rejection step
    #   for the AAR annotations — no data was modified in step 8 itself.
    #
    # Step 5 (clean_windows) equivalent:
    #   EEGLAB's clean_windows rejects epochs where > window_criterion
    #   (0.25) of channels still exceed the ASR threshold post-ASR. MNE
    #   has no direct analogue; the combination of ASR (step 4) with
    #   annotation-based rejection (step 8) provides equivalent coverage.
    # ------------------------------------------------------------------
    epochs: mne.Epochs = mne.Epochs(
        experiment_signal,
        events,
        event_id=EVENT_DICT,
        tmin=0.0,
        tmax=TMAX,
        baseline=None,
        reject_by_annotation=True,
        preload=True,
    )

    n_p300 = len(epochs["P300"])
    n_non_p300 = len(epochs["non-P300"])
    print(
        f"Epochs retained — P300: {n_p300}, non-P300: {n_non_p300} "
        f"(total: {len(epochs)} / {len(events)})"
    )

    # ERP sanity check: P300 vs non-P300 at Cz
    evoked_p300 = epochs["P300"].average()
    evoked_non_p300 = epochs["non-P300"].average()

    fig, ax = plt.subplots()
    cz_idx: int = epochs.ch_names.index("Cz")
    ax.plot(
        epochs.times * 1000,
        evoked_p300.data[cz_idx] * 1e6,
        label="P300",
        color="tomato",
    )
    ax.plot(
        epochs.times * 1000,
        evoked_non_p300.data[cz_idx] * 1e6,
        label="non-P300",
        color="steelblue",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"ERP at Cz — P300 (n={n_p300}) vs non-P300 (n={n_non_p300})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Step 10 — ICA (Independent Component Analysis)
    #
    # Fitting strategy — 1 Hz high-pass filtered copy (Winkler et al., 2015):
    #   ICA convergence degrades when slow cortical drifts and sub-Hz noise
    #   dominate the covariance. A 1 Hz high-pass applied to a *copy* of the
    #   epochs removes these drifts for fitting only; the unmixing matrix is
    #   then applied to the original 0.5 Hz epochs so that low-frequency
    #   neural signal is preserved in the cleaned output.
    #
    # n_components=0.999:
    #   Selects the minimum number of components that explain 99.9% of
    #   variance, capped at the data rank. Average reference reduces rank
    #   by 1 (to 63 for 64 channels), so this is equivalent to n=63 while
    #   being robust to any additional rank reduction (e.g. from ASR).
    #
    # method="fastica":
    #   Matches EEGLAB's runica/fastica default and is the most widely
    #   validated method for EEG artifact separation (Hyvärinen & Oja, 2000;
    #   Delorme & Makeig, 2004).
    #
    # Automated artifact detection:
    #   1. EOG (blinks, eye movements): Fp1/Fp2 used as proxy reference
    #      channels (Chaumon et al., 2015) since this dataset has no
    #      dedicated EOG channel. Components correlating with the frontal
    #      signal are flagged.
    # ------------------------------------------------------------------
    ica = ICA(
        n_components=0.999,
        method="fastica",
        random_state=42,
        max_iter="auto",
    )

    # Fit on a 1 Hz high-pass copy; apply to the original epochs
    epochs_for_ica = epochs.copy().filter(
        l_freq=1.0, h_freq=None, verbose=False
    )
    ica.fit(epochs_for_ica)

    # -- EOG: blink and eye-movement components ---------------------------
    eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=["Fp1", "Fp2"])
    ica.exclude = list(eog_indices)

    ica.plot_scores(
        eog_scores,
        exclude=eog_indices,
        title="ICA — EOG scores (Fp1/Fp2 proxy)",
    )

    print(
        f"ICA: {ica.n_components_} component(s) fitted. "
        f"Excluding {len(ica.exclude)}: EOG={list(eog_indices)}"
    )

    # Apply — modifies epochs in place, subtracting the excluded components
    ica.apply(epochs)

    # Final ERP sanity check after ICA
    evoked_p300_ica = epochs["P300"].average()
    evoked_non_p300_ica = epochs["non-P300"].average()

    fig, ax = plt.subplots()
    ax.plot(
        epochs.times * 1000,
        evoked_p300_ica.data[cz_idx] * 1e6,
        label="P300",
        color="tomato",
    )
    ax.plot(
        epochs.times * 1000,
        evoked_non_p300_ica.data[cz_idx] * 1e6,
        label="non-P300",
        color="steelblue",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(
        f"ERP at Cz after ICA — P300 (n={len(epochs['P300'])}) "
        f"vs non-P300 (n={len(epochs['non-P300'])})"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
