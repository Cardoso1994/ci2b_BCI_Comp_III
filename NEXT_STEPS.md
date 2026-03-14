# Next Steps: Replicating EEGLAB's `clean_rawdata` Pipeline in Python/MNE

This document tracks the implementation of EEGLAB's `clean_rawdata` / `clean_artifacts`
preprocessing pipeline in Python using MNE-Python and supporting libraries. The goal is to
produce a preprocessing pipeline that is methodologically equivalent to the ci²b MATLAB
pipeline, suitable for publication.

---

## Pipeline Overview

The following table maps each EEGLAB step to its Python/MNE equivalent, in the verified
execution order sourced directly from `clean_artifacts.m`:

| # | EEGLAB function | ci²b parameter | Python/MNE equivalent | Status |
|---|---|---|---|---|
| 1 | `clean_flatlines` | `flatline_criterion = 5` | `find_flat_channels()` in `eeg_preproc` | ✅ Done |
| 2 | `clean_drifts` | `highpass = [0.1 0.5]` | `raw.filter()` — needs parameter fix | ⬜ Todo |
| 3 | `clean_channels` | `channel_criterion = 0.8` + `line_noise_criterion = 4` | `pyprep` (RANSAC + HF noise, combined with logical OR) | ⬜ Todo |
| 4 | `clean_asr` | `burst_criterion = 5`, `window_criterion = 0.25` | `meegkit.asr.ASR` | ✅ Done |
| 5 | `clean_windows` | `window_criterion = 0.25` | Post-ASR epoch rejection (at epoching stage) | ⬜ Todo |
| 6 | *(not in EEGLAB)* | — | `raw.interpolate_bads()` | ⬜ Todo |

> **Important architectural note:** In EEGLAB's `clean_rawdata`, bad channels detected in
> step 3 are **permanently removed** before ASR runs — they are never interpolated within
> the pipeline itself. In the MNE adaptation (step 6), interpolation is added as a
> MNE-specific step required before average re-referencing and ICA, which expect a full
> electrode set.

---

## Step 1 — Flatline Channel Removal ✅

**EEGLAB function:** `clean_flatlines`
**Parameter:** `flatline_criterion = 5` (seconds)

**What it does:** Removes channels that have any contiguous flat segment (near-zero
amplitude change between consecutive samples) lasting longer than `flatline_criterion`
seconds. Flatlines arise from electrode detachment, amplifier saturation, or
cable/connector failure. Even a single sustained flatline collapses the channel covariance
matrix rank, destabilising ASR calibration, ICA, and any other algorithm that inverts the
covariance matrix.

**Implementation:** `find_flat_channels()` in `src/eeg_preproc/bad_channels.py`.
Direct port of EEGLAB's algorithm (absolute sample-to-sample differences + run-length
encoding). Handles edge cases: NaN-valued channels, constant channels, recordings shorter
than the criterion.

**Usage:**
```python
from eeg_preproc import find_flat_channels

flat_bads = find_flat_channels(raw, flatline_criterion=5.0)
raw.info["bads"] += flat_bads
```

**Dependencies:** `mne`, `numpy` — no new packages needed.

---

## Step 2 — High-Pass Filter ⬜

**EEGLAB function:** `clean_drifts`
**Parameter:** `highpass = [0.1 0.5]` (Hz)

**What it does:** Applies a high-pass filter to remove slow DC drifts and very low
frequency noise before bad channel and ASR evaluation. The two values define the filter's
transition band: `0.1 Hz` is the stopband edge (full attenuation) and `0.5 Hz` is the
passband edge (-6 dB point). EEGLAB uses a zero-phase FIR filter with a Hamming window,
which MNE replicates with `phase='zero'`.

**Current issue in `scripts/01_preprocessing.py`:** The filter call is missing the
explicit transition bandwidth:
```python
# Current (incorrect transition band — MNE auto-calculates 0.5 Hz width, giving
# an effective passband edge near 1.0 Hz instead of 0.5 Hz):
experiment_signal.filter(l_freq=0.5, h_freq=40)

# Correct (transition width = 0.5 - 0.1 = 0.4 Hz, matching EEGLAB's [0.1 0.5]):
experiment_signal.filter(
    l_freq=0.5,
    h_freq=40,
    l_trans_bandwidth=0.4,
    fir_window="hamming",
    phase="zero",
)
```

**What needs to be done:**
- Update the `filter()` call in `scripts/01_preprocessing.py` with the explicit
  `l_trans_bandwidth=0.4`, `fir_window="hamming"`, and `phase="zero"` arguments.
- Consider moving this into a helper function in `eeg_preproc` for reusability and
  to document the parameter mapping explicitly.

**Dependencies:** `mne` — no new packages needed.

---

## Step 3 — Bad Channel Detection (RANSAC + Line Noise) ⬜

**EEGLAB function:** `clean_channels`
**Parameters:** `channel_criterion = 0.8`, `line_noise_criterion = 4`

**What it does:** Detects and removes bad channels using two complementary criteria,
combined with a logical OR — a channel failing *either* check is removed:

- **`channel_criterion = 0.8` (RANSAC correlation):** The channel's signal is predicted
  by spatially interpolating its neighbours using RANSAC (Random Sample Consensus).
  The correlation between the actual and predicted signal is computed in 5-second sliding
  windows. A channel is bad if its correlation with the RANSAC prediction drops below
  `0.8` in more than a threshold fraction of windows. This catches channels that are
  spatially inconsistent with their neighbours (bridging, poor contact, local muscle).

- **`line_noise_criterion = 4` (line noise z-score):** The ratio of line-frequency noise
  power to broadband signal power is computed per channel. Channels with a robust z-score
  above `4` standard deviations are flagged. This catches channels dominated by
  50/60 Hz powerline interference.

**Key architectural note:** In EEGLAB, both checks are performed in a **single call** to
`clean_channels()` and the results are combined as a logical OR. The bad channels are then
**permanently removed** (not marked) before ASR runs. In MNE, we mark them as
`raw.info["bads"]`, and `mne.pick_types(..., exclude="bads")` ensures they are excluded
from the ASR data extraction automatically.

**What needs to be done:**
1. Add `pyprep` to the project dependencies:
   ```toml
   # in pyproject.toml
   "pyprep>=0.5.0",
   ```
   Then run `uv sync`.

2. Implement a `find_bad_channels()` function in `src/eeg_preproc/bad_channels.py`
   that calls both checks and combines their results with a logical OR, mirroring
   EEGLAB's unified `clean_channels()` step:

   ```python
   from pyprep import NoisyChannels
   from pyprep.ransac import find_bad_by_ransac

   # Line noise criterion (approximation using broadband HF noise ratio)
   nc = NoisyChannels(raw, do_detrend=True, random_state=42)
   nc.find_bad_by_hfnoise(HF_zscore_threshold=4.0)
   line_noise_bads = nc.bad_by_hfnoise

   # Channel correlation criterion via RANSAC spatial reconstruction
   # Requires electrode positions — available via raw.info after set_montage()
   ransac_bads, _ = find_bad_by_ransac(
       data=raw.get_data(picks="eeg"),
       pos=...,               # channel positions from raw.info
       ch_names=raw.ch_names,
       sample_rate=raw.info["sfreq"],
       corr_thresh=0.8,       # channel_criterion
       frac_bad=0.25,
       corr_window_secs=5.0,  # EEGLAB default window length
       random_state=42,
   )

   # Combine — logical OR, matching EEGLAB's clean_channels()
   channel_bads = list(set(line_noise_bads) | set(ransac_bads))
   raw.info["bads"] += channel_bads
   ```

3. Expose `find_bad_channels()` via `src/eeg_preproc/__init__.py`.

**Fidelity notes:**
- `find_bad_by_ransac` with `corr_thresh=0.8` is the closest available Python equivalent
  to EEGLAB's `channel_criterion=0.8`. Both use RANSAC spatial reconstruction.
- `find_bad_by_hfnoise` uses broadband high-frequency noise ratio, which is an
  approximation of EEGLAB's line-frequency-specific SNR check. It is not identical but
  captures the same class of corrupted channels in practice.

**Dependencies:** `pyprep>=0.5.0` (confirmed compatible with NumPy 2.x and MNE>=1.3).

---

## Step 4 — Artifact Subspace Reconstruction (ASR) ✅

**EEGLAB function:** `clean_asr`
**Parameters:** `burst_criterion = 5`, `window_criterion = 0.25`

**What it does:** Reconstructs EEG data segments where the signal power exceeds
`burst_criterion` standard deviations above a clean baseline, using a subspace projection
learned from the clean portions of the signal. This removes muscle bursts, movement
artefacts, and other transient high-amplitude events while preserving the underlying
neural signal.

**Implementation:** `meegkit.asr.ASR` from the master branch of
`python-meegkit` (installed via git to pick up NumPy 2.x compatibility fixes not yet
released on PyPI as of v0.1.9).

**Current implementation in `scripts/01_preprocessing.py`:**
```python
eeg_picks = mne.pick_types(experiment_signal.info, eeg=True)
eeg_data = experiment_signal.get_data(picks=eeg_picks)
asr = ASR(sfreq=experiment_signal.info["sfreq"], cutoff=5)
asr.max_bad_chans = 0.25  # window_criterion; not exposed in constructor
asr.fit(eeg_data)
cleaned = asr.transform(eeg_data)
experiment_signal._data[eeg_picks] = cleaned
```

Note: `mne.pick_types(..., eeg=True)` uses `exclude="bads"` by default, so channels
marked bad in steps 1 and 3 are automatically excluded from ASR calibration and
transformation without any additional code change.

**Dependencies:** `meegkit @ git+https://github.com/nbara/python-meegkit.git@master`
(already in `pyproject.toml`).

---

## Step 5 — Post-ASR Window Rejection ⬜

**EEGLAB function:** `clean_windows`
**Parameter:** `window_criterion = 0.25`

**What it does:** After ASR repairs burst artefacts, this step removes any remaining data
windows where more than `window_criterion` (25%) of channels still exceed a power
threshold of ±7 standard deviations from the robust channel mean. These are windows that
ASR could not adequately reconstruct. In EEGLAB, `clean_windows` cuts these segments out
of the continuous raw recording.

**Mapping to MNE:** MNE's epoch-based workflow makes raw-data cutting undesirable (it
breaks event timing). The equivalent is applied at the **epoching stage** using
`autoreject` or a manual amplitude-based epoch rejection:

```python
import autoreject

# After creating epochs:
ar = autoreject.AutoReject(n_interpolate=[0], random_state=42)
epochs_clean = ar.fit_transform(epochs)

# Or simpler threshold-based rejection matching EEGLAB's ±7 SD criterion:
epochs.drop_bad(reject={"eeg": 7 * epochs.get_data().std()})
```

**What needs to be done:**
- Implement this at the epoch creation step (a later script, after `01_preprocessing.py`).
- Decide between `autoreject` (data-driven, more principled) or a fixed SD threshold
  (closer to EEGLAB's literal behaviour).

**Dependencies:**
- `autoreject` (recommended) — add `"autoreject>=0.4"` to `pyproject.toml`, or
- MNE built-in `epochs.drop_bad()` — no new packages needed.

---

## Step 6 — Bad Channel Interpolation ⬜

**Note:** This step does not exist in EEGLAB's `clean_rawdata`. EEGLAB permanently
removes bad channels. This step is **MNE-specific** and is added because subsequent steps
(average reference, ICA) expect a complete, full-rank electrode set.

**What it does:** Spherical spline interpolation reconstructs the signal at each bad
channel position from the weighted contributions of surrounding good channels. This
restores the full channel count before re-referencing and ICA.

**What needs to be done:**
- Call `raw.interpolate_bads(reset_bads=True)` after step 3 (bad channel detection) and
  before average re-referencing and ICA.
- `reset_bads=True` clears `raw.info["bads"]` after interpolation, so downstream steps
  see a full clean channel set.

```python
raw.interpolate_bads(reset_bads=True)
```

**Dependencies:** `mne` — no new packages needed. Requires that a montage with 3D
electrode positions has been set (`info.set_montage("standard_1020")`), which is already
done in `scripts/01_preprocessing.py`.
