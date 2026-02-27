"""Bad channel detection algorithms for EEG preprocessing.

Implements channel quality checks following the methodology of EEGLAB's
``clean_rawdata`` plugin (Kothe, 2012; Kothe & Jung, 2016), adapted for
MNE-Python's SI unit (Volt) convention and data model.

Functions
---------
find_flat_channels
    Detect channels with sustained flatline segments (``clean_flatlines``).
find_bad_channels
    Detect bad channels via RANSAC and high-frequency noise (``clean_channels``).
"""

from __future__ import annotations

import mne
from mne.utils import logger
from mne.utils import verbose
import numpy as np
from numpy.typing import NDArray


@verbose
def find_flat_channels(
    raw: mne.io.BaseRaw,
    flatline_criterion: float = 5.0,
    flat_threshold: float = 1e-15,
    picks: str | list[str] | None = "eeg",
    verbose: bool | str | int | None = None,
) -> list[str]:
    """Detect channels with sustained flatline artifacts.

    A channel is marked as bad when any contiguous segment of near-zero
    amplitude change exceeds ``flatline_criterion`` seconds. This replicates
    the behaviour of EEGLAB's ``clean_flatlines`` (Kothe, 2012), adapted for
    MNE-Python's Volt unit convention.

    Flat segments arise when an electrode loses scalp contact, an amplifier
    saturates (rails at ±V_max), or a cable/connector fails. Despite
    appearing clean — no large spikes, no obvious noise — a flat channel
    introduces a near-zero eigenvalue into the channel covariance matrix.
    This destabilises every algorithm that inverts or decomposes that matrix,
    most critically ASR calibration, ICA decomposition, and source
    beamformers.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG recording. Must be preloaded (``raw.preload == True``).
    flatline_criterion : float, optional
        Maximum tolerated duration (seconds) of a contiguous flat segment
        before the channel is flagged as bad. Default ``5.0`` s matches
        EEGLAB's ``clean_rawdata`` default (``flatline_criterion = 5``).
    flat_threshold : float, optional
        Amplitude-change threshold (Volts) below which two consecutive
        samples are considered identical. Default ``1e-15`` V (1 femtovolt)
        is the Volt equivalent of EEGLAB's ``1e-10`` µV — well above float64
        rounding noise for typical EEG amplitudes (~10–100 µV) while
        remaining orders of magnitude below any physiologically meaningful
        signal change. Raise this value if your data was quantised at low
        resolution (e.g. 8-bit ADC) and shows staircase patterns that should
        not be treated as flatlines.
    picks : "eeg" | list[str] | None, optional
        Channels to evaluate. ``"eeg"`` (default) selects all EEG channels
        and automatically excludes those already listed in
        ``raw.info["bads"]``. Pass an explicit list of channel names to
        restrict the check to a subset.
    verbose : bool | str | int | None, optional
        Verbosity level. ``None`` inherits MNE's global setting. ``True`` or
        ``"INFO"`` prints a summary; ``"DEBUG"`` adds per-channel detail.
        See :func:`mne.utils.set_log_level`.

    Returns
    -------
    list[str]
        Names of channels that contain at least one flatline segment
        exceeding ``flatline_criterion`` seconds. Returns an empty list when
        no flat channels are found. Does **not** modify ``raw.info["bads"]``;
        the caller decides how to handle the result.

    Raises
    ------
    RuntimeError
        If ``raw`` is not preloaded.
    ValueError
        If ``flatline_criterion`` or ``flat_threshold`` are not strictly
        positive.
    ValueError
        If ``picks`` is not ``"eeg"``, ``None``, or a list of strings.

    Notes
    -----
    **Algorithm** (direct port of EEGLAB ``clean_flatlines``):

    For each channel *c* with signal *x_c[t]*:

    1. Compute absolute sample-to-sample differences::

           d_c[t] = |x_c[t+1] - x_c[t]|,   t = 0, ..., T-2

    2. Label sample *t* as *flat* when ``d_c[t] < flat_threshold``.

    3. Identify contiguous runs of flat samples using run-length encoding
       (padding the boolean array with ``False`` sentinels on both ends, then
       differencing to locate run boundaries — identical to EEGLAB's
       ``diff([false is_flat false])`` idiom).

    4. Flag the channel when::

           max(run lengths) > sfreq * flatline_criterion

    **Edge cases handled:**

    - **NaN values**: flagged unconditionally. NaN propagates silently through
      covariance computation and must be excluded regardless of run length.
    - **Constant channel** (zero peak-to-peak): flagged unconditionally
      without evaluating run lengths; a perfectly constant signal is an
      infinite flatline by definition.
    - **Short recording**: if the recording duration is shorter than or equal
      to ``flatline_criterion``, the duration criterion can never be satisfied
      by a partial flatline. A warning is issued and only entirely flat
      channels (constant signal or all-NaN) can be detected. The function
      still returns a valid result.
    - **Empty picks**: returns an empty list immediately without accessing
      the data array.

    References
    ----------
    [1] Kothe, C. A. (2012). *clean_rawdata* [Software].
           https://github.com/sccn/clean_rawdata
    [2] Kothe, C. A., & Jung, T.-P. (2016). Artifact removal techniques
           with signal reconstruction. US Patent 20160113587A1.
    [3] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., &
           Robbins, K. A. (2015). The PREP pipeline: standardized
           preprocessing for large-scale EEG analysis.
           *Frontiers in Neuroinformatics*, 9, 16.
           https://doi.org/10.3389/fninf.2015.00016

    Examples
    --------
    Basic usage with default parameters:

    >>> flat_bads = find_flat_channels(raw, flatline_criterion=5.0)
    >>> print(f"Flat channels detected: {flat_bads}")
    >>> raw.info["bads"] += flat_bads

    Verbose per-channel output for inspection:

    >>> flat_bads = find_flat_channels(raw, verbose="DEBUG")
    """
    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    if not raw.preload:
        raise RuntimeError(
            "Raw data must be preloaded. Call raw.load_data() first."
        )
    if flatline_criterion <= 0:
        raise ValueError(
            f"flatline_criterion must be strictly positive, got {flatline_criterion!r}."
        )
    if flat_threshold <= 0:
        raise ValueError(
            f"flat_threshold must be strictly positive, got {flat_threshold!r}."
        )
    if picks is not None and not isinstance(picks, (str, list)):
        raise ValueError(
            f"picks must be 'eeg', a list of channel names, or None. Got {picks!r}."
        )

    # ------------------------------------------------------------------ #
    # Channel selection                                                    #
    # ------------------------------------------------------------------ #
    if isinstance(picks, list):
        picks_idx: NDArray[np.intp] = mne.pick_channels(
            raw.ch_names, include=picks, exclude=raw.info["bads"]
        )
    else:
        # picks == "eeg" or None — both default to EEG channels, bads excluded
        picks_idx = mne.pick_types(raw.info, eeg=True, exclude="bads")

    if picks_idx.size == 0:
        logger.warning(
            "No channels matched the picks criteria. Returning empty list."
        )
        return []

    ch_names: list[str] = [raw.ch_names[i] for i in picks_idx]
    sfreq: float = raw.info["sfreq"]
    min_flat_samples: float = flatline_criterion * sfreq

    logger.info(
        "Scanning %d channel(s) for flatline segments "
        "(criterion: %.1f s, threshold: %.2e V).",
        len(ch_names),
        flatline_criterion,
        flat_threshold,
    )

    # ------------------------------------------------------------------ #
    # Short-recording guard                                                #
    # ------------------------------------------------------------------ #
    recording_duration: float = raw.n_times / sfreq
    if recording_duration <= flatline_criterion:
        mne.utils.warn(
            f"Recording duration ({recording_duration:.2f} s) is shorter than or "
            f"equal to flatline_criterion ({flatline_criterion} s). Only entirely "
            "flat channels (constant signal or NaN) can be reliably detected."
        )

    # ------------------------------------------------------------------ #
    # Data extraction                                                    #
    # Compute all sample-to-sample differences in a single vectorised    #
    # call to avoid repeated get_data() overhead per channel.            #
    # ------------------------------------------------------------------ #
    data: NDArray[np.floating] = raw.get_data(picks=picks_idx)
    # shape: (n_channels, n_times)

    diffs: NDArray[np.floating] = np.abs(np.diff(data, axis=1))
    # shape: (n_channels, n_times - 1)
    # diffs[c, t] = |x_c[t+1] - x_c[t]|

    # ------------------------------------------------------------------ #
    # Per-channel flatline evaluation                                    #
    # ------------------------------------------------------------------ #
    flat_channels: list[str] = []

    for ch_idx, ch_name in enumerate(ch_names):
        signal = data[ch_idx]

        # -- NaN guard: flag immediately, skip run-length logic -----------
        # NaN can arise from dropped packets or hardware faults.
        # A channel with any NaN must be excluded regardless of duration.
        if np.any(np.isnan(signal)):
            logger.warning(
                "Channel %s contains NaN values — flagged as bad.", ch_name
            )
            flat_channels.append(ch_name)
            continue

        # -- Zero peak-to-peak: entirely constant signal ------------------
        # Avoids run-length logic for the degenerate case and handles
        # channels whose diff array would be all-zero.
        if signal.max() == signal.min():
            logger.debug(
                "Channel %s has zero peak-to-peak amplitude — flagged.", ch_name
            )
            flat_channels.append(ch_name)
            continue

        # -- Run-length encoding of flat samples --------------------------
        # Equivalent to EEGLAB's:
        #   reshape(find(diff([false abs(diff(x)) < 1e-10 false])), 2, [])'
        #
        # Padding with False sentinels on both ends ensures that runs
        # starting at sample 0 or ending at the last sample are correctly
        # captured by np.diff.
        is_flat: NDArray[np.bool_] = diffs[ch_idx] < flat_threshold

        padded = np.concatenate(([False], is_flat, [False])).astype(np.int8)
        transitions = np.diff(padded)
        run_starts: NDArray[np.intp] = np.where(transitions == 1)[0]
        run_ends: NDArray[np.intp] = np.where(transitions == -1)[0]

        if run_starts.size == 0:
            continue

        max_flat_samples: int = int((run_ends - run_starts).max())

        if max_flat_samples > min_flat_samples:
            flat_channels.append(ch_name)
            logger.debug(
                "Channel %s flagged: longest flat segment = %.3f s "
                "(criterion: %.1f s).",
                ch_name,
                max_flat_samples / sfreq,
                flatline_criterion,
            )

    logger.info(
        "Flatline detection complete: %d / %d channel(s) flagged.",
        len(flat_channels),
        len(ch_names),
    )
    return flat_channels


@verbose
def find_bad_channels(
    raw: mne.io.BaseRaw,
    channel_criterion: float = 0.8,
    line_noise_criterion: float = 4.0,
    frac_bad: float = 0.4,
    corr_window_secs: float = 5.0,
    random_state: int = 42,
    verbose: bool | str | int | None = None,
) -> list[str]:
    """Detect bad channels via RANSAC correlation and HF noise z-score.

    Replicates EEGLAB's ``clean_channels`` step from ``clean_rawdata``,
    which combines two independent bad-channel criteria with a logical OR:
    a channel failing **either** check is flagged. The two criteria map to
    ``pyprep``'s :class:`~pyprep.NoisyChannels` API as follows:

    +---------------------------------+-------------------------------------+------------------+
    | EEGLAB parameter                | Criterion                           | pyprep method    |
    +=================================+=====================================+==================+
    | ``channel_criterion = 0.8``     | RANSAC spatial correlation < 0.8    | ``find_bad_by_ransac`` |
    +---------------------------------+-------------------------------------+------------------+
    | ``line_noise_criterion = 4``    | HF noise robust z-score > 4         | ``find_bad_by_hfnoise`` |
    +---------------------------------+-------------------------------------+------------------+

    **RANSAC criterion** (``channel_criterion``): The channel's signal is
    predicted by spatially interpolating neighbours via RANSAC. The correlation
    between the actual and reconstructed signal is computed in
    ``corr_window_secs``-second non-overlapping windows. A channel is bad if
    the fraction of windows with correlation below ``channel_criterion`` exceeds
    ``frac_bad`` (EEGLAB default: ``max_broken_time = 0.4``). This catches
    channels that are spatially inconsistent with the surrounding electrode
    neighbourhood — typically bridging artefacts, poor scalp contact, or
    highly localised muscle contamination.

    **HF noise criterion** (``line_noise_criterion``): The ratio of broadband
    high-frequency signal power to total signal power is computed per channel
    and converted to a robust z-score across channels. Channels with a z-score
    above ``line_noise_criterion`` are flagged. This is an approximation of
    EEGLAB's line-frequency SNR check: pyprep evaluates broadband HF content
    rather than 50/60 Hz specifically, capturing the same class of amplifier
    noise and powerline contamination in practice.

    .. warning::
        **Interaction with upstream low-pass filtering**: If a low-pass filter
        was applied before this function (e.g. ``h_freq=40.0`` in step 2),
        broadband HF signal power above 40 Hz is zeroed out. The HF noise
        criterion then becomes vacuous — all channels will appear clean by
        this measure regardless of their actual quality. For the RANSAC
        criterion to remain valid the low-pass is irrelevant; for the HF
        noise criterion to be meaningful the data must retain signal above
        roughly 40–50 Hz. Set ``h_freq=None`` in the upstream filter call
        for strict EEGLAB fidelity.

    **Architectural note**: In EEGLAB, detected bad channels are permanently
    removed before ASR runs. In MNE the equivalent is marking them in
    ``raw.info["bads"]`` — :func:`mne.pick_types` with ``exclude="bads"``
    (the default) then automatically excludes them from the ASR data
    extraction, interpolation target list, and any subsequent step that
    respects the MNE bads convention.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG recording. Must be preloaded. Channels already listed
        in ``raw.info["bads"]`` (e.g. flatlines from step 1) are automatically
        excluded from evaluation and from use as RANSAC predictors — pyprep's
        :class:`~pyprep.NoisyChannels` calls ``raw.pick("eeg")`` internally,
        which drops bad channels from its working copy.
    channel_criterion : float, optional
        Minimum acceptable RANSAC correlation between a channel's actual and
        spatially-predicted signal, in range [0, 1]. Corresponds to EEGLAB's
        ``channel_criterion = 0.8``. Channels whose correlation falls below
        this threshold in more than ``frac_bad`` of windows are flagged.
        Default ``0.8``.
    line_noise_criterion : float, optional
        Robust z-score threshold for broadband high-frequency noise content.
        Corresponds to EEGLAB's ``line_noise_criterion = 4``. Channels whose
        HF noise z-score exceeds this value are flagged. Default ``4.0``.
    frac_bad : float, optional
        Fraction of RANSAC windows that must fall below ``channel_criterion``
        for a channel to be considered bad. Corresponds to EEGLAB's
        ``max_broken_time = 0.4``. Default ``0.4``.
    corr_window_secs : float, optional
        Duration (seconds) of each non-overlapping RANSAC evaluation window.
        Default ``5.0`` s matches EEGLAB's ``clean_channels`` window length.
    random_state : int, optional
        Seed for RANSAC random sampling, for reproducibility. Default ``42``.
    verbose : bool | str | int | None, optional
        Verbosity level. ``None`` inherits MNE's global setting.

    Returns
    -------
    list[str]
        Channel names flagged as bad by either criterion (logical OR). Returns
        an empty list when no bad channels are found. Does **not** modify
        ``raw.info["bads"]``; the caller decides how to handle the result.

    Raises
    ------
    RuntimeError
        If ``raw`` is not preloaded.
    ValueError
        If ``channel_criterion`` is not in (0, 1].
    ValueError
        If ``line_noise_criterion`` is not strictly positive.
    ValueError
        If ``frac_bad`` is not in (0, 1].

    Notes
    -----
    Requires a montage with 3-D electrode positions to be set on ``raw``
    (e.g. ``raw.info.set_montage("standard_1020")``). RANSAC uses electrode
    coordinates to build the spatial interpolation model; without positions
    ``find_bad_by_ransac`` will raise an error.

    References
    ----------
    .. [1] Kothe, C. A. (2012). *clean_rawdata* [Software].
           https://github.com/sccn/clean_rawdata
    .. [2] Kothe, C. A., & Jung, T.-P. (2016). Artifact removal techniques
           with signal reconstruction. US Patent 20160113587A1.
    .. [3] Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus:
           a paradigm for model fitting with applications to image analysis
           and automated cartography. *CACM*, 24(6), 381–395.
           https://doi.org/10.1145/358669.358692
    .. [4] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., &
           Robbins, K. A. (2015). The PREP pipeline: standardized
           preprocessing for large-scale EEG analysis.
           *Frontiers in Neuroinformatics*, 9, 16.
           https://doi.org/10.3389/fninf.2015.00016

    Examples
    --------
    >>> from eeg_preproc import find_bad_channels
    >>> channel_bads = find_bad_channels(raw)
    >>> print(f"Bad channels: {channel_bads}")
    >>> raw.info["bads"] += channel_bads
    """
    from pyprep import NoisyChannels  # deferred: optional heavy dependency

    if not raw.preload:
        raise RuntimeError(
            "Raw data must be preloaded. Call raw.load_data() first."
        )
    if not (0 < channel_criterion <= 1):
        raise ValueError(
            f"channel_criterion must be in (0, 1], got {channel_criterion!r}."
        )
    if line_noise_criterion <= 0:
        raise ValueError(
            f"line_noise_criterion must be strictly positive, "
            f"got {line_noise_criterion!r}."
        )
    if not (0 < frac_bad <= 1):
        raise ValueError(f"frac_bad must be in (0, 1], got {frac_bad!r}.")

    n_already_bad = len(raw.info["bads"])
    logger.info(
        "Running bad channel detection (RANSAC + HF noise). "
        "%d channel(s) already in info['bads'] will be excluded.",
        n_already_bad,
    )

    # NoisyChannels internally calls raw.copy().pick("eeg"), which drops
    # channels in raw.info["bads"], so step-1 flatline bads are auto-excluded.
    nc = NoisyChannels(raw, do_detrend=True, random_state=random_state)

    # ------------------------------------------------------------------ #
    # Criterion 1: high-frequency broadband noise z-score                 #
    # Approximates EEGLAB's line_noise_criterion (50/60 Hz SNR).          #
    # NOTE: ineffective if h_freq < ~50 Hz was applied upstream.          #
    # ------------------------------------------------------------------ #
    nc.find_bad_by_hfnoise(HF_zscore_threshold=line_noise_criterion)
    hf_bads: list[str] = nc.bad_by_hf_noise

    logger.info(
        "HF noise criterion (z > %.1f): %d channel(s) flagged: %s",
        line_noise_criterion,
        len(hf_bads),
        hf_bads or "none",
    )

    # ------------------------------------------------------------------ #
    # Criterion 2: RANSAC spatial correlation                             #
    # Approximates EEGLAB's channel_criterion (RANSAC corr < threshold). #
    # ------------------------------------------------------------------ #
    nc.find_bad_by_ransac(
        corr_thresh=channel_criterion,
        frac_bad=frac_bad,
        corr_window_secs=corr_window_secs,
    )
    ransac_bads: list[str] = nc.bad_by_ransac

    logger.info(
        "RANSAC criterion (corr < %.2f in > %.0f%% of windows): "
        "%d channel(s) flagged: %s",
        channel_criterion,
        frac_bad * 100,
        len(ransac_bads),
        ransac_bads or "none",
    )

    # ------------------------------------------------------------------ #
    # Combine — logical OR, matching EEGLAB's clean_channels()           #
    # ------------------------------------------------------------------ #
    channel_bads: list[str] = sorted(set(hf_bads) | set(ransac_bads))

    logger.info(
        "Bad channel detection complete: %d channel(s) flagged "
        "(HF noise: %d, RANSAC: %d, overlap: %d).",
        len(channel_bads),
        len(hf_bads),
        len(ransac_bads),
        len(set(hf_bads) & set(ransac_bads)),
    )
    return channel_bads
