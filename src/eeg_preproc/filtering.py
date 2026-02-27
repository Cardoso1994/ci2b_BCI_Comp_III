"""Filtering utilities for EEG preprocessing.

Implements frequency-domain preprocessing steps following EEGLAB's
``clean_rawdata`` plugin methodology (Kothe, 2012; Widmann et al., 2015),
adapted for MNE-Python's API and SI unit (Volt) convention.
"""

from __future__ import annotations

import mne
from mne.utils import logger
from mne.utils import verbose


@verbose
def highpass_filter(
    raw: mne.io.BaseRaw,
    passband_edge: float = 0.5,
    stopband_edge: float = 0.1,
    h_freq: float | None = None,
    fir_window: str = "hamming",
    phase: str = "zero",
    verbose: bool | str | int | None = None,
) -> None:
    """Apply a high-pass FIR filter replicating EEGLAB's ``clean_drifts``.

    Removes slow DC drifts and low-frequency noise before bad channel
    detection and ASR, matching the ``clean_drifts`` step of EEGLAB's
    ``clean_rawdata`` plugin when called with ``highpass = [0.1 0.5]``.

    The two EEGLAB parameters map to MNE arguments as follows:

    +---------------------+-------------------+-------------------------------------------+
    | EEGLAB              | MNE               | Description                               |
    +=====================+===================+===========================================+
    | ``highpass(1)=0.1`` | ``stopband_edge`` | Frequency of full attenuation (−∞ dB)    |
    +---------------------+-------------------+-------------------------------------------+
    | ``highpass(2)=0.5`` | ``passband_edge`` | Frequency of −6 dB (``l_freq`` in MNE)   |
    +---------------------+-------------------+-------------------------------------------+
    | (derived)           | ``l_trans_bandwidth = passband_edge − stopband_edge``             |
    +---------------------+-------------------------------------------------------------------+

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG recording. Must be preloaded (``raw.preload == True``).
        Modified in-place; the function returns ``None``.
    passband_edge : float, optional
        High-pass passband edge in Hz — the frequency at which the filter
        reaches −6 dB. Corresponds to ``highpass(2)`` in EEGLAB's
        ``clean_rawdata`` (default ``0.5`` Hz). This is the ``l_freq``
        argument passed to :func:`mne.io.BaseRaw.filter`.
    stopband_edge : float, optional
        High-pass stopband edge in Hz — the frequency at which full
        attenuation is reached. Corresponds to ``highpass(1)`` in EEGLAB's
        ``clean_rawdata`` (default ``0.1`` Hz). The transition bandwidth is
        derived as ``passband_edge − stopband_edge``.
    h_freq : float | None, optional
        Low-pass cutoff in Hz. When ``None`` (default), no low-pass is
        applied and this function is a pure high-pass, exactly matching
        EEGLAB's ``clean_drifts``. Pass a value (e.g. ``40.0``) to also
        apply a low-pass in the same filter design call — this is not part
        of EEGLAB's step 2 but is sometimes applied here for convenience.
    fir_window : str, optional
        Window function for FIR filter design. Default ``"hamming"`` matches
        EEGLAB's ``clean_drifts`` implementation (Kaiser window in older
        EEGLABs; Hamming is equivalent for the typical passband ripple
        tolerance and is MNE's default window).
    phase : str, optional
        Filter phase. Default ``"zero"`` applies a zero-phase (forward +
        backward) filter, matching EEGLAB's behaviour and preventing
        phase-induced temporal distortion of ERP waveforms. Use
        ``"zero-double"`` for maximum precision at ~2× computational cost.
    verbose : bool | str | int | None, optional
        Verbosity level. ``None`` inherits MNE's global setting.

    Returns
    -------
    None
        The filter is applied in-place to ``raw._data``.

    Raises
    ------
    RuntimeError
        If ``raw`` is not preloaded.
    ValueError
        If ``stopband_edge >= passband_edge``, which would produce a
        non-positive transition bandwidth (undefined filter).
    ValueError
        If ``h_freq`` is provided and ``h_freq <= passband_edge``.

    Notes
    -----
    **FIR order:** MNE auto-selects the filter order to guarantee the
    requested transition bandwidth at the given sampling rate, using the
    Kaiser-window formula. The resulting order may be very large for low
    transition bandwidths (``0.4 Hz`` at ``240 Hz → ~1780 taps``); this is
    expected and correctly handled by MNE's overlap-add FFT convolution.

    **Relationship to ``clean_rawdata``:** EEGLAB applies ``clean_drifts``
    **before** ``clean_channels`` and ASR. Filtering must therefore precede
    bad channel detection and ASR calibration in the MNE pipeline as well.
    Flatline channels (step 1) should be marked bad **before** calling this
    function; MNE's filter applies to all non-bad EEG channels by default
    (``picks="data"`` in recent MNE versions includes bad channels — be
    explicit if needed, but for most pipelines the default is fine).

    References
    ----------
    .. [1] Kothe, C. A. (2012). *clean_rawdata* [Software].
           https://github.com/sccn/clean_rawdata
    .. [2] Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter
           design for electrophysiological data — a practical approach.
           *Journal of Neuroscience Methods*, 250, 34–46.
           https://doi.org/10.1016/j.jneumeth.2014.08.002
    .. [3] de Cheveigné, A., & Nelken, I. (2019). Filters: When, why, and
           how (not) to use them. *Neuron*, 102(2), 280–293.
           https://doi.org/10.1016/j.neuron.2019.02.039

    Examples
    --------
    Replicate EEGLAB's ``clean_drifts`` with ``highpass = [0.1 0.5]``:

    >>> from eeg_preproc import highpass_filter
    >>> highpass_filter(raw, passband_edge=0.5, stopband_edge=0.1)

    High-pass and low-pass in one call (not matching EEGLAB step 2 exactly):

    >>> highpass_filter(raw, passband_edge=0.5, stopband_edge=0.1, h_freq=40.0)
    """
    if not raw.preload:
        raise RuntimeError(
            "Raw data must be preloaded. Call raw.load_data() first."
        )
    if stopband_edge >= passband_edge:
        raise ValueError(
            f"stopband_edge ({stopband_edge} Hz) must be strictly less than "
            f"passband_edge ({passband_edge} Hz) to produce a valid positive "
            "transition bandwidth."
        )
    if h_freq is not None and h_freq <= passband_edge:
        raise ValueError(
            f"h_freq ({h_freq} Hz) must be greater than passband_edge "
            f"({passband_edge} Hz)."
        )

    l_trans_bandwidth: float = passband_edge - stopband_edge

    logger.info(
        "Applying high-pass FIR filter: passband %.2f Hz, stopband %.2f Hz "
        "(transition bandwidth %.2f Hz), window=%s, phase=%s.",
        passband_edge,
        stopband_edge,
        l_trans_bandwidth,
        fir_window,
        phase,
    )

    raw.filter(
        l_freq=passband_edge,
        h_freq=h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        fir_window=fir_window,
        phase=phase,
        verbose=verbose,
    )
