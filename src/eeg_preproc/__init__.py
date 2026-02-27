"""EEG preprocessing utilities for the mcic analysis pipeline."""

from eeg_preproc.bad_channels import find_bad_channels, find_flat_channels
from eeg_preproc.filtering import highpass_filter

__all__ = ["find_bad_channels", "find_flat_channels", "highpass_filter"]
