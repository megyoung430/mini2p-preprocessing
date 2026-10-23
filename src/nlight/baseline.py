"""Baseline (F0) estimation and dF/F for fluorescence traces.

All functions are pure: they take arrays and return arrays. Traces are laid out
as ``(..., frames)`` — a 1-D whole-FOV trace or a 2-D ``tiles x frames`` array.
Bad frames are passed as a boolean mask over frames; they are excluded from
baseline estimation and are NaN in the returned dF/F.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fraction of the window that must hold valid samples before a rolling
# percentile is reported; shorter support at the recording edges is allowed.
MIN_VALID_FRACTION = 0.25


def window_frames(window_s: float, fs: float) -> int:
    """Odd window length in frames (>= 3) for a centered window of ``window_s``."""
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"baseline window must be positive and finite, got {window_s!r}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"frame rate must be positive and finite, got {fs!r}")
    window = max(3, int(round(window_s * fs)))
    # odd so the window is symmetric about the frame it is centered on
    return window + (window % 2 == 0)


def min_valid_samples(window: int, n_frames: int) -> int:
    """Minimum valid samples a rolling window needs before it yields a value."""
    return max(1, int(np.ceil(MIN_VALID_FRACTION * min(window, n_frames))))


def mask_bad_frames(F: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Float copy of ``F`` with bad and non-finite samples set to NaN."""
    F = np.asarray(F, dtype=float)
    bad = _check_bad(bad, F.shape[-1])
    masked = np.where(bad, np.nan, F)
    masked[~np.isfinite(masked)] = np.nan
    return masked


def rolling_percentile(F: np.ndarray, bad: np.ndarray, fs: float,
                       window_s: float, percentile: float) -> np.ndarray:
    """Centered rolling ``percentile`` of ``F``, ignoring bad frames."""
    _check_percentile(percentile)
    masked = mask_bad_frames(F, bad)
    window = window_frames(window_s, fs)
    minimum = min_valid_samples(window, masked.shape[-1])
    # DataFrame.rolling works column-wise, so frames go on the rows.
    frames_by_trace = pd.DataFrame(np.atleast_2d(masked).T)
    F0 = (frames_by_trace.rolling(window, center=True, min_periods=minimum)
          .quantile(percentile / 100).to_numpy().T)
    return F0.reshape(masked.shape)


def fixed_percentile(F: np.ndarray, bad: np.ndarray, percentile: float) -> np.ndarray:
    """One baseline per trace: ``percentile`` over the full recording, broadcast over frames."""
    _check_percentile(percentile)
    masked = mask_bad_frames(F, bad)
    if np.all(np.isnan(masked), axis=-1).any():
        raise ValueError("a trace has no valid samples for a fixed baseline")
    level = np.nanpercentile(masked, percentile, axis=-1, keepdims=True)
    return np.broadcast_to(level, masked.shape).copy()


def delta_f_over_f(F: np.ndarray, F0: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Fractional (F - F0) / F0; NaN on bad frames and where F0 is missing or <= 0."""
    F = np.asarray(F, dtype=float)
    F0 = np.asarray(F0, dtype=float)
    if F.shape != F0.shape:
        raise ValueError(f"F {F.shape} and F0 {F0.shape} differ in shape")
    bad = _check_bad(bad, F.shape[-1])
    dff = np.full(F.shape, np.nan)
    valid = np.isfinite(F) & np.isfinite(F0) & (F0 > 0) & ~bad
    np.divide(F - F0, F0, out=dff, where=valid)
    return dff


def _check_bad(bad: np.ndarray, n_frames: int) -> np.ndarray:
    bad = np.asarray(bad, dtype=bool).ravel()
    if bad.shape != (n_frames,):
        raise ValueError(f"bad-frame mask has {bad.size} entries for {n_frames} frames")
    if bad.all():
        raise ValueError("every frame is marked bad")
    return bad


def _check_percentile(percentile: float) -> None:
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
