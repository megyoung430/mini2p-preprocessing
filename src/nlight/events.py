"""Event-aligned trials: windows, trial means/SEM, per-trial prestimulus subtraction.

Traces are ``(traces, frames)`` arrays (a single trace is promoted to one row).
Trials are ``(events, traces, relative_time)``. Only events whose full window
lies inside the recording are used.
"""
from __future__ import annotations

import numpy as np


def window_offsets(fs: float, pre_s: float, post_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame offsets and their times (s) for a ``-pre_s`` to ``+post_s`` window."""
    if pre_s < 0 or post_s <= 0:
        raise ValueError("need pre_s >= 0 and post_s > 0")
    offsets = np.arange(-int(round(pre_s * fs)), int(round(post_s * fs)) + 1)
    return offsets, offsets / fs


def aligned_trials(x: np.ndarray, onsets: np.ndarray,
                   offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(trials, used_onsets)``; events without a complete window are dropped."""
    x = np.atleast_2d(x)
    onsets = np.asarray(onsets, dtype=int)
    complete = (onsets + offsets[0] >= 0) & (onsets + offsets[-1] < x.shape[1])
    used = onsets[complete]
    if not len(used):
        raise ValueError("no event has a complete window inside the recording")
    trials = x[:, used[:, None] + offsets[None, :]].transpose(1, 0, 2)
    return trials, used


def mean_sem(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, SEM and count over finite trials (axis 0). SEM is across events in
    this recording, not across animals."""
    count = np.isfinite(trials).sum(axis=0)
    mean = np.full(count.shape, np.nan)
    np.divide(np.nansum(trials, axis=0), count, out=mean, where=count > 0)
    sq = np.nansum((trials - mean) ** 2, axis=0)
    var = np.full(count.shape, np.nan)
    np.divide(sq, count - 1, out=var, where=count > 1)
    return mean, np.sqrt(var / np.maximum(count, 1)), count


def subtract_prestimulus(trials: np.ndarray, relative_s: np.ndarray,
                         start_s: float = -2.0, min_valid_fraction: float = 0.8
                         ) -> tuple[np.ndarray, np.ndarray]:
    """``(original, centred)`` trials on matched eligible trial/trace pairs.

    Each trial/trace is centred on its own mean over ``[start_s, 0)``. A pair
    with fewer than ``min_valid_fraction`` valid prestimulus samples is NaN in
    *both* outputs, so the two averages use the same trials.
    """
    window = (relative_s >= start_s) & (relative_s < 0)
    if not window.any():
        raise ValueError(f"window has no samples in [{start_s}, 0) s")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    pre = trials[..., window]
    count = np.isfinite(pre).sum(axis=-1)
    level = np.full(count.shape, np.nan)
    np.divide(np.nansum(pre, axis=-1), count, out=level,
              where=count >= np.ceil(min_valid_fraction * window.sum()))
    original = np.where(np.isfinite(level)[..., None], trials, np.nan)
    return original, original - level[..., None]
