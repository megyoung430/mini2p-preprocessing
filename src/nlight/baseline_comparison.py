"""Does the choice of F0 change the whole-FOV signal?

Starting from the raw whole-FOV fluorescence (pixels averaged *before* any
normalisation), compute dF/F with several baseline definitions and ask two
questions of each, relative to the current rolling baseline:

1. **Are slow fluctuations preserved?** Each dF/F is split into a slow part (a
   centered rolling mean over ``slow_timescale_s``) and the fast remainder. We
   report the correlation and the amplitude ratio of the slow parts.
2. **Are relative dF/F amplitudes preserved?** Within contiguous blocks of the
   recording, regress the method's fast component on the reference's. The
   slope is the method's local gain. A gain that is constant across blocks
   (low ``gain_cv``) means transients keep their relative sizes, even if all
   of them are scaled. A gain that drifts means the baseline is rescaling
   early and late transients differently, for example under bleaching.

Everything here is pure: arrays in, arrays and DataFrames out.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import baseline


def candidate_baselines(F: np.ndarray, bad: np.ndarray, fs: float,
                        current_window_s: float, current_percentile: float,
                        fixed_percentiles: tuple[float, ...] = (10,),
                        long_windows_s: tuple[float, ...] = (180, 300, 600)) -> dict[str, np.ndarray]:
    """F0 traces keyed by label; the first entry is the current method (the reference)."""
    duration_s = np.shape(F)[-1] / fs  # F may be one trace or tiles x frames
    baselines = {
        current_label(current_window_s, current_percentile):
            baseline.rolling_percentile(F, bad, fs, current_window_s, current_percentile)}
    for p in fixed_percentiles:
        baselines[f"Fixed full-recording p{p:g}"] = baseline.fixed_percentile(F, bad, p)
    for seconds in long_windows_s:
        if seconds >= duration_s:
            # Deliberate skip, not an error: a window longer than the recording
            # is just a fixed baseline with edge effects, already covered above.
            warnings.warn(f"skipping {seconds:g} s window: recording is {duration_s:.0f} s")
            continue
        baselines[f"{seconds:g} s rolling p{current_percentile:g}"] = baseline.rolling_percentile(
            F, bad, fs, seconds, current_percentile)
    return baselines


def current_label(window_s: float, percentile: float) -> str:
    return f"Current: {window_s:g} s rolling p{percentile:g}"


def slow_component(x: np.ndarray, fs: float, timescale_s: float) -> np.ndarray:
    """Centered rolling mean over ``timescale_s``; NaN where ``x`` is NaN."""
    window = baseline.window_frames(timescale_s, fs)
    smooth = (pd.Series(x).rolling(window, center=True,
                                   min_periods=baseline.min_valid_samples(window, len(x)))
              .mean().to_numpy())
    return np.where(np.isfinite(x), smooth, np.nan)


def block_gains(fast: np.ndarray, fast_reference: np.ndarray, n_blocks: int) -> np.ndarray:
    """Least-squares slope of ``fast`` on ``fast_reference`` in each contiguous block."""
    if n_blocks < 1:
        raise ValueError("need at least one block")

    def gain(a: np.ndarray, b: np.ndarray) -> float:
        use = np.isfinite(a) & np.isfinite(b)
        if use.sum() < 3 or np.var(b[use]) == 0:
            return np.nan
        return float(np.cov(a[use], b[use])[0, 1] / np.var(b[use], ddof=1))

    return np.array([gain(a, b) for a, b in zip(np.array_split(fast, n_blocks),
                                                np.array_split(fast_reference, n_blocks))])


def raw_drift(F: np.ndarray, bad: np.ndarray, fs: float, edge_s: float = 60.0) -> dict:
    """Mean raw F in the first and last ``edge_s`` of good frames, and the % change.

    For recordings shorter than ``4 * edge_s`` the edges shrink to a quarter of
    the recording each; the edge length actually used is returned as ``first_s``.
    """
    masked = baseline.mask_bad_frames(F, bad)
    n = max(1, min(int(round(edge_s * fs)), len(F) // 4))
    first, last = np.nanmean(masked[:n]), np.nanmean(masked[-n:])
    return dict(first_s=n / fs, F_start=float(first), F_end=float(last),
                percent_change=float(100 * (last - first) / first))


def compare_baselines(F: np.ndarray, bad: np.ndarray, fs: float,
                      baselines: dict[str, np.ndarray],
                      slow_timescale_s: float = 10.0, n_blocks: int = 5) -> dict:
    """dF/F, slow/fast parts, and preservation metrics for every baseline of one trace.

    The first key of ``baselines`` is the reference.
    """
    dff = {k: baseline.delta_f_over_f(F, F0, bad) for k, F0 in baselines.items()}
    return dict(compare_dff(dff, fs, slow_timescale_s, n_blocks),
                time_s=np.arange(len(F)) / fs, F=baseline.mask_bad_frames(F, bad),
                baselines=baselines, drift=raw_drift(F, bad, fs))


def compare_dff(dff: dict[str, np.ndarray], fs: float, slow_timescale_s: float = 10.0,
                n_blocks: int = 5) -> dict:
    """Slow/fast split and preservation metrics for dF/F traces of one recording.

    ``dff`` maps method label -> trace; the first key is the reference. All
    metrics use the same samples: frames where every method's dF/F is finite.
    Amplitudes are in percentage points (pp) of dF/F.
    """
    labels = list(dff)
    reference = labels[0]
    common = np.logical_and.reduce([np.isfinite(v) for v in dff.values()])
    if common.sum() < 3:
        raise ValueError("fewer than 3 frames are valid under every baseline")
    on_common = {k: np.where(common, v, np.nan) for k, v in dff.items()}
    slow = {k: slow_component(v, fs, slow_timescale_s) for k, v in on_common.items()}
    fast = {k: on_common[k] - slow[k] for k in labels}

    gains = pd.DataFrame({k: block_gains(fast[k], fast[reference], n_blocks) for k in labels})
    gains.index.name = "block"

    def row(k: str) -> dict:
        pct = on_common[k][common] * 100
        g = gains[k]
        return dict(
            method=k,
            median_dff_pct=float(np.median(pct)),
            p5_to_p95_pp=float(np.subtract(*np.percentile(pct, [95, 5]))),
            slow_p5_to_p95_pp=100 * _spread(slow[k]),
            fast_std_pp=100 * _std(fast[k]),
            correlation_with_current=_corr(on_common[k], on_common[reference]),
            slow_correlation_with_current=_corr(slow[k], slow[reference]),
            slow_amplitude_ratio=_spread(slow[k]) / _spread(slow[reference]),
            fast_amplitude_ratio=_std(fast[k]) / _std(fast[reference]),
            median_block_gain=float(g.median()),
            block_gain_cv=float(g.std(ddof=1) / g.mean()) if n_blocks > 1 else np.nan,
            last_over_first_block_gain=float(g.iloc[-1] / g.iloc[0]),
            shared_frames=int(common.sum()),
        )

    return dict(dff=dff, slow=slow, fast=fast, common=common,
                metrics=pd.DataFrame([row(k) for k in labels]).set_index("method"),
                block_gains=gains, slow_timescale_s=slow_timescale_s)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    use = np.isfinite(a) & np.isfinite(b)
    if use.sum() < 3 or np.std(a[use]) == 0 or np.std(b[use]) == 0:
        return np.nan
    return float(np.corrcoef(a[use], b[use])[0, 1])


def _spread(x: np.ndarray) -> float:
    """5th-95th percentile range: an outlier-robust amplitude."""
    return float(np.subtract(*np.nanpercentile(x, [95, 5])))


def _std(x: np.ndarray) -> float:
    return float(np.nanstd(x))
