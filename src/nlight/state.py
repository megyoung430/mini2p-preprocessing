"""dF/F against arousal-related state: pupil (correlation, lag, tuning) and running.

All descriptive and within one recording. Consecutive frames are correlated,
so frame counts overstate independent samples; no p-values are computed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lag_correlations(traces: np.ndarray, predictor: np.ndarray, shifts: np.ndarray,
                     minimum: int = 30) -> np.ndarray:
    """Pearson r of each trace with ``predictor`` at each frame shift, ``traces x shifts``.

    Positive shift: fluorescence follows the predictor, corr(F[t + lag], P[t]).
    Pairwise-finite samples; no wrap-around. NaN with fewer than ``minimum`` pairs.
    """
    traces = np.atleast_2d(traces)

    def at(shift: int) -> np.ndarray:
        x = traces[:, shift:] if shift > 0 else traces[:, :shift] if shift < 0 else traces
        y = predictor[:-shift] if shift > 0 else predictor[-shift:] if shift < 0 else predictor
        good = np.isfinite(x) & np.isfinite(y)[None, :]
        n = good.sum(axis=1)
        xs, ys = np.where(good, x, 0.0), np.where(good, y[None, :], 0.0)
        xc = np.where(good, xs - xs.sum(1, keepdims=True) / np.maximum(n, 1)[:, None], 0.0)
        yc = np.where(good, ys - ys.sum(1, keepdims=True) / np.maximum(n, 1)[:, None], 0.0)
        den = np.sqrt((xc * xc).sum(1) * (yc * yc).sum(1))
        r = np.full(len(x), np.nan)
        np.divide((xc * yc).sum(1), den, out=r, where=(n >= minimum) & (den > 0))
        return r

    return np.column_stack([at(int(s)) for s in shifts])


def smooth_with_derivative(x: np.ndarray, fs: float, window_s: float
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay smoothed signal and derivative (units/s), within valid segments only.

    Gaps are never interpolated across; filter-edge samples of each segment
    are discarded. ``window_s == 0`` returns the raw signal and a central difference.
    """
    from scipy.signal import savgol_filter

    smooth, deriv = np.full(len(x), np.nan), np.full(len(x), np.nan)
    window = max(3, int(round(window_s * fs)) | 1) if window_s else 0
    edges = np.diff(np.r_[False, np.isfinite(x), False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        seg = x[start:stop]
        if window:
            if len(seg) < window:
                continue
            half = window // 2
            smooth[start + half:stop - half] = savgol_filter(seg, window, 2)[half:len(seg) - half]
            deriv[start + half:stop - half] = savgol_filter(
                seg, window, 2, deriv=1, delta=1 / fs)[half:len(seg) - half]
        elif len(seg) >= 3:
            smooth[start:stop] = seg
            deriv[start + 1:stop - 1] = np.gradient(seg, 1 / fs)[1:-1]
    return smooth, deriv


def tuning_curve(y: np.ndarray, x: np.ndarray, edges: np.ndarray,
                 min_samples: int = 30) -> pd.DataFrame:
    """Mean/SEM of ``y`` in bins of ``x``; bins with fewer than ``min_samples`` frames are NaN."""
    valid = np.isfinite(y) & np.isfinite(x)
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)

    def stats(b: int) -> dict:
        v = y[valid & (which == b)]
        ok = len(v) >= min_samples
        return dict(bin_left=edges[b], bin_right=edges[b + 1], center=(edges[b] + edges[b + 1]) / 2,
                    n_frames=len(v), mean=float(v.mean()) if ok else np.nan,
                    sem=float(v.std(ddof=1) / np.sqrt(len(v))) if ok else np.nan)

    return pd.DataFrame([stats(b) for b in range(len(edges) - 1)])


def running_mask(speed: np.ndarray, fs: float, threshold: float,
                 min_bout_s: float = 1.0) -> np.ndarray:
    """True while running: speed above ``threshold`` for at least ``min_bout_s``.

    Shorter supra-threshold bouts count as stationary; NaN speed is not running
    (callers drop those frames via ``np.isfinite(speed)``).
    """
    above = np.nan_to_num(speed, nan=-np.inf) > threshold
    edges = np.diff(np.r_[False, above, False].astype(int))
    running = np.zeros(len(speed), bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (stop - start) / fs >= min_bout_s:
            running[start:stop] = True
    return running


def state_onsets(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames where ``mask`` turns on and where it turns off."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    on, off = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return on[on > 0], off[off < len(mask)]


def bout_durations(mask: np.ndarray, fs: float) -> np.ndarray:
    """Length (s) of every contiguous True stretch, including ones cut by the recording edges."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / fs


def state_summary(dff: dict[str, np.ndarray], running: np.ndarray,
                  valid: np.ndarray) -> pd.DataFrame:
    """Mean dF/F (%) while running vs stationary, per baseline method."""
    def row(label: str, trace: np.ndarray) -> dict:
        ok = valid & np.isfinite(trace)
        run, still = trace[ok & running] * 100, trace[ok & ~running] * 100
        return dict(method=label, running_mean_pct=run.mean() if len(run) else np.nan,
                    stationary_mean_pct=still.mean() if len(still) else np.nan,
                    running_minus_stationary_pp=(run.mean() - still.mean())
                    if len(run) and len(still) else np.nan,
                    running_frames=len(run), stationary_frames=len(still))

    return pd.DataFrame([row(k, v) for k, v in dff.items()]).set_index("method")
