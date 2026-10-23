"""Regress running out of dF/F with a lagged linear model.

The design holds running speed at a set of lags (positive lag: fluorescence
follows speed) plus a running/stationary indicator. Least squares on frames
where the design and the trace are finite. Removing the fitted running part
keeps the trace's mean, so dF/F levels stay interpretable.

Pupil and running are correlated. What survives regression is the part of the
pupil relationship that running does not explain linearly at these lags, not
"the pupil effect": shared variance goes to running.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunningModel:
    """How running is defined and modelled; shared by every notebook that uses it."""
    threshold: float = 2.0      # speed (table units) above which the animal is running
    min_bout_s: float = 1.0     # shorter supra-threshold bouts count as stationary
    min_lag_s: float = -1.0     # speed lags in the design; positive = fluorescence follows speed
    max_lag_s: float = 5.0
    step_s: float = 0.25
    cv_folds: int = 5           # contiguous held-out blocks for R²
    cv_gap_s: float = 30.0      # excluded from training either side of a test block


def lagged(x: np.ndarray, fs: float, lags_s: np.ndarray) -> np.ndarray:
    """``frames x lags``: column j is ``x[t - lag_j]`` (NaN where out of range)."""
    shifts = np.rint(np.asarray(lags_s) * fs).astype(int)
    out = np.full((len(x), len(shifts)), np.nan)
    for j, s in enumerate(shifts):
        if s >= 0:
            out[s:, j] = x[:len(x) - s]
        else:
            out[:s, j] = x[-s:]
    return out


def running_design(speed: np.ndarray, running: np.ndarray, fs: float,
                   min_lag_s: float = -1.0, max_lag_s: float = 5.0,
                   step_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """``(design, lags_s)``: lagged speed columns, then the running indicator. No intercept."""
    if max_lag_s < min_lag_s or step_s <= 0:
        raise ValueError("need min_lag_s <= max_lag_s and step_s > 0")
    lags_s = np.arange(min_lag_s, max_lag_s + step_s / 2, step_s)
    return np.column_stack([lagged(speed, fs, lags_s), running.astype(float)]), lags_s


def fit(y: np.ndarray, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Least-squares coefficients ``(1 + columns) x traces`` (intercept first) on ``rows``."""
    A = np.column_stack([np.ones(rows.sum()), X[rows]])
    return np.linalg.lstsq(A, y[rows], rcond=None)[0]


def common_rows(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Frames where the design and every trace are finite."""
    finite_y = np.isfinite(y) if y.ndim == 1 else np.isfinite(y).all(axis=1)
    return finite_y & np.isfinite(X).all(axis=1)


def regress_out(y: np.ndarray, X: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """Residual = y minus the mean-centred fitted design part; NaN outside ``rows``.

    ``y`` is ``frames`` or ``frames x traces``. Returns ``residual``, ``beta``
    (without intercept), in-sample ``r2`` and ``rows``.
    """
    rows = common_rows(y, X) if rows is None else rows
    if rows.sum() <= X.shape[1] + 1:
        raise ValueError("fewer usable frames than regressors")
    coef = fit(y, X, rows)
    fitted = X[rows] @ coef[1:]
    # subtract the running part centred on its mean, so the residual keeps y's mean
    explained = np.full(y.shape, np.nan)
    explained[rows] = fitted - fitted.mean(axis=0)
    residual = np.where(rows if y.ndim == 1 else rows[:, None], y - explained, np.nan)
    yr = y[rows]
    ss_res = np.sum((yr - coef[0] - fitted) ** 2, axis=0)
    ss_tot = np.sum((yr - yr.mean(axis=0)) ** 2, axis=0)
    return dict(residual=residual, beta=coef[1:], r2=1 - ss_res / ss_tot, rows=rows)


def blocked_cv_r2(y: np.ndarray, X: np.ndarray, rows: np.ndarray, fs: float,
                  n_folds: int = 5, gap_s: float = 30.0) -> float:
    """Held-out R² over contiguous test blocks, with ``gap_s`` dropped either
    side of each block from training so lagged regressors cannot leak."""
    n = len(y)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    gap = int(round(gap_s * fs))
    pred = np.full(n, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.zeros(n, bool)
        test[a:b] = True
        train = rows.copy()
        train[max(0, a - gap):min(n, b + gap)] = False
        if train.sum() <= X.shape[1] + 1:
            raise ValueError("too few training frames in a fold; use fewer folds or a smaller gap")
        coef = fit(y, X, train)
        use = test & rows
        pred[use] = coef[0] + X[use] @ coef[1:]
    use = rows & np.isfinite(pred)
    return float(1 - np.sum((y[use] - pred[use]) ** 2) / np.sum((y[use] - y[use].mean()) ** 2))
