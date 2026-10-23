"""Per-baseline analyses for one recording.

Each function takes ``variants`` — dF/F traces keyed by baseline method (whole
FOV: frames; tiles: tiles x frames) — and returns results keyed the same way,
so every analysis can be read under every baseline definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import events, state


def event_responses(variants: dict[str, np.ndarray], onsets: dict[str, np.ndarray], fs: float,
                    pre_s: float = 2.0, post_s: float = 20.0, prestim_start_s: float = -2.0,
                    min_valid_fraction: float = 0.8) -> dict[str, dict[str, dict]]:
    """``result[event][method]``: trial mean/SEM, original and prestimulus-subtracted.

    Original and subtracted means use the same eligible trial/trace pairs.
    """
    offsets, relative_s = events.window_offsets(fs, pre_s, post_s)

    def one(x: np.ndarray, event_onsets: np.ndarray) -> dict:
        trials, used = events.aligned_trials(x, event_onsets, offsets)
        original, centred = events.subtract_prestimulus(trials, relative_s, prestim_start_s,
                                                        min_valid_fraction)
        mean, sem, n = events.mean_sem(original)
        c_mean, c_sem, _ = events.mean_sem(centred)
        squeeze = (lambda a: a[0]) if np.ndim(x) == 1 else (lambda a: a)
        return dict(relative_s=relative_s, mean=squeeze(mean), sem=squeeze(sem),
                    centred_mean=squeeze(c_mean), centred_sem=squeeze(c_sem),
                    n_trials=squeeze(n), onsets=used, n_detected=len(event_onsets))

    return {name: {m: one(x, o) for m, x in variants.items()} for name, o in onsets.items()}


def event_summary(responses: dict, window_s: tuple[float, float] = (0, 5)) -> pd.DataFrame:
    """Mean post-onset response (%, original and prestimulus-subtracted) per event x method."""
    def row(event: str, method: str, r: dict) -> dict:
        use = (r["relative_s"] >= window_s[0]) & (r["relative_s"] <= window_s[1])
        return dict(event=event, method=method, events_used=len(r["onsets"]),
                    mean_response_pct=100 * float(np.nanmean(r["mean"][..., use])),
                    centred_response_pp=100 * float(np.nanmean(r["centred_mean"][..., use])))

    return pd.DataFrame([row(e, m, r) for e, by_method in responses.items()
                         for m, r in by_method.items()]).set_index(["event", "method"])


def lag_curves(variants: dict[str, np.ndarray], predictor: np.ndarray, fs: float,
               max_lag_s: float = 3.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Whole-FOV correlation with ``predictor`` from ``-max_lag_s`` to ``+max_lag_s``."""
    shifts = np.arange(-int(round(max_lag_s * fs)), int(round(max_lag_s * fs)) + 1)
    return shifts / fs, {m: state.lag_correlations(x, predictor, shifts)[0]
                         for m, x in variants.items()}


def tile_correlations(variants: dict[str, np.ndarray], predictor: np.ndarray) -> dict[str, np.ndarray]:
    """Zero-lag Pearson r of every tile with ``predictor``, per method."""
    return {m: state.lag_correlations(x, predictor, np.array([0]))[:, 0] for m, x in variants.items()}


def tuning_curves(variants: dict[str, np.ndarray], x: np.ndarray, n_bins: int = 10,
                  min_samples: int = 30) -> dict[str, pd.DataFrame]:
    """Whole-FOV dF/F binned by ``x`` (shared equal-width bins over finite ``x``)."""
    finite = x[np.isfinite(x)]
    if len(finite) < min_samples or np.ptp(finite) <= 0:
        raise ValueError("too few or constant samples of the state variable")
    edges = np.linspace(finite.min(), finite.max(), n_bins + 1)
    return {m: state.tuning_curve(v, x, edges, min_samples) for m, v in variants.items()}


def grid_map(values: np.ndarray, tile_ids: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Put per-tile values back on the tile grid (NaN for excluded tiles)."""
    rows, cols = grid_shape
    out = np.full(rows * cols, np.nan)
    out[tile_ids] = values
    return out.reshape(rows, cols)


def regress_out_running(variants: dict[str, np.ndarray], speed: np.ndarray, running: np.ndarray,
                        fs: float, min_lag_s: float = -1.0, max_lag_s: float = 5.0,
                        step_s: float = 0.25, cv_folds: int = 5, cv_gap_s: float = 30.0) -> dict:
    """Running-regressed traces per method, with variance explained and fitted kernels.

    ``variants`` values are whole-FOV traces (frames) or tiles x frames. Returns
    ``residual`` (same layout), ``table`` (whole-FOV only: in-sample and held-out
    R²) and ``kernels`` (speed coefficients vs lag; tiles: median across tiles).
    """
    from . import regression

    X, lags_s = regression.running_design(speed, running, fs, min_lag_s, max_lag_s, step_s)
    residual, rows, kernels = {}, [], {}
    for method, x in variants.items():
        y = x.T if x.ndim == 2 else x
        r = regression.regress_out(y, X)
        residual[method] = r["residual"].T if x.ndim == 2 else r["residual"]
        speed_beta = r["beta"][:-1]
        kernels[method] = np.median(speed_beta, axis=1) if x.ndim == 2 else speed_beta
        if x.ndim == 1:
            rows.append(dict(method=method, r2_in_sample=float(r["r2"]),
                             r2_held_out=regression.blocked_cv_r2(y, X, r["rows"], fs, cv_folds, cv_gap_s),
                             running_indicator_beta_pp=100 * float(r["beta"][-1]),
                             frames_used=int(r["rows"].sum())))
        else:
            rows.append(dict(method=method, median_tile_r2=float(np.median(r["r2"]))))
    return dict(residual=residual, table=pd.DataFrame(rows).set_index("method"),
                kernels=kernels, lags_s=lags_s)


def running_regression(variants: dict[str, np.ndarray], speed: np.ndarray, fs: float,
                       model) -> dict:
    """:func:`regress_out_running` with a :class:`regression.RunningModel`; adds ``running``.

    Residual frames are NaN where speed is missing and within the lag range of
    the recording edges, because the lagged design is undefined there.
    """
    running = state.running_mask(speed, fs, model.threshold, model.min_bout_s)
    out = regress_out_running(variants, speed, running, fs, model.min_lag_s, model.max_lag_s,
                              model.step_s, model.cv_folds, model.cv_gap_s)
    return dict(out, running=running)
