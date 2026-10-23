"""Panels for the whole-recording and event-aligned notebooks. Each draws into a
given ``ax`` (or makes one) so notebooks can lay them out per baseline method."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import _colors, _short


def _ax(ax, size=(8, 3.5)):
    return plt.subplots(figsize=size)[1] if ax is None else ax


def bin_columns(x: np.ndarray, fs: float, bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Average consecutive frames into ``bin_s`` bins (display only); returns (times, binned)."""
    n = max(1, int(round(bin_s * fs)))
    usable = (x.shape[-1] // n) * n
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            binned = np.nanmean(x[..., :usable].reshape(*x.shape[:-1], -1, n), axis=-1)
    return (np.arange(binned.shape[-1]) + 0.5) * n / fs, binned


def robust_limit(arrays, percentile: float = 98) -> float:
    values = np.concatenate([np.abs(np.asarray(a)[np.isfinite(a)]).ravel() for a in arrays])
    if not values.size:
        raise ValueError("no finite values to scale")
    return max(float(np.percentile(values, percentile)), 1e-6)


def plot_tile_heatmap(dff: np.ndarray, fs: float, ax=None, *, title: str = "",
                      vmax: float | None = None, bin_s: float = 1.0, t0_s: float = 0.0):
    """All retained tiles (grid order) x time, % dF/F, viridis from 0; gray = missing.

    ``t0_s`` is the recording time of the first column, for zoomed windows."""
    ax = _ax(ax, (14, 4))
    t, binned = bin_columns(dff * 100, fs, bin_s)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(binned), aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=vmax or robust_limit([binned], 99),
                   extent=[t0_s / 60, (t0_s + t[-1]) / 60 + bin_s / 120, len(binned) - 0.5, -0.5])
    ax.set(title=title, xlabel="Time (minutes)", ylabel="Tile (grid order)")
    plt.colorbar(im, ax=ax, label="ΔF/F₀ (%)", extend="both")
    return ax


def plot_event_mean(relative_s, mean, sem, ax=None, *, label="", color=None):
    """Trial-mean whole-FOV dF/F (%) with SEM across events."""
    ax = _ax(ax, (5, 3.5))
    ax.fill_between(relative_s, (mean - sem) * 100, (mean + sem) * 100, color=color, alpha=0.2)
    ax.plot(relative_s, mean * 100, color=color, lw=1.5, label=label)
    ax.axvline(0, color="0.5", ls="--", lw=0.8)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set(xlabel="Time from onset (s)", ylabel="ΔF/F₀ (%)")
    return ax


def plot_event_heatmap(relative_s, mean_tiles, ax=None, *, title="", limit=None):
    """Tiles x time trial means, % dF/F, symmetric blue-white-red."""
    ax = _ax(ax, (5, 4))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")
    dt = np.median(np.diff(relative_s))
    limit = limit or robust_limit([mean_tiles * 100])
    im = ax.imshow(np.ma.masked_invalid(mean_tiles * 100), aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=-limit, vmax=limit,
                   extent=[relative_s[0] - dt / 2, relative_s[-1] + dt / 2, len(mean_tiles) - 0.5, -0.5])
    ax.axvline(0, color="0.3", ls="--", lw=0.8)
    ax.set(title=title, xlabel="Time from onset (s)", ylabel="Tile")
    return im


def plot_lag_curves(lags_s, curves: dict[str, np.ndarray], ax=None, *, title=""):
    """Whole-FOV correlation vs lag, one line per baseline method."""
    ax = _ax(ax, (6, 4))
    for color, (label, r) in zip(_colors(len(curves)), curves.items()):
        ax.plot(lags_s, r, color=color, lw=1.5, label=label)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    ax.set(title=title, xlabel="Lag (s); positive = fluorescence follows", ylabel="Pearson r")
    ax.legend(fontsize=8)
    return ax


def plot_tuning(curves: dict[str, pd.DataFrame], ax=None, *, xlabel="", title=""):
    """Mean whole-FOV dF/F per bin of a state variable, one line per baseline method."""
    ax = _ax(ax, (6, 4))
    for color, (label, df) in zip(_colors(len(curves)), curves.items()):
        ax.fill_between(df.center, (df["mean"] - df["sem"]) * 100, (df["mean"] + df["sem"]) * 100,
                        color=color, alpha=0.15)
        ax.plot(df.center, df["mean"] * 100, "o-", ms=3, color=color, label=label)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel="ΔF/F₀ (%)")
    ax.legend(fontsize=8)
    return ax


def plot_state_difference(table: pd.DataFrame, ax=None, *, title="Running − stationary"):
    """Running-minus-stationary mean dF/F per baseline method."""
    ax = _ax(ax, (6, 4))
    x = np.arange(len(table))
    ax.bar(x, table.running_minus_stationary_pp, color=_colors(len(table)))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x, [_short(m) for m in table.index], fontsize=9)
    ax.set(title=title, ylabel="Δ mean ΔF/F₀ (pp)")
    return ax


def plot_speed_distribution(speed, threshold, ax=None):
    """Histogram of running speed with the running threshold, to sanity-check it."""
    ax = _ax(ax, (6, 3.5))
    finite = speed[np.isfinite(speed)]
    ax.hist(finite, bins=100, color="teal", alpha=0.8)
    ax.axvline(threshold, color="darkorange", ls="--", label=f"threshold = {threshold:g}")
    ax.set(yscale="log", xlabel="Speed (table units)", ylabel="Frames", title="Running speed")
    ax.legend()
    return ax


def plot_running_kernels(lags_s, kernels: dict[str, np.ndarray], ax=None, *,
                         title="Fitted running kernel"):
    """Speed coefficients vs lag per baseline: how dF/F follows running speed."""
    ax = _ax(ax, (6, 4))
    for color, (label, k) in zip(_colors(len(kernels)), kernels.items()):
        ax.plot(lags_s, np.asarray(k) * 100, color=color, lw=1.5, label=label)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    ax.set(title=title, xlabel="Lag (s); positive = fluorescence follows speed",
           ylabel="ΔF/F₀ (pp) per speed unit")
    ax.legend(fontsize=8)
    return ax
