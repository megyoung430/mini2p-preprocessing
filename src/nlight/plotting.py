"""Plots for nLightG2 traces. Each panel function draws into a given ``ax``
(creating one only if none is passed), so panels compose into larger figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelsize": 12, "axes.titlesize": 14,
}


def use_lab_style() -> None:
    """Apply the shared nLightG2 typography to the current matplotlib session."""
    mpl.rcParams.update(PLOT_STYLE)


def _ax(ax: plt.Axes | None) -> plt.Axes:
    return plt.subplots(figsize=(14, 3))[1] if ax is None else ax


def _colors(n: int) -> list:
    return [plt.get_cmap("tab10")(i % 10) for i in range(n)]


def plot_raw_fov(time_s: np.ndarray, F: np.ndarray, ax: plt.Axes | None = None,
                 slow: np.ndarray | None = None, drift: dict | None = None) -> plt.Axes:
    """Raw whole-FOV fluorescence before dF/F (bad frames already NaN)."""
    ax = _ax(ax)
    minutes = time_s / 60
    ax.plot(minutes, F, color="black", lw=0.5, label="Raw F (mean over valid crop)")
    if slow is not None:
        ax.plot(minutes, slow, color="tab:orange", lw=1.2, label="Slow component")
    title = "Raw whole-FOV fluorescence (registered, before ΔF/F)"
    if drift is not None:
        title += (f" — first vs last {drift['first_s']:g} s: "
                  f"{drift['percent_change']:+.1f}%")
    ax.set(title=title, ylabel="F (a.u.)")
    ax.legend(fontsize=10, loc="upper right")
    return ax


def plot_baselines(time_s: np.ndarray, F: np.ndarray, baselines: dict[str, np.ndarray],
                   ax: plt.Axes | None = None) -> plt.Axes:
    """Every candidate F0 drawn over the raw trace."""
    ax = _ax(ax)
    minutes = time_s / 60
    ax.plot(minutes, F, color="0.8", lw=0.4, label="Raw F")
    for color, (label, F0) in zip(_colors(len(baselines)), baselines.items()):
        ax.plot(minutes, F0, color=color, lw=1.2, label=label)
    ax.set(title="Baseline definitions", ylabel="F₀ (a.u.)")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    return ax


def plot_dff(time_s: np.ndarray, dff: np.ndarray, ax: plt.Axes | None = None, *,
             label: str = "", color=None, slow: np.ndarray | None = None) -> plt.Axes:
    ax = _ax(ax)
    minutes = time_s / 60
    ax.plot(minutes, dff * 100, color=color, lw=0.5, alpha=0.8)
    if slow is not None:
        ax.plot(minutes, slow * 100, color="black", lw=1)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set(title=label, ylabel="ΔF/F₀ (%)")
    return ax


def plot_slow_overlay(time_s: np.ndarray, slow: dict[str, np.ndarray],
                      ax: plt.Axes | None = None, timescale_s: float | None = None) -> plt.Axes:
    """Slow dF/F components of every method on one axis: preserved slow
    fluctuations show up as overlapping curves."""
    ax = _ax(ax)
    for color, (label, trace) in zip(_colors(len(slow)), slow.items()):
        ax.plot(time_s / 60, trace * 100, color=color, lw=1.2, label=label)
    ax.axhline(0, color="0.5", lw=0.5)
    span = f" ({timescale_s:g} s rolling mean)" if timescale_s else ""
    ax.set(title=f"Slow ΔF/F components{span}", ylabel="ΔF/F₀ (%)")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    return ax


def plot_block_gains(gains: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Fast-component gain vs the current method, per recording block. Flat
    lines mean relative transient amplitudes are preserved."""
    ax = plt.subplots(figsize=(6, 4))[1] if ax is None else ax
    blocks = np.arange(1, len(gains) + 1)
    for color, label in zip(_colors(gains.shape[1]), gains.columns):
        ax.plot(blocks, gains[label], marker="o", color=color, label=label)
    ax.axhline(1, color="0.5", lw=0.5, ls="--")
    ax.set(xticks=blocks, xlabel="Recording block (contiguous, equal length)",
           ylabel="Gain of fast ΔF/F vs current", title="Relative-amplitude preservation")
    ax.legend(fontsize=9)
    return ax


def baseline_comparison_figure(result: dict, title: str = "") -> plt.Figure:
    """Raw trace, baselines, each method's dF/F (shared y), slow overlay, block gains."""
    labels = list(result["dff"])
    n = len(labels)
    fig = plt.figure(figsize=(15, 3 * (n + 3)), constrained_layout=True)
    grid = fig.add_gridspec(n + 3, 3)
    axes = [fig.add_subplot(grid[i, :]) for i in range(n + 2)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    t, F = result["time_s"], result["F"]
    raw_slow = pd.Series(F).rolling(
        max(3, int(round(result["slow_timescale_s"] / np.median(np.diff(t))))),
        center=True, min_periods=1).mean().to_numpy()
    plot_raw_fov(t, F, axes[0], slow=raw_slow, drift=result["drift"])
    plot_baselines(t, F, result["baselines"], axes[1])

    pooled = np.concatenate([v[np.isfinite(v)] * 100 for v in result["dff"].values()])
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    for i, (label, color) in enumerate(zip(labels, _colors(n))):
        ax = axes[i + 2]
        plot_dff(t, result["dff"][label], ax, label=label, color=color,
                 slow=result["slow"][label])
        ax.set_ylim(lo - pad, hi + pad)
    plot_slow_overlay(t, result["slow"], fig.add_subplot(grid[n + 2, :2], sharex=axes[0]),
                      result["slow_timescale_s"])
    plot_block_gains(result["block_gains"], fig.add_subplot(grid[n + 2, 2]))
    fig.axes[-2].set_xlabel("Time (minutes)")
    fig.suptitle(title + "\nShared ΔF/F axes across methods (0.5–99.5th pct); black = slow component")
    return fig


# --- cross-recording comparison -------------------------------------------

def _short(label: str) -> str:
    """Compact x-tick label for a baseline method."""
    return (label.replace("Current: ", "current\n").replace("Fixed full-recording ", "fixed\n")
            .replace(" s rolling ", " s\n"))


def plot_relative_raw(summaries: dict[str, dict], ax: plt.Axes | None = None,
                      bin_s: float = 1.0) -> plt.Axes:
    """Raw whole-FOV F (% of first minute, plane-averaged) for each recording."""
    from .recording_comparison import bin_trace
    ax = _ax(ax)
    for color, (name, s) in zip(_colors(len(summaries)), summaries.items()):
        t, x = bin_trace(s["time_s"], s["raw_relative_pct"], bin_s)
        ax.plot(t / 60, x, color=color, lw=1,
                label=f"{name} ({s['n_planes']} plane{'s' * (s['n_planes'] > 1)}, "
                      f"{s['raw_drift_pct']:+.0f}% start→end)")
    ax.axhline(100, color="0.5", lw=0.5)
    ax.set(title="Raw whole-FOV F before ΔF/F (% of first minute)", ylabel="F (% of start)")
    ax.legend(fontsize=9)
    return ax


def plot_recordings_dff(summaries: dict[str, dict], method: str, ax: plt.Axes | None = None,
                        bin_s: float = 1.0) -> plt.Axes:
    """One baseline method's plane-mean dF/F for every recording, on a common time grid."""
    from .recording_comparison import bin_trace
    ax = _ax(ax)
    for color, (name, s) in zip(_colors(len(summaries)), summaries.items()):
        if method not in s["dff"]:
            continue
        t, x = bin_trace(s["time_s"], s["dff"][method] * 100, bin_s)
        ax.plot(t / 60, x, color=color, lw=0.8, label=name)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set(title=f"{method} — plane-mean ΔF/F, {bin_s:g} s bins", ylabel="ΔF/F₀ (%)")
    ax.legend(fontsize=9)
    return ax


def plot_metric_by_method(table: pd.DataFrame, metric: str, ax: plt.Axes | None = None,
                          ylabel: str | None = None) -> plt.Axes:
    """``metric`` for each recording (lines) across baseline methods (x). Parallel
    lines mean the between-recording difference does not depend on the baseline."""
    ax = plt.subplots(figsize=(6, 4))[1] if ax is None else ax
    methods = list(dict.fromkeys(table.index.get_level_values("method")))
    recordings = list(dict.fromkeys(table.index.get_level_values("recording")))
    # keep the table's order so colours match the trace panels
    wide = table[metric].unstack("recording").reindex(index=methods, columns=recordings)
    x = np.arange(len(methods))
    for color, name in zip(_colors(wide.shape[1]), wide.columns):
        ax.plot(x, wide[name], marker="o", color=color, label=name)
    ax.set_xticks(x, [_short(m) for m in methods], fontsize=9)
    ax.set(ylabel=ylabel or metric, title=ylabel or metric)
    ax.legend(fontsize=8)
    return ax


def recording_comparison_figure(summaries: dict[str, dict], table: pd.DataFrame,
                                title: str = "", bin_s: float = 1.0) -> plt.Figure:
    """Raw-F drift, each method's dF/F overlaid across recordings, and metric-by-method panels."""
    methods = list(dict.fromkeys(table.index.get_level_values("method")))
    n = len(methods)
    fig = plt.figure(figsize=(16, 2.8 * (n + 1) + 4.5), constrained_layout=True)
    grid = fig.add_gridspec(n + 2, 3, height_ratios=[1] * (n + 1) + [1.6])
    top = fig.add_subplot(grid[0, :])
    plot_relative_raw(summaries, top, bin_s)
    for i, method in enumerate(methods):
        plot_recordings_dff(summaries, method, fig.add_subplot(grid[i + 1, :], sharex=top), bin_s)
    fig.axes[-1].set_xlabel("Time from recording start (minutes)")
    for j, (metric, label) in enumerate([("p5_to_p95_pp", "ΔF/F 5–95% range (pp)"),
                                         ("slow_p5_to_p95_pp", "Slow-component 5–95% range (pp)"),
                                         ("fast_std_pp", "Fast-component SD (pp)")]):
        plot_metric_by_method(table, metric, fig.add_subplot(grid[n + 1, j]), label)
    fig.suptitle(title + "\nSame baseline definitions for every recording; "
                 "multi-plane ΔF/F computed per plane, then averaged")
    return fig
