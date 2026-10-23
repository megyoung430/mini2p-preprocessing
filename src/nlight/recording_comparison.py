"""Compare whole-FOV dF/F between recordings under the same baseline definitions.

Each recording is reduced to one whole-FOV dF/F per baseline definition. For a
multi-plane recording, dF/F is computed per plane with that baseline and then
averaged across planes, so a bright plane does not outweigh a dim one. Every
recording gets the same definitions, set in seconds and percentiles so they
mean the same thing at different frame rates.

The question this answers is whether a difference between recordings (puff vs
no puff, passive viewing vs pre-corridor baseline) survives every reasonable
baseline choice, or depends on one. Absolute dF/F levels across different
fields of view, depths, zoom or laser power are not directly comparable; the
per-method pattern is the more robust readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import baseline, baseline_comparison


class PlaneTrace(NamedTuple):
    F: np.ndarray       # raw whole-FOV fluorescence, frames
    bad: np.ndarray     # bad-frame mask, frames
    fs: float           # per-plane frame rate (Hz)


@dataclass(frozen=True)
class BaselineSpec:
    """The baseline definitions applied identically to every recording."""
    current_window_s: float = 60.0
    current_percentile: float = 50.0
    fixed_percentiles: tuple[float, ...] = (10,)
    long_windows_s: tuple[float, ...] = (180, 300, 600)


def plane_mean_dff(planes: list[PlaneTrace], spec: BaselineSpec) -> dict[str, np.ndarray]:
    """Per-plane dF/F for every baseline in ``spec``, averaged across planes.

    A frame is NaN in the mean if it is invalid in any plane, so every plane
    contributes to every averaged sample.
    """
    _check_planes(planes)

    def plane_dff(p: PlaneTrace) -> dict[str, np.ndarray]:
        F0s = baseline_comparison.candidate_baselines(
            p.F, p.bad, p.fs, spec.current_window_s, spec.current_percentile,
            spec.fixed_percentiles, spec.long_windows_s)
        return {k: baseline.delta_f_over_f(p.F, F0, p.bad) for k, F0 in F0s.items()}

    per_plane = [plane_dff(p) for p in planes]
    labels = [k for k in per_plane[0] if all(k in d for d in per_plane)]
    return {k: np.mean([d[k] for d in per_plane], axis=0) for k in labels}


def relative_raw_F(planes: list[PlaneTrace], edge_s: float = 60.0) -> np.ndarray:
    """Raw F as % of its mean over the first ``edge_s``, averaged across planes.

    Puts recordings of different brightness on one axis to compare bleaching
    and slow drift before any baseline is applied.
    """
    _check_planes(planes)

    def relative(p: PlaneTrace) -> np.ndarray:
        masked = baseline.mask_bad_frames(p.F, p.bad)
        n = max(1, min(int(round(edge_s * p.fs)), len(masked) // 4))
        return 100 * masked / np.nanmean(masked[:n])

    return np.mean([relative(p) for p in planes], axis=0)


def summarize_recording(planes: list[PlaneTrace], spec: BaselineSpec,
                        slow_timescale_s: float = 10.0, n_blocks: int = 5) -> dict:
    """Plane-mean dF/F per baseline, its preservation metrics, and raw-F drift."""
    fs = planes[0].fs
    dff = plane_mean_dff(planes, spec)
    drifts = [baseline_comparison.raw_drift(p.F, p.bad, p.fs) for p in planes]
    return dict(baseline_comparison.compare_dff(dff, fs, slow_timescale_s, n_blocks),
                time_s=np.arange(len(planes[0].F)) / fs, fs=fs, n_planes=len(planes),
                raw_relative_pct=relative_raw_F(planes),
                raw_drift_pct=float(np.mean([d["percent_change"] for d in drifts])))


def comparison_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """One row per recording x baseline: amplitudes, slow/fast parts, raw drift."""
    def rows(name: str, s: dict) -> pd.DataFrame:
        m = s["metrics"].reset_index()
        return m.assign(recording=name, n_planes=s["n_planes"], fs_hz=s["fs"],
                        duration_min=len(s["time_s"]) / s["fs"] / 60,
                        raw_drift_pct=s["raw_drift_pct"])

    table = pd.concat([rows(n, s) for n, s in summaries.items()], ignore_index=True)
    return table.set_index(["recording", "method"])


def paired_difference(table: pd.DataFrame, a: str, b: str,
                      metrics=("median_dff_pct", "p5_to_p95_pp",
                               "slow_p5_to_p95_pp", "fast_std_pp")) -> pd.DataFrame:
    """``a`` minus ``b`` and ``a`` / ``b`` for each metric, per baseline method.

    A difference whose sign holds under every method does not hinge on the
    baseline choice. Only methods present for both recordings are compared.
    """
    left, right = table.loc[a, list(metrics)], table.loc[b, list(metrics)]
    shared = left.index.intersection(right.index, sort=False)
    left, right = left.loc[shared], right.loc[shared]
    return pd.concat({f"{a} − {b}": left - right, f"{a} / {b}": left / right}, axis=1)


def bin_trace(time_s: np.ndarray, x: np.ndarray, bin_s: float,
              min_valid_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``x`` in consecutive ``bin_s`` bins; NaN where too few samples are valid.

    Puts recordings with different frame rates on a common time grid for plotting.
    """
    if bin_s <= 0:
        raise ValueError("bin width must be positive")
    index = np.floor(time_s / bin_s).astype(int)
    frame = pd.DataFrame({"bin": index, "x": x})
    grouped = frame.groupby("bin")["x"]
    means = grouped.mean()
    valid = grouped.apply(lambda v: np.isfinite(v).mean()) >= min_valid_fraction
    return (means.index.to_numpy() + 0.5) * bin_s, means.where(valid).to_numpy()


def _check_planes(planes: list[PlaneTrace]) -> None:
    if not planes:
        raise ValueError("a recording needs at least one plane")
    lengths = {len(p.F) for p in planes}
    rates = [p.fs for p in planes]
    if len(lengths) != 1:
        raise ValueError(f"planes have different frame counts {sorted(lengths)}")
    if not np.allclose(rates, rates[0], rtol=1e-3):
        raise ValueError(f"planes have different frame rates {rates}")
