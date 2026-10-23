"""Plane-level entry points that join I/O, analysis and plotting.

Notebooks and command-line scripts call these; the logic lives in the pure
modules they compose.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import baseline_comparison, plotting, recording_comparison, suite2p_io


def whole_fov_baseline_check(export_dir: Path, fixed_percentiles=(10,),
                             long_windows_s=(180, 300, 600), slow_timescale_s: float = 10.0,
                             n_blocks: int = 5) -> dict:
    """Baseline comparison for one plane's saved whole-FOV export (a folder
    holding ``whole_image_roi_dff.npz``). Read-only.

    Fails if recomputing the current baseline does not reproduce the saved
    dF/F, which would mean the metadata no longer describes the saved file.
    """
    arrays, meta = suite2p_io.load_whole_fov_export(export_dir)
    F, bad, t = arrays["F"].astype(float), arrays["badframes"].astype(bool), arrays["time_s"]
    fs = float(meta["fs_hz"]) if "fs_hz" in meta else 1 / np.median(np.diff(t))
    # Older exports record only the window in frames; both give the same window_frames().
    window_s = meta.get("baseline_window_s_requested", meta["baseline_window_frames"] / fs)
    baselines = baseline_comparison.candidate_baselines(
        F, bad, fs, window_s, meta["baseline_percentile"], fixed_percentiles, long_windows_s)
    result = baseline_comparison.compare_baselines(F, bad, fs, baselines,
                                                   slow_timescale_s, n_blocks)
    reference = next(iter(result["dff"].values()))
    np.testing.assert_allclose(reference, arrays["dff"], rtol=1e-5, atol=1e-7, equal_nan=True,
                               err_msg=f"{export_dir}: current baseline does not reproduce saved dff")
    return result


def save_baseline_check(export_dir: Path, **kwargs) -> Path:
    """Run :func:`whole_fov_baseline_check`; save figure + CSVs next to the export."""
    export_dir = Path(export_dir)
    out_dir = export_dir / "whole_fov_baseline_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = whole_fov_baseline_check(export_dir, **kwargs)
    fig = plotting.baseline_comparison_figure(
        result, title=f"{export_dir.parent.name} · {export_dir.name} · whole-FOV baseline sensitivity")
    fig.savefig(out_dir / "whole_fov_baseline_check.png", dpi=150)
    plt.close(fig)
    result["metrics"].to_csv(out_dir / "metrics.csv")
    result["block_gains"].to_csv(out_dir / "block_gains.csv")
    return out_dir


def load_recording(export_root: Path) -> list[recording_comparison.PlaneTrace]:
    """Raw whole-FOV traces of every plane exported under ``export_root``."""
    def plane_trace(plane_dir: Path) -> recording_comparison.PlaneTrace:
        arrays, meta = suite2p_io.load_whole_fov_export(plane_dir)
        t = arrays["time_s"]
        fs = float(meta["fs_hz"]) if "fs_hz" in meta else 1 / np.median(np.diff(t))
        return recording_comparison.PlaneTrace(arrays["F"].astype(float),
                                               arrays["badframes"].astype(bool), fs)

    return [plane_trace(d) for d in
            suite2p_io.plane_dirs(export_root, marker=suite2p_io.WHOLE_FOV_EXPORT + ".npz")]


def compare_recordings(export_roots: dict[str, Path],
                       spec: recording_comparison.BaselineSpec = recording_comparison.BaselineSpec(),
                       slow_timescale_s: float = 10.0, n_blocks: int = 5) -> tuple[dict, pd.DataFrame]:
    """Summaries and comparison table for named recordings. Read-only."""
    summaries = {name: recording_comparison.summarize_recording(
                     load_recording(root), spec, slow_timescale_s, n_blocks)
                 for name, root in export_roots.items()}
    return summaries, recording_comparison.comparison_table(summaries)


def save_recording_comparison(export_roots: dict[str, Path], out_dir: Path, title: str = "",
                              **kwargs) -> Path:
    """Run :func:`compare_recordings`; save figure, table and pairwise differences."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, table = compare_recordings(export_roots, **kwargs)
    fig = plotting.recording_comparison_figure(summaries, table, title=title)
    fig.savefig(out_dir / "recording_comparison.png", dpi=150)
    plt.close(fig)
    table.to_csv(out_dir / "recording_comparison.csv")
    names = list(export_roots)
    if len(names) == 2:
        recording_comparison.paired_difference(table, *names).to_csv(out_dir / "paired_difference.csv")
    return out_dir
