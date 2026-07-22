"""Align ScanImage frame clocks to a pyControl magnitude-bandit session.

The ScanImage HDF5 sidecar is expected to contain ``FrameClock`` and
``PyControlPulse`` waveforms sampled at the root ``samplerate``.  The pyControl
TSV must contain ``event/sync/rsync`` rows.  Output is a pickle containing the
bandit project's canonical session dictionary plus imaging-frame timing.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_BEHAVIOR_PROJECT = Path("/Users/megyoung/magnitude-bandit-analysis")


def rare_pulse_edges(signal: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Return edges entering the rarer TTL state (handles active-high/low)."""
    # Rsync pulses can occupy <1% of a recording, so ordinary 1st/99th
    # percentiles may both lie on the baseline. These DAQ TTL channels have
    # clean, bounded levels; their full-range midpoint is the reliable split.
    low, high = float(np.min(signal)), float(np.max(signal))
    threshold = float((low + high) / 2)
    above = signal > threshold
    if above.mean() <= 0.5:
        edges = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) + 1
        polarity = "rising"
    else:
        edges = np.flatnonzero(np.diff(above.astype(np.int8)) == -1) + 1
        polarity = "falling"
    return edges.astype(np.int64), polarity, threshold


def read_scanimage_clocks(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        sample_rate = float(handle.attrs["samplerate"])
        missing = {name for name in ("FrameClock", "PyControlPulse") if name not in handle}
        if missing:
            raise KeyError(f"Missing HDF5 datasets: {sorted(missing)}")
        frame_signal = handle["FrameClock"][...]
        frame_threshold = float((frame_signal.min() + frame_signal.max()) / 2)
        frame_high = frame_signal > frame_threshold
        # In this ScanImage export the clock idles low before acquisition and
        # each frame begins on a rising edge. Falling edges mark the short
        # inter-frame low interval and omit the first frame.
        frame_edges = np.flatnonzero(np.diff(frame_high.astype(np.int8)) == 1) + 1
        frame_polarity = "rising"
        sync_edges, sync_polarity, sync_threshold = rare_pulse_edges(handle["PyControlPulse"][...])
    return {
        "sample_rate_hz": sample_rate,
        "frame_samples": frame_edges,
        "frame_times_scanimage_s": frame_edges / sample_rate,
        "sync_samples": sync_edges,
        "sync_times_scanimage_s": sync_edges / sample_rate,
        "frame_polarity": frame_polarity,
        "sync_polarity": sync_polarity,
        "frame_threshold": frame_threshold,
        "sync_threshold": sync_threshold,
    }


def read_pycontrol(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, sep="\t")
    required = {"time", "type", "subtype", "content"}
    if not required.issubset(frame.columns):
        raise ValueError(f"TSV lacks required columns: {sorted(required - set(frame.columns))}")
    mask = frame["type"].eq("event") & frame["subtype"].eq("sync") & frame["content"].eq("rsync")
    sync_times = frame.loc[mask, "time"].to_numpy(dtype=float)
    if len(sync_times) < 10:
        raise ValueError(f"Only {len(sync_times)} pyControl rsync pulses found; need at least 10")
    return frame, sync_times


def match_rsync_pulses(
    pycontrol_s: np.ndarray,
    scanimage_s: np.ndarray,
    chunk_size: int = 5,
    max_rms_s: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match unique random-IPI chunks, robust to missing leading/middle pulses."""
    if len(pycontrol_s) <= chunk_size or len(scanimage_s) <= chunk_size:
        raise ValueError("Too few rsync pulses for alignment")
    a = np.lib.stride_tricks.sliding_window_view(np.diff(pycontrol_s), chunk_size)
    b = np.lib.stride_tricks.sliding_window_view(np.diff(scanimage_s), chunk_size)
    pairs: list[tuple[int, int, float]] = []
    for i, window in enumerate(a):
        rms = np.sqrt(np.mean((b - window) ** 2, axis=1))
        j = int(np.argmin(rms))
        if rms[j] <= max_rms_s and i == int(np.argmin(np.sqrt(np.mean((a - b[j]) ** 2, axis=1)))):
            for offset in range(chunk_size + 1):
                pairs.append((i + offset, j + offset, float(rms[j])))
    if not pairs:
        raise ValueError("No matching rsync chunks found; verify the HDF5 and TSV belong to the same recording")

    # Retain the lowest-error proposal for each pulse, then enforce one-to-one,
    # monotonically increasing correspondence.
    pairs.sort(key=lambda item: item[2])
    chosen: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for item in pairs:
        if item[0] not in used_a and item[1] not in used_b:
            chosen.append(item)
            used_a.add(item[0])
            used_b.add(item[1])
    chosen.sort()
    monotonic: list[tuple[int, int, float]] = []
    last_b = -1
    for item in chosen:
        if item[1] > last_b:
            monotonic.append(item)
            last_b = item[1]
    if len(monotonic) < 6:
        raise ValueError(f"Only {len(monotonic)} rsync pulses matched")
    ia = np.array([item[0] for item in monotonic], dtype=np.int64)
    ib = np.array([item[1] for item in monotonic], dtype=np.int64)
    errors = np.array([item[2] for item in monotonic])
    return ia, ib, errors


def interpolate_with_nan(x: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Piecewise-linear conversion without extrapolating beyond sync coverage."""
    converted = np.interp(x, source, target)
    converted[(x < source[0]) | (x > source[-1])] = np.nan
    return converted


def nearest_frame_indices(event_times_s: np.ndarray, frame_times_s: np.ndarray) -> np.ndarray:
    valid_frames = np.flatnonzero(np.isfinite(frame_times_s))
    if not len(valid_frames):
        return np.full(len(event_times_s), -1, dtype=np.int64)
    finite_frame_times = frame_times_s[valid_frames]
    valid = np.isfinite(event_times_s) & (event_times_s >= finite_frame_times[0]) & (
        event_times_s <= finite_frame_times[-1]
    )
    result = np.full(len(event_times_s), -1, dtype=np.int64)
    positions = np.searchsorted(finite_frame_times, event_times_s[valid])
    positions = np.clip(positions, 0, len(finite_frame_times) - 1)
    left = np.maximum(positions - 1, 0)
    choose_left = np.abs(finite_frame_times[left] - event_times_s[valid]) <= np.abs(
        finite_frame_times[positions] - event_times_s[valid]
    )
    positions[choose_left] = left[choose_left]
    result[valid] = valid_frames[positions]
    return result


def load_bandit_session(tsv_path: Path, project: Path) -> dict[str, Any]:
    """Use magnitude-bandit-analysis itself to preserve its canonical fields."""
    src = str(project / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import_module = importlib.import_module("behavior_import.import_data")
    extract_trials = importlib.import_module("behavior_import.extract_trials").extract_trials
    raw = import_module.read_tsv_file(tsv_path)
    towers = import_module.extract_session_towers(raw)

    # magnitude-bandit-analysis.expand_content_columns passes every successfully
    # decoded JSON value to pandas.json_normalize. Some info rows contain valid
    # scalar JSON (e.g. an integer task hash), which newer pandas versions reject.
    # Normalise dictionary-valued rows only; scalar content has no fields to add.
    def parse_dict(value: Any) -> dict[str, Any]:
        if pd.isna(value):
            return {}
        try:
            decoded = json.loads(value) if isinstance(value, str) else value
        except (TypeError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    expanded = raw.copy()
    normalized = pd.json_normalize(expanded["content"].map(parse_dict), sep=".")
    for column in normalized.columns:
        expanded[column] = normalized[column].to_numpy()
    session = {
        "data": expanded,
        "df": raw.copy(),
        "problem": None,
        "choice_towers": towers["choice_towers"],
        "initiation_tower": towers["initiation_tower"],
    }
    parsed = extract_trials({"subject": {"session": session}})
    return parsed["subject"]["session"]


def add_event_frames(session: dict[str, Any], raw: pd.DataFrame, frame_times_pycontrol_s: np.ndarray) -> None:
    event_rows = raw.loc[raw["type"].isin(["event", "state"]), ["time", "type", "subtype", "content"]]
    events: dict[str, dict[str, np.ndarray]] = {}
    for name, group in event_rows.groupby("content", sort=False):
        if not isinstance(name, str) or not name:
            continue
        times = group["time"].to_numpy(dtype=float)
        events[name] = {"time_s": times, "frame": nearest_frame_indices(times, frame_times_pycontrol_s)}
    session["events"] = events

    trial_rows = raw.loc[raw["type"].eq("variable") & raw["subtype"].eq("print")].sort_values("time")
    trial_times = trial_rows["time"].to_numpy(dtype=float)
    session["trial_time_s"] = trial_times
    session["trial_frame"] = nearest_frame_indices(trial_times, frame_times_pycontrol_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--behavior-tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--behavior-project", type=Path, default=DEFAULT_BEHAVIOR_PROJECT)
    parser.add_argument("--plot", type=Path, help="Optional rsync residual QC PNG")
    args = parser.parse_args()

    clocks = read_scanimage_clocks(args.h5)
    raw, pycontrol_sync_s = read_pycontrol(args.behavior_tsv)
    ia, ib, chunk_errors = match_rsync_pulses(pycontrol_sync_s, clocks["sync_times_scanimage_s"])
    matched_py = pycontrol_sync_s[ia]
    matched_scan = clocks["sync_times_scanimage_s"][ib]
    frame_times_py = interpolate_with_nan(clocks["frame_times_scanimage_s"], matched_scan, matched_py)

    # Global affine residual is a useful QC diagnostic for clock drift/nonlinearity;
    # the actual conversion below remains piecewise-linear between pulse pairs.
    affine = np.polyfit(matched_scan, matched_py, 1)
    residual_ms = (np.polyval(affine, matched_scan) - matched_py) * 1000
    session = load_bandit_session(args.behavior_tsv, args.behavior_project)
    session.update(
        {
            "frame_times_scanimage_s": clocks["frame_times_scanimage_s"],
            "frame_times_pycontrol_s": frame_times_py,
            "frame_valid": np.isfinite(frame_times_py),
            "rsync": {
                "pycontrol_times_s": pycontrol_sync_s,
                "scanimage_times_s": clocks["sync_times_scanimage_s"],
                "matched_pycontrol_indices": ia,
                "matched_scanimage_indices": ib,
                "chunk_rms_s": chunk_errors,
                "residual_ms": residual_ms,
                "frame_clock_polarity": clocks["frame_polarity"],
                "sync_polarity": clocks["sync_polarity"],
                "sample_rate_hz": clocks["sample_rate_hz"],
            },
        }
    )
    add_event_frames(session, raw, frame_times_py)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(session, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
        axes[0].plot(matched_py, matched_scan - matched_py)
        axes[0].set(xlabel="pyControl time (s)", ylabel="ScanImage − pyControl (s)", title="Clock offset/drift")
        axes[1].plot(matched_py, chunk_errors * 1000, ".")
        axes[1].set(xlabel="pyControl time (s)", ylabel="chunk RMS (ms)", title="Rsync match error")
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)

    print(f"ScanImage frames: {len(clocks['frame_samples'])}")
    print(f"Rsync pulses: pyControl={len(pycontrol_sync_s)}, ScanImage={len(clocks['sync_samples'])}, matched={len(ia)}")
    print(f"Aligned frames within rsync coverage: {np.isfinite(frame_times_py).sum()}/{len(frame_times_py)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
