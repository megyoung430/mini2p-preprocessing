"""Behaviour signals row-aligned to imaging frames: events, pupil, running speed.

Row ``i`` of the behaviour table is imaging frame (or, for multi-plane
recordings, volume) ``i``. That alignment comes from the acquisition code; it
is checked here, never inferred. The only accepted mismatch is one extra final
row, and only when the TIFF-consolidation record shows that a final incomplete
volume was dropped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Confirmed codes: Stim 2 = Grating 1, Stim 4 = Grating 2, Reward 1 = reward.
DEFAULT_EVENTS = {"Grating 1": ("Stim", 2), "Grating 2": ("Stim", 4), "Reward": ("Reward", 1)}
PUPIL_COLUMN = "pupil_diameter"
SPEED_COLUMN = "Speed_Absolute"


class Behavior(NamedTuple):
    onsets: dict[str, np.ndarray]   # event name -> onset frame indices
    pupil: np.ndarray | None        # frames; NaN where missing or non-positive
    speed: np.ndarray | None        # frames, in the table's units


# --- I/O -----------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Behaviour table from ``data.pkl`` (a DataFrame) or ``all_data.pkl`` (its ``aligned_data``)."""
    data = pd.read_pickle(Path(path))
    if isinstance(data, dict):
        if "aligned_data" not in data:
            raise ValueError(f"{path}: dictionary without an 'aligned_data' table")
        data = data["aligned_data"]
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"{path}: expected a DataFrame, got {type(data).__name__}")
    return data


def verified_extra_rows(suite2p_root: Path, n_frames: int, n_planes: int) -> int:
    """1 if the consolidation record shows a dropped final partial volume, else 0."""
    records = list((Path(suite2p_root) / "input").glob("*complete_volumes.json"))
    if len(records) != 1:
        return 0
    record = json.loads(records[0].read_text())
    if record["output_pages"] != n_frames * n_planes:
        raise ValueError(f"{records[0]}: {record['output_pages']} pages != "
                         f"{n_frames} frames x {n_planes} planes")
    return int(record["omitted_final_pages"] > 0)


# --- analysis ------------------------------------------------------------

def align_rows(table: pd.DataFrame, n_frames: int, extra_rows_allowed: int = 0) -> pd.DataFrame:
    """The first ``n_frames`` rows, after checking the row count is the expected one."""
    extra = len(table) - n_frames
    if extra not in (0, extra_rows_allowed):
        raise ValueError(f"behaviour table has {len(table)} rows for {n_frames} imaging frames"
                         + (f" (only {extra_rows_allowed} verified extra row allowed)"
                            if extra_rows_allowed else ""))
    return table.iloc[:n_frames].reset_index(drop=True)


def event_onsets(values, code) -> np.ndarray:
    """Rising edges into ``code``: consecutive active rows form one onset."""
    active = np.asarray(values) == code
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def positive_or_nan(values) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=True)
    x[~np.isfinite(x) | (x <= 0)] = np.nan
    return x


def behavior_from_table(table: pd.DataFrame, events: dict = DEFAULT_EVENTS) -> Behavior:
    """Events, pupil and speed from an aligned table. A signal whose column is
    absent (e.g. no stimuli in a pre-corridor baseline) is returned empty/None."""
    onsets = {name: event_onsets(table[column], code)
              for name, (column, code) in events.items() if column in table}
    pupil = positive_or_nan(table[PUPIL_COLUMN]) if PUPIL_COLUMN in table else None
    speed = (pd.to_numeric(table[SPEED_COLUMN], errors="coerce").to_numpy(dtype=float, copy=True)
             if SPEED_COLUMN in table else None)
    return Behavior({k: v for k, v in onsets.items() if len(v)}, pupil, speed)
