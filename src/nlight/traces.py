"""Registered movie -> tile and whole-FOV F, F0 and dF/F.

``compute_*`` functions are pure analysis on in-memory data; ``extract_*``
functions are the thin I/O wrappers that read a Suite2p plane and save the
exports downstream scripts use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from . import baseline, suite2p_io, tiles


class TraceSettings(BaseModel, frozen=True):
    """Parameters for tile and whole-FOV trace extraction."""
    tile_size_px: int = Field(16, ge=1)
    batch_frames: int = Field(500, ge=1)
    exclude_below_percentile: float = Field(0.0, ge=0, lt=100)
    baseline_window_s: float = Field(60.0, gt=0)
    baseline_percentile: float = Field(20.0, ge=0, le=100)

    def resolved(self, fs: float, n_frames: int) -> dict:
        """Settings plus the frame-based quantities actually used, for metadata."""
        window = baseline.window_frames(self.baseline_window_s, fs)
        return dict(self.model_dump(), fs=fs, baseline_window_frames=window,
                    baseline_min_valid_samples=baseline.min_valid_samples(window, n_frames))


def rolling_dff(F: np.ndarray, bad: np.ndarray, fs: float,
                settings: TraceSettings) -> tuple[np.ndarray, np.ndarray]:
    """The pipeline's standard baseline: centered rolling percentile, then dF/F."""
    F0 = baseline.rolling_percentile(F, bad, fs, settings.baseline_window_s,
                                     settings.baseline_percentile)
    return F0, baseline.delta_f_over_f(F, F0, bad)


def compute_tile_traces(movie: np.ndarray, crop: tiles.Crop, bad: np.ndarray, fs: float,
                        settings: TraceSettings) -> dict:
    """Tile F (all tiles), brightness selection, and F0/dF/F for retained tiles."""
    grid = tiles.tile_grid(crop, settings.tile_size_px)
    F_all = tiles.tile_means(movie, grid, settings.batch_frames)
    keep, brightness, cutoff = tiles.brightness_selection(
        F_all, bad, settings.exclude_below_percentile)
    F = F_all[keep]
    F0, dff = rolling_dff(F, bad, fs, settings)
    return dict(F=F, F0=F0.astype(np.float32), dff=dff.astype(np.float32),
                keep=keep, brightness=brightness, cutoff=cutoff,
                bounds=grid.bounds, grid_shape=grid.shape, time_s=np.arange(len(bad)) / fs)


def compute_whole_fov_traces(F: np.ndarray, bad: np.ndarray, fs: float,
                             settings: TraceSettings) -> dict:
    """Rolling-baseline dF/F of an already-averaged raw whole-FOV trace."""
    F0, dff = rolling_dff(F, bad, fs, settings)
    return dict(F=F, F0=F0, dff=dff, time_s=np.arange(len(F)) / fs)


def extract_tiles(plane_dir: Path, settings: TraceSettings, fs: float | None = None,
                  out_root: Path | None = None) -> Path:
    """Tile export for one plane, saved to ``out_root/<planeN>`` (default: the plane folder)."""
    plane = suite2p_io.load_plane(plane_dir)
    fs = _resolve_fs(plane, fs)
    suite2p_io.export_dir(plane, out_root)  # fail before the slow step, not after
    result = compute_tile_traces(plane.movie(), plane.crop, plane.bad, fs, settings)
    path = suite2p_io.save_tile_export(plane, result, settings.resolved(fs, plane.n_frames),
                                       out_root)
    print(f"{plane.directory.name}: retained {result['keep'].sum()}/{result['keep'].size} "
          f"tiles; brightness cutoff {result['cutoff']:.4g}", flush=True)
    return path


def extract_whole_fov(plane_dir: Path, settings: TraceSettings, fs: float | None = None,
                      out_root: Path | None = None) -> Path:
    """Whole-FOV export for one plane, saved to ``out_root/<planeN>`` (default: the plane folder)."""
    plane = suite2p_io.load_plane(plane_dir)
    fs = _resolve_fs(plane, fs)
    suite2p_io.export_dir(plane, out_root)
    F = suite2p_io.cached_whole_fov_raw(plane, out_root)
    if F is None:
        F = tiles.whole_fov_mean(plane.movie(), plane.crop)
    result = compute_whole_fov_traces(F, plane.bad, fs, settings)
    return suite2p_io.save_whole_fov_export(plane, result, settings.resolved(fs, plane.n_frames),
                                            out_root)


def _resolve_fs(plane: suite2p_io.Plane, fs: float | None) -> float:
    """Per-plane frame rate: the caller's value, checked against ops['fs'] when both exist."""
    if fs is None:
        if plane.fs is None:
            raise ValueError(f"{plane.directory}: no 'fs' in ops.npy; pass the verified rate")
        return plane.fs
    if plane.fs is not None and not np.isclose(fs, plane.fs, rtol=1e-3):
        raise ValueError(f"{plane.directory}: fs={fs} disagrees with ops['fs']={plane.fs}")
    return float(fs)
