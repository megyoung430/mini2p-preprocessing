"""Spatial averaging of a registered movie: square tiles and the whole FOV.

Pure functions on in-memory arrays (or a numpy memmap). A crop is
``(y0, y1, x0, x1)``, half-open, in pixel coordinates of the full frame —
Suite2p's ``yrange``/``xrange`` after registration.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

Crop = tuple[int, int, int, int]


class TileGrid(NamedTuple):
    shape: tuple[int, int]      # (rows, columns) of complete tiles
    bounds: np.ndarray          # tiles x 4, (y0, y1, x0, x1), row-major tile IDs
    tile_size: int


def check_crop(crop: Crop, frame_shape: tuple[int, int]) -> Crop:
    y0, y1, x0, x1 = map(int, crop)
    height, width = frame_shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError(f"crop {crop} does not fit a {height}x{width} frame")
    return y0, y1, x0, x1


def tile_grid(crop: Crop, tile_size: int) -> TileGrid:
    """Complete ``tile_size`` square tiles covering the crop; partial edge tiles are dropped."""
    if tile_size < 1 or int(tile_size) != tile_size:
        raise ValueError(f"tile size must be a positive integer, got {tile_size!r}")
    tile_size = int(tile_size)
    y0, y1, x0, x1 = crop
    n_rows, n_cols = (y1 - y0) // tile_size, (x1 - x0) // tile_size
    if min(n_rows, n_cols) < 1:
        raise ValueError(f"crop {crop} holds no complete {tile_size}px tile")
    rows, cols = np.divmod(np.arange(n_rows * n_cols), n_cols)
    bounds = np.column_stack([y0 + rows * tile_size, y0 + (rows + 1) * tile_size,
                              x0 + cols * tile_size, x0 + (cols + 1) * tile_size])
    return TileGrid((n_rows, n_cols), bounds, tile_size)


def tile_means(movie: np.ndarray, grid: TileGrid, batch_frames: int = 500) -> np.ndarray:
    """Mean fluorescence of every tile, ``tiles x frames`` (float32).

    Reads ``movie`` (frames x height x width) in batches so a memmapped
    registered binary is never loaded whole.
    """
    n_rows, n_cols = grid.shape
    size = grid.tile_size
    y0, x0 = grid.bounds[0, 0], grid.bounds[0, 2]
    y1, x1 = y0 + n_rows * size, x0 + n_cols * size

    def batch_means(start: int) -> np.ndarray:
        block = np.asarray(movie[start:start + batch_frames, y0:y1, x0:x1], dtype=np.float32)
        # frames x tile-row x pixel-row x tile-col x pixel-col -> mean over pixels
        tiles = block.reshape(len(block), n_rows, size, n_cols, size).mean(axis=(2, 4))
        return tiles.reshape(len(block), -1)

    starts = range(0, len(movie), _check_batch(batch_frames))
    return np.concatenate([batch_means(s) for s in starts]).T


def whole_fov_mean(movie: np.ndarray, crop: Crop, batch_frames: int = 100) -> np.ndarray:
    """Mean raw fluorescence over the whole crop, per frame (float64).

    Averages pixels *before* any normalisation, so this is the raw FOV trace
    that a whole-FOV dF/F is computed from — not a mean of tile dF/F traces.
    """
    y0, y1, x0, x1 = crop
    starts = range(0, len(movie), _check_batch(batch_frames))
    return np.concatenate([
        movie[s:s + batch_frames, y0:y1, x0:x1].mean(axis=(1, 2), dtype=np.float64)
        for s in starts])


def brightness_selection(F: np.ndarray, bad: np.ndarray,
                         exclude_below_percentile: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep tiles whose time-mean raw F (over good frames) is >= a percentile cutoff.

    Returns ``(keep_mask, brightness, cutoff)``. Ties at the cutoff are kept;
    ``exclude_below_percentile == 0`` keeps every tile with finite brightness.
    This is a screen on intensity, not evidence that an excluded tile is noise.
    """
    if not 0 <= exclude_below_percentile < 100:
        raise ValueError("brightness percentile must be in [0, 100); 0 disables filtering")
    bad = np.asarray(bad, dtype=bool)
    brightness = np.nanmean(F[:, ~bad], axis=1)
    finite = np.isfinite(brightness)
    if not finite.any():
        raise ValueError("no tile has finite brightness")
    cutoff = float(np.percentile(brightness[finite], exclude_below_percentile))
    keep = finite & (brightness >= cutoff) if exclude_below_percentile > 0 else finite
    return keep, brightness, cutoff


def _check_batch(batch_frames: int) -> int:
    if batch_frames < 1 or int(batch_frames) != batch_frames:
        raise ValueError(f"batch size must be a positive integer, got {batch_frames!r}")
    return int(batch_frames)
