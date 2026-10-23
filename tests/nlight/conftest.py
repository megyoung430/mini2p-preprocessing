"""Synthetic Suite2p plane folders: a tiny registered movie plus ops.npy."""
from pathlib import Path

import numpy as np
import pytest


def write_plane(directory: Path, movie: np.ndarray, crop=(2, 30, 3, 37), fs=10.0,
                bad: np.ndarray | None = None) -> Path:
    directory.mkdir(parents=True)
    n_frames, height, width = movie.shape
    movie.astype(np.int16).tofile(directory / "data.bin")
    np.save(directory / "ops.npy", dict(
        nframes=n_frames, Ly=height, Lx=width, fs=fs,
        yrange=[crop[0], crop[1]], xrange=[crop[2], crop[3]],
        badframes=np.zeros(n_frames, bool) if bad is None else bad,
        meanImg=movie.mean(axis=0), reg_file=str(directory / "data.bin")), allow_pickle=True)
    return directory


@pytest.fixture
def synthetic_plane(tmp_path) -> Path:
    """600 frames at 10 Hz, 32x40 px, spatially varying brightness, slow drift,
    a few bright transients and two bad frames."""
    rng = np.random.default_rng(0)
    n_frames, height, width = 600, 32, 40
    t = np.arange(n_frames) / 10
    spatial = 800 + 400 * np.linspace(0, 1, width)[None, :] * np.ones((height, 1))
    temporal = (1 + 0.05 * np.sin(2 * np.pi * t / 30)) * np.exp(-t / 300)
    temporal[200:210] *= 1.3
    movie = spatial[None] * temporal[:, None, None] + rng.normal(0, 20, (n_frames, height, width))
    bad = np.zeros(n_frames, bool)
    bad[[50, 51]] = True
    return write_plane(tmp_path / "suite2p" / "plane0", movie, bad=bad)
