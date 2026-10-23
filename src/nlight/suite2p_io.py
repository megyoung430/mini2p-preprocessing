"""Reading Suite2p plane folders and writing nLightG2 trace exports.

This is the only module that touches the disk. The file names and array keys
are the ones the downstream event, pupil and video scripts already read, so
they must not change:

* ``tile_fluorescence_dff.npz/.json`` plus ``F/F0/dff/tile_ids/tile_bounds_yxyx.npy``
  and ``tile_selection.npz`` — retained tiles x frames
* ``whole_image_roi_dff.npz/.json`` — one trace over the whole valid crop

Exports go to ``<out_root>/<planeN>/`` when an output root is given, so a
Suite2p folder on a shared drive is only ever read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tiles import Crop, check_crop

TILE_EXPORT = "tile_fluorescence_dff"
WHOLE_FOV_EXPORT = "whole_image_roi_dff"


@dataclass(frozen=True)
class Plane:
    """A registered Suite2p plane: what the analysis needs from ``ops.npy``."""
    directory: Path
    binary: Path
    n_frames: int
    frame_shape: tuple[int, int]
    crop: Crop
    bad: np.ndarray
    mean_image: np.ndarray
    fs: float | None

    def movie(self) -> np.memmap:
        """Registered movie as a read-only memmap (frames x height x width)."""
        return np.memmap(self.binary, dtype=np.int16, mode="r",
                         shape=(self.n_frames, *self.frame_shape))


def load_plane(plane_dir: Path) -> Plane:
    plane_dir = Path(plane_dir)
    ops_path = plane_dir / "ops.npy"
    if not ops_path.is_file():
        raise FileNotFoundError(f"no Suite2p ops.npy in {plane_dir}")
    ops = np.load(ops_path, allow_pickle=True).item()
    binary = plane_dir / "data.bin"
    if not binary.is_file():
        binary = Path(ops["reg_file"])
    if not binary.is_file():
        raise FileNotFoundError(f"registered binary not found for {plane_dir}")
    n_frames = int(ops["nframes"])
    frame_shape = (int(ops["Ly"]), int(ops["Lx"]))
    crop = check_crop((*map(int, ops["yrange"]), *map(int, ops["xrange"])), frame_shape)
    bad = np.asarray(ops.get("badframes", np.zeros(n_frames, bool)), dtype=bool).ravel()
    if bad.shape != (n_frames,):
        raise ValueError(f"{plane_dir}: bad-frame mask length {bad.size} != {n_frames} frames")
    fs = float(ops["fs"]) if "fs" in ops else None
    return Plane(plane_dir, binary, n_frames, frame_shape, crop, bad,
                 np.asarray(ops.get("meanImg")), fs)


def plane_dirs(root: Path, marker: str = "ops.npy") -> list[Path]:
    """``plane0``, ``plane1``, … under ``root`` that contain ``marker``, in numeric order."""
    dirs = [p for p in Path(root).glob("plane[0-9]*") if (p / marker).is_file()]
    if not dirs:
        raise FileNotFoundError(f"no plane folders with {marker} under {root}")
    return sorted(dirs, key=lambda p: int(p.name[5:]))


# --- tile export ---------------------------------------------------------

def export_dir(plane: Plane, out_root: Path | None) -> Path:
    """Where a plane's exports go: ``out_root/<planeN>``, or the plane folder itself.

    Writing into the plane folder is refused when it holds a Suite2p detection
    run (``stat.npy``): its ``F.npy`` is the ROI fluorescence and the tile export
    would overwrite it.
    """
    if out_root is None:
        if (plane.directory / "stat.npy").exists():
            raise FileExistsError(
                f"{plane.directory} holds Suite2p ROI outputs (stat.npy, F.npy); "
                "pass an output root so they are not overwritten")
        return plane.directory
    d = Path(out_root) / plane.directory.name
    if d.resolve() == plane.directory.resolve():
        return export_dir(plane, None)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_tile_export(plane: Plane, result: dict, settings: dict,
                     out_root: Path | None = None) -> Path:
    d = export_dir(plane, out_root)
    keep = result["keep"]
    ids = np.flatnonzero(keep)
    retained = dict(F=result["F"], F0=result["F0"], dff=result["dff"],
                    tile_ids=ids, tile_bounds_yxyx=result["bounds"][keep])
    for name, array in retained.items():
        np.save(d / f"{name}.npy", array)
    np.savez_compressed(d / f"{TILE_EXPORT}.npz", time_s=result["time_s"],
                        badframes=plane.bad, **retained)
    np.savez_compressed(d / "tile_selection.npz", brightness=result["brightness"],
                        keep_mask=keep, all_tile_bounds_yxyx=result["bounds"])
    metadata = dict(
        registered_binary=str(plane.binary),
        ops_path=str(plane.directory / "ops.npy"),
        array_axes=["retained_tiles", "frames"],
        tile_size_px=settings["tile_size_px"],
        grid_shape=list(result["grid_shape"]),
        original_tile_count=int(keep.size),
        retained_tile_count=int(keep.sum()),
        brightness_exclusion_percentile=float(settings["exclude_below_percentile"]),
        brightness_threshold=float(result["cutoff"]),
        brightness_statistic="time mean of raw fluorescence over non-bad frames, per plane; not abs(F)",
        exclusion_rule="exclude brightness strictly below cutoff; ties retained; 0 disables filtering",
        tile_ids="original row-major grid IDs, aligned with rows of F/F0/dff",
        **_baseline_metadata(settings),
        array_dtype="float32",
        neuropil_subtraction=False,
    )
    _write_json(d / f"{TILE_EXPORT}.json", metadata)
    return d / f"{TILE_EXPORT}.npz"


def load_tile_export(plane_dir: Path) -> tuple[dict, dict]:
    path = Path(plane_dir) / f"{TILE_EXPORT}.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    return arrays, json.loads(path.with_suffix(".json").read_text())


# --- whole-FOV export ----------------------------------------------------

def save_whole_fov_export(plane: Plane, result: dict, settings: dict,
                          out_root: Path | None = None) -> Path:
    d = export_dir(plane, out_root)
    np.savez_compressed(d / f"{WHOLE_FOV_EXPORT}.npz", F=result["F"], F0=result["F0"],
                        dff=result["dff"], time_s=result["time_s"], badframes=plane.bad,
                        roi_bounds_yxyx=np.array(plane.crop))
    meta = _baseline_metadata(settings)
    _write_json(d / f"{WHOLE_FOV_EXPORT}.json", dict(
        roi="entire valid registered crop, including partial tile edges",
        roi_bounds_yxyx=list(plane.crop),
        calculation="average pixels first, then compute rolling F0 and (F-F0)/F0",
        baseline_window_s_requested=meta["baseline_window_s_requested"],
        baseline_window_frames=meta["baseline_window_frames"],
        baseline_percentile=meta["baseline_percentile"],
        baseline_min_valid_samples=meta["baseline_min_valid_samples"],
        baseline_centered=True,
        fs_hz=meta["fs_hz"],
        dff_units="fraction",
        registered_binary=str(plane.binary),
    ))
    return d / f"{WHOLE_FOV_EXPORT}.npz"


def load_whole_fov_export(plane_dir: Path) -> tuple[dict, dict]:
    path = Path(plane_dir) / f"{WHOLE_FOV_EXPORT}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no whole-FOV export in {plane_dir}; run extract_whole_fov first")
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    return arrays, json.loads(path.with_suffix(".json").read_text())


def cached_whole_fov_raw(plane: Plane, out_root: Path | None = None) -> np.ndarray | None:
    """Raw whole-FOV F from an existing export, if it was made from this registration.

    Averaging the full registered binary is the slow step, so a saved export
    whose crop and frame count match ``plane`` is reused. Returns None
    otherwise, and the caller recomputes.
    """
    d = plane.directory if out_root is None else Path(out_root) / plane.directory.name
    path = d / f"{WHOLE_FOV_EXPORT}.npz"
    if not path.is_file():
        return None
    with np.load(path) as saved:
        same_crop = tuple(saved["roi_bounds_yxyx"]) == tuple(plane.crop)
        F = saved["F"]
    return F if same_crop and F.shape == (plane.n_frames,) else None


def _baseline_metadata(settings: dict) -> dict:
    return dict(
        fs_hz=float(settings["fs"]),
        baseline_window_s_requested=float(settings["baseline_window_s"]),
        baseline_window_frames=int(settings["baseline_window_frames"]),
        baseline_window_s_effective=settings["baseline_window_frames"] / settings["fs"],
        baseline_percentile=float(settings["baseline_percentile"]),
        baseline_centered=True,
        baseline_min_valid_samples=int(settings["baseline_min_valid_samples"]),
        bad_frame_policy="exclude baseline samples; dff NaN; preserve time axis",
        dff_units="fraction",
    )


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2))
