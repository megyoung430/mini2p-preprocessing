"""Load one recording's exports plus behaviour, and compute dF/F under every
baseline definition (cached, since rolling percentiles over long windows on
hundreds of tiles are slow)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Memory

from . import analyses, baseline, baseline_comparison, behavior, recording_comparison, suite2p_io
from .recording_comparison import BaselineSpec, PlaneTrace
from .regression import RunningModel

# Outside the repo and outside any shared drive; deleting it only costs time.
CACHE = Memory(os.environ.get("NLIGHT_CACHE_DIR", Path.home() / ".cache" / "nlight"), verbose=0)


@dataclass(frozen=True)
class Recording:
    name: str
    export_root: Path                    # holds planeN/ with whole-FOV (and tile) exports
    behavior_table: Path | None = None   # data.pkl or all_data.pkl, row-aligned to frames
    suite2p_root: Path | None = None     # for the consolidation record of dropped volumes
    derived_root: Path | None = None     # local folder for derived outputs (never the share)


@dataclass(frozen=True)
class TilePlane:
    name: str
    F: np.ndarray          # retained tiles x frames, raw
    tile_ids: np.ndarray
    bounds: np.ndarray
    grid_shape: tuple[int, int]
    bad: np.ndarray


def load_whole_fov(recording: Recording) -> list[PlaneTrace]:
    from .workflows import load_recording
    return load_recording(recording.export_root)


def load_tiles(recording: Recording) -> list[TilePlane]:
    def plane(d: Path) -> TilePlane:
        arrays, meta = suite2p_io.load_tile_export(d)
        return TilePlane(d.name, arrays["F"], arrays["tile_ids"], arrays["tile_bounds_yxyx"],
                         tuple(meta["grid_shape"]), arrays["badframes"].astype(bool))
    return [plane(d) for d in suite2p_io.plane_dirs(recording.export_root,
                                                     marker=suite2p_io.TILE_EXPORT + ".npz")]


def load_behavior(recording: Recording, n_frames: int, n_planes: int,
                  events: dict = behavior.DEFAULT_EVENTS) -> behavior.Behavior | None:
    if recording.behavior_table is None:
        return None
    extra = (behavior.verified_extra_rows(recording.suite2p_root, n_frames, n_planes)
             if recording.suite2p_root is not None else 0)
    table = behavior.align_rows(behavior.read_table(recording.behavior_table), n_frames, extra)
    return behavior.behavior_from_table(table, events)


def whole_fov_variants(planes: list[PlaneTrace], spec: BaselineSpec) -> dict[str, np.ndarray]:
    """Plane-mean whole-FOV dF/F for every baseline definition."""
    return recording_comparison.plane_mean_dff(planes, spec)


@CACHE.cache
def tile_variants(F: np.ndarray, bad: np.ndarray, fs: float,
                  spec: BaselineSpec) -> dict[str, np.ndarray]:
    """Tile dF/F (float32, tiles x frames) for every baseline definition."""
    F0s = baseline_comparison.candidate_baselines(
        F, bad, fs, spec.current_window_s, spec.current_percentile,
        spec.fixed_percentiles, spec.long_windows_s)
    return {k: baseline.delta_f_over_f(F, F0, bad).astype(np.float32) for k, F0 in F0s.items()}


def load_recordings(config_path: Path, meg1_root: Path | None = None) -> dict[str, Recording]:
    """Recordings from a JSON config. ``{meg1}`` in a path is replaced by
    ``meg1_root`` (default: the config's ``meg1_root``); relative paths are
    resolved against the config file's folder."""
    import json
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    meg1 = str(meg1_root or os.environ.get("NLIGHT_MEG1_ROOT", config["meg1_root"]))

    def path(value):
        if value is None:
            return None
        p = Path(value.replace("{meg1}", meg1))
        return p if p.is_absolute() else (config_path.parent / p).resolve()

    return {name: Recording(name, path(r["export_root"]), path(r.get("behavior_table")),
                            path(r.get("suite2p_root")), path(r.get("derived_root")))
            for name, r in config["recordings"].items()}


def open_session(recording: Recording, spec: BaselineSpec, tiles: bool = True,
                 events: dict = behavior.DEFAULT_EVENTS, regress_running: bool = False,
                 running_model: RunningModel = RunningModel()) -> dict:
    """Everything the analysis notebooks need for one recording.

    ``wf`` holds plane-mean whole-FOV dF/F per baseline; ``tiles`` one entry
    per plane with tile dF/F per baseline (cached on disk). With
    ``regress_running=True`` both are replaced by running-regressed dF/F
    (see :func:`apply_running_regression`); the originals stay in
    ``wf_original`` and each tile entry's ``dff_original`` either way.
    """
    planes = load_whole_fov(recording)
    fs, n_frames = planes[0].fs, len(planes[0].F)
    tile_planes = load_tiles(recording) if tiles else []
    wf = whole_fov_variants(planes, spec)
    tile_entries = []
    for t in tile_planes:
        dff = tile_variants(t.F, t.bad, fs, spec)
        tile_entries.append(dict(plane=t, dff=dff, dff_original=dff))
    s = dict(recording=recording, spec=spec, fs=fs, n_frames=n_frames, n_planes=len(planes),
             time_s=np.arange(n_frames) / fs, planes=planes, wf=wf, wf_original=wf,
             valid=np.logical_and.reduce([~p.bad for p in planes]), tiles=tile_entries,
             behavior=load_behavior(recording, n_frames, len(planes), events),
             regress_running=False)
    return apply_running_regression(s, running_model) if regress_running else s


@CACHE.cache
def cached_running_regression(variants, speed, fs, model):
    return analyses.running_regression(variants, speed, fs, model)


def apply_running_regression(s: dict, model: RunningModel = RunningModel()) -> dict:
    """Session with ``wf`` and tile ``dff`` replaced by running-regressed dF/F.

    Regression is fitted separately for every baseline method, and for every
    tile. Fits, variance explained and kernels are kept under ``regression``
    (whole FOV) and each tile entry's ``regression``.
    """
    b = s["behavior"]
    if b is None or b.speed is None:
        raise ValueError(f"{s['recording'].name}: no running speed, so running cannot be regressed out")
    wf = cached_running_regression(s["wf_original"], b.speed, s["fs"], model)
    tiles = [dict(t, dff=r["residual"], regression=r) for t, r in
             ((t, cached_running_regression(t["dff_original"], b.speed, s["fs"], model))
              for t in s["tiles"])]
    return dict(s, wf=wf["residual"], regression=wf, tiles=tiles, running=wf["running"],
                regress_running=True, running_model=model)


def save_running_regressed(s: dict, out_dir: Path | None = None) -> Path:
    """Write running-regressed dF/F (whole FOV and tiles, every baseline) to a local folder.

    Default: ``<derived_root>/running_regressed``. Refuses a folder inside the
    recording's Suite2p root, which may be on the shared drive.
    """
    import json
    import re

    if not s["regress_running"]:
        raise ValueError("open the session with regress_running=True first")
    rec = s["recording"]
    out_dir = Path(out_dir or (rec.derived_root or rec.export_root) / "running_regressed")
    if rec.suite2p_root is not None and Path(rec.suite2p_root).resolve() in [out_dir.resolve(), *out_dir.resolve().parents]:
        raise ValueError(f"{out_dir} is inside the Suite2p folder; choose a local derived_root")
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = {m: re.sub(r"[^0-9a-zA-Z]+", "_", m).strip("_").lower() for m in s["wf"]}
    np.savez_compressed(out_dir / "whole_fov_running_regressed_dff.npz", time_s=s["time_s"],
                        running=s["running"], **{keys[m]: v for m, v in s["wf"].items()})
    for t in s["tiles"]:
        np.savez_compressed(out_dir / f"{t['plane'].name}_tiles_running_regressed_dff.npz",
                            tile_ids=t["plane"].tile_ids, time_s=s["time_s"],
                            **{keys[m]: v for m, v in t["dff"].items()})
    s["regression"]["table"].to_csv(out_dir / "whole_fov_regression.csv")
    (out_dir / "settings.json").write_text(json.dumps(dict(
        recording=rec.name, baseline_methods=keys, baseline_spec=vars(s["spec"]),
        running_model=vars(s["running_model"]),
        method="Per baseline method and per trace: least squares of dF/F on lagged running speed "
               "+ running indicator + intercept; the mean-centred fitted running part is subtracted",
        units="fractional dF/F; NaN where speed is missing or the lagged design is undefined",
        whole_fov="mean of per-plane dF/F across planes, regressed after averaging"), indent=2, default=list))
    return out_dir
