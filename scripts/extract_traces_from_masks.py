from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PLANE_DIR = Path("'/Users/megyoung/Desktop/mini2p Images/distdetrd/suite2p/plane0")
GREEN_SEG = PLANE_DIR / "meanImg_ch1_seg.npy"
RED_SEG = PLANE_DIR / "data_chan2_mean_seg.npy"

# Use the registered green functional channel for all trace extraction.
# If your green channel is saved as data_chan2.bin, change this to PLANE_DIR / "data_chan2.bin".
GREEN_BIN = PLANE_DIR / "data.bin"
OUT_DIR = PLANE_DIR / "custom_mask_traces"


def load_cellpose_masks(seg_path: Path) -> np.ndarray:
    seg = np.load(seg_path, allow_pickle=True).item()
    if "masks" not in seg:
        raise KeyError(f"No 'masks' array found in {seg_path}")
    masks = np.asarray(seg["masks"])
    if masks.ndim != 2:
        raise ValueError(f"Expected 2D masks in {seg_path}, got shape {masks.shape}")
    return masks


def masks_to_rois(mask_image: np.ndarray, source_channel: str) -> list[dict]:
    rois = []
    for mask_id in sorted(int(v) for v in np.unique(mask_image) if v > 0):
        ypix, xpix = np.nonzero(mask_image == mask_id)
        if len(ypix) == 0:
            continue
        rois.append(
            {
                "source_channel": source_channel,
                "source_mask_id": mask_id,
                "ypix": ypix.astype(np.int64),
                "xpix": xpix.astype(np.int64),
                "lam": np.ones(len(ypix), dtype=np.float32),
                "npix": int(len(ypix)),
                "med": np.array([np.median(ypix), np.median(xpix)], dtype=np.float32),
            }
        )
    return rois


def suite2p_stat_from_rois(rois: list[dict]) -> np.ndarray:
    stat = []
    for roi in rois:
        stat.append(
            {
                "ypix": roi["ypix"],
                "xpix": roi["xpix"],
                "lam": roi["lam"],
                "npix": roi["npix"],
                "med": roi["med"],
                "source_channel": roi["source_channel"],
                "source_mask_id": roi["source_mask_id"],
            }
        )
    return np.asarray(stat, dtype=object)


def extract_f_traces(movie: np.memmap, rois: list[dict], chunk_size: int) -> np.ndarray:
    n_frames = movie.shape[0]
    f = np.zeros((len(rois), n_frames), dtype=np.float32)

    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        chunk = movie[start:stop]
        for roi_i, roi in enumerate(rois):
            pixels = chunk[:, roi["ypix"], roi["xpix"]]
            f[roi_i, start:stop] = pixels.mean(axis=1)
        print(f"extracted {stop}/{n_frames} frames")

    return f


def calculate_dff(f: np.ndarray, baseline_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.percentile(f, baseline_percentile, axis=1, keepdims=True)
    f0 = np.maximum(f0, 1e-6)
    dff = (f - f0) / f0
    return f0[:, 0], dff.astype(np.float32)


def save_roi_table(path: Path, rois: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["roi_id", "roi_source_channel", "source_mask_id", "npix", "med_y", "med_x"],
        )
        writer.writeheader()
        for roi_id, roi in enumerate(rois):
            writer.writerow(
                {
                    "roi_id": roi_id,
                    "roi_source_channel": roi["source_channel"],
                    "source_mask_id": roi["source_mask_id"],
                    "npix": roi["npix"],
                    "med_y": float(roi["med"][0]),
                    "med_x": float(roi["med"][1]),
                }
            )


def load_session_dict(path: Path) -> dict[str, Any]:
    """Load an already frame-aligned behavioural session dictionary."""
    with path.open("rb") as handle:
        session = pickle.load(handle)
    if not isinstance(session, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(session).__name__}")
    return session


def validate_session_frame_indices(session: dict[str, Any], n_frames: int) -> None:
    """Catch common alignment/indexing mistakes before packaging a session."""
    for key in ("frame_times_scanimage_s", "frame_times_pycontrol_s", "frame_valid"):
        if key in session and len(np.asarray(session[key])) != n_frames:
            raise ValueError(
                f"session[{key!r}] has {len(np.asarray(session[key]))} samples but dF has {n_frames} frames"
            )

    for key in ("licks", "rewards", "thresholded_licks"):
        if key not in session:
            continue
        indices = np.asarray(session[key])
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"session[{key!r}] must be a 1D integer array of frame indices")
        if indices.size and (indices.min() < 0 or indices.max() >= n_frames):
            raise ValueError(
                f"session[{key!r}] contains indices outside the trace range 0..{n_frames - 1}"
            )

    # The supplied reference file has one more dF sample than its framewise
    # behaviour arrays. Accept that established convention, but reject larger
    # discrepancies which almost always indicate a failed alignment.
    for key in ("position", "total_dist", "speed", "lap_idx", "lm_idx"):
        if key not in session:
            continue
        values = np.asarray(session[key])
        if values.ndim != 1:
            raise ValueError(f"session[{key!r}] must be a 1D framewise array")
        if len(values) not in (n_frames, n_frames - 1):
            raise ValueError(
                f"session[{key!r}] has {len(values)} samples but dF has {n_frames} frames; "
                "expected the same length or one fewer, as in the reference file"
            )


def save_img_session_data(
    path: Path,
    d_f: np.ndarray,
    session: dict[str, Any],
    frame_rate: float,
) -> None:
    """Save the core schema used by the supplied img_session_data.pkl example.

    Tuning curves, preferences, and selected-cell arrays are deliberately not
    added here: those are analysis outputs rather than preprocessing outputs.
    """
    validate_session_frame_indices(session, d_f.shape[1])
    payload = {
        "dF": np.asarray(d_f, dtype=np.float32),
        "session": session,
        "frame_rate": int(frame_rate) if float(frame_rate).is_integer() else float(frame_rate),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract green-channel fluorescence traces from Cellpose seg.npy masks on a Suite2p registered binary."
    )
    parser.add_argument("--plane-dir", type=Path, default=PLANE_DIR, help="Suite2p plane0 directory containing ops.npy.")
    parser.add_argument("--green-seg", type=Path, default=GREEN_SEG, help="Cellpose *_seg.npy from green channel.")
    parser.add_argument("--red-seg", type=Path, default=RED_SEG, help="Cellpose *_seg.npy from red structural channel.")
    parser.add_argument(
        "--green-bin",
        type=Path,
        default=GREEN_BIN,
        help="Registered green-channel binary.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--baseline-percentile", type=float, default=20.0)
    parser.add_argument(
        "--session-pkl",
        type=Path,
        help="Pickle containing a frame-aligned pyControl session dict. If supplied, also write img_session_data.pkl.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        help="Imaging frame rate in Hz for img_session_data.pkl (defaults to ops['fs']).",
    )
    parser.add_argument("--session-output", type=Path, help="Output pickle path (default: OUT_DIR/img_session_data.pkl).")
    args = parser.parse_args()

    plane_dir = args.plane_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    ly, lx = int(ops["Ly"]), int(ops["Lx"])

    green_bin = args.green_bin
    n_frames = green_bin.stat().st_size // (np.dtype("int16").itemsize * ly * lx)
    if n_frames == 0:
        raise ValueError(f"No frames found in {green_bin}")

    green_masks = load_cellpose_masks(args.green_seg)
    red_masks = load_cellpose_masks(args.red_seg)
    if green_masks.shape != (ly, lx):
        raise ValueError(f"Green masks shape {green_masks.shape} does not match Suite2p frame shape {(ly, lx)}")
    if red_masks.shape != (ly, lx):
        raise ValueError(f"Red masks shape {red_masks.shape} does not match Suite2p frame shape {(ly, lx)}")

    rois = masks_to_rois(green_masks, "green") + masks_to_rois(red_masks, "red")
    if not rois:
        raise ValueError("No ROIs found in the provided seg.npy masks")

    movie = np.memmap(green_bin, dtype="int16", mode="r", shape=(n_frames, ly, lx))
    print(f"green binary: {green_bin}")
    print(f"movie shape: {movie.shape}")
    print(f"green ROIs: {sum(roi['source_channel'] == 'green' for roi in rois)}")
    print(f"red structural ROIs: {sum(roi['source_channel'] == 'red' for roi in rois)}")
    print("all traces will be extracted from the green binary")

    f = extract_f_traces(movie, rois, chunk_size=args.chunk_size)
    f0, dff = calculate_dff(f, baseline_percentile=args.baseline_percentile)
    d_f = (f - f0[:, None]).astype(np.float32)

    np.save(out_dir / "F.npy", f)
    np.save(out_dir / "F0.npy", f0)
    np.save(out_dir / "dF.npy", d_f)
    np.save(out_dir / "dff.npy", dff)
    np.save(out_dir / "stat.npy", suite2p_stat_from_rois(rois))
    np.save(out_dir / "iscell.npy", np.column_stack([np.ones(len(rois)), np.ones(len(rois))]))
    np.save(out_dir / "roi_source_channel.npy", np.asarray([roi["source_channel"] for roi in rois]))
    save_roi_table(out_dir / "roi_table.csv", rois)

    if args.session_pkl is not None:
        session = load_session_dict(args.session_pkl)
        frame_rate = args.frame_rate if args.frame_rate is not None else ops.get("fs")
        if frame_rate is None:
            raise ValueError("No frame rate supplied and ops.npy has no 'fs' value")
        session_output = args.session_output or (out_dir / "img_session_data.pkl")
        session_output.parent.mkdir(parents=True, exist_ok=True)
        save_img_session_data(session_output, d_f, session, float(frame_rate))
        print(f"saved aligned session data to: {session_output}")

    print(f"saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
