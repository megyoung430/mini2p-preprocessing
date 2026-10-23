"""Convert Cellpose masks for every MY_71_R run in sessions 216–231.

Run with the same Python environment as Convert_seg_to_stat (Suite2p, Torch,
and NumPy). Preview the inputs without loading Suite2p or writing data:
    python scripts/convert_seg_to_stat_216_231.py --dry-run
Run all calculations and save into each run's suite2p/plane0 folder:
    python scripts/convert_seg_to_stat_216_231.py

Existing output arrays are copied to a timestamped backup in each plane folder.
Registered binaries and Cellpose segmentations are never replaced. Each run uses
its own segmentation. Failures are reported and remaining runs are attempted.
"""

import argparse
import logging
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import shutil

SUBJECT_PATH = Path(
    "/Volumes/behrens/meg/3x3_field_magnitude_bandit/derivatives/"
    "cohort-01/sub-03_id-MY_71_R"
)
SEG_NAME = "meanImg_green_functional_contrast_seg.npy"
OUTPUT_NAMES = (
    "F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy",
    "iscell.npy", "ops.npy", "spks.npy", "stat.npy",
)


def discover_runs(subject_path, seg_name):
    if not subject_path.is_dir():
        raise FileNotFoundError(f"Subject folder is unavailable: {subject_path}")
    ready, problems = [], []
    for session in range(216, 232):
        planes = sorted(subject_path.glob(
            f"ses-{session}_*/funcimg/run-*/suite2p/plane0"
        ))
        if not planes:
            problems.append(f"Session {session}: no run plane0 folders found")
        for plane in planes:
            missing = [name for name in (seg_name, "ops.npy", "data.bin")
                       if not (plane / name).is_file()]
            if missing:
                problems.append(f"{plane}: missing {', '.join(missing)}")
            else:
                ready.append(plane)
    return ready, problems


def convert_run(wd, seg_name):
    import numpy as np
    import torch
    import suite2p
    from suite2p.extraction.masks import create_masks
    from suite2p.extraction.extract import extraction_wrapper
    from suite2p.detection import roi_stats

    cellpose_masks = np.load(wd / seg_name, allow_pickle=True).item()
    ops = np.load(wd / "ops.npy", allow_pickle=True).item()
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    masks = np.asarray(cellpose_masks["masks"])
    if (masks.shape != (Ly, Lx)
            or not np.issubdtype(masks.dtype, np.integer)
            or np.any(masks < 0) or not np.any(masks > 0)):
        raise ValueError(f"Expected nonempty nonnegative integer masks of shape {(Ly, Lx)}")
    binary_paths = [wd / "data.bin"]
    if (wd / "data_chan2.bin").is_file():
        binary_paths.append(wd / "data_chan2.bin")
    elif ops.get("nchannels", 1) == 2:
        raise FileNotFoundError(f"Two-channel ops but no data_chan2.bin in {wd}")
    sizes = [p.stat().st_size for p in binary_paths]
    if any(size == 0 or size % (2 * Ly * Lx) for size in sizes):
        raise ValueError("Registered binaries must contain complete int16 frames")
    if len(set(sizes)) != 1:
        raise ValueError("Registered channels have different frame counts")
    print(f"  {Ly} × {Lx} pixels, {sizes[0] // (2 * Ly * Lx)} frames/channel, "
          f"{len(binary_paths)} channel(s), {sum(sizes) / 1024**3:.2f} GiB to read; CPU extraction.",
          flush=True)
    print("  Building ROI and neuropil masks...", flush=True)

    with ExitStack() as stack:
        f_reg = stack.enter_context(suite2p.io.BinaryFile(Ly, Lx, binary_paths[0]))
        f_reg2 = (stack.enter_context(suite2p.io.BinaryFile(Ly, Lx, binary_paths[1]))
                  if len(binary_paths) == 2 else None)
        # Using these inputs, we will first mimic the stat array made by suite2p
        masks = cellpose_masks['masks']
        stat = []
        for u_ix, u in enumerate(np.unique(masks[masks > 0])):
            ypix,xpix = np.nonzero(masks==u)
            npix = len(ypix)
            stat.append({'ypix': ypix, 'xpix': xpix, 'npix': npix, 'lam': np.ones(npix, np.float32), 'med': [np.mean(ypix), np.mean(xpix)]})
        stat = np.array(stat)
        stat = roi_stats(stat, Ly, Lx)  # This function fills in remaining roi properties to make it compatible with the rest of the suite2p pipeline/GUI

        # Using the constructed stat file, get masks.  Suite2p's current API takes
        # lam_percentile as the fourth positional argument, so pass every option by
        # name rather than accidentally passing the complete ops dictionary there.
        cell_masks, neuropil_masks = create_masks(
            stat,
            Ly=Ly,
            Lx=Lx,
            lam_percentile=ops.get('lam_percentile', 50.0),
            allow_overlap=ops.get('allow_overlap', False),
            neuropil_extract=True,
            inner_neuropil_radius=ops.get('inner_neuropil_radius', 2),
            min_neuropil_pixels=ops.get('min_neuropil_pixels', 350),
            circular_neuropil=ops.get('circular_neuropil', False),
        )

        # Feed these values into the wrapper functions
        # Current Suite2p returns four fluorescence arrays (not stat) and accepts an
        # extraction settings dictionary rather than ops=.  Supplying the masks above
        # ensures the exact same masks are used for both channels. Use CPU explicitly
        # so this also runs on Macs without CUDA.
        F, Fneu, F_chan2, Fneu_chan2 = extraction_wrapper(
            stat,
            f_reg,
            f_reg_chan2=f_reg2,
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            settings={'batch_size': ops.get('batch_size', 500)},
            device=torch.device('cpu'),
        )
        stat_after_extraction = stat


        # Do cell classification
        print("  Classifying ROIs...", flush=True)
        classfile = suite2p.classification.builtin_classfile
        iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

        # Apply preprocessing step for deconvolution
        # Some newer Suite2p ops files omit the older flat deconvolution keys. Use
        # Suite2p's documented defaults, including the standard 0.7 neuropil factor,
        # and add them to ops so the saved outputs remain GUI-compatible.
        deconv_defaults = {
            'neucoeff': 0.7,
            'baseline': 'maximin',
            'win_baseline': 60.0,
            'sig_baseline': 10.0,
            'prctile_baseline': 8.0,
            'batch_size': 500,
            'tau': 1.0,
            'fs': 7.52,
        }
        for key, value in deconv_defaults.items():
            ops.setdefault(key, value)

        dF = F.copy() - ops['neucoeff'] * Fneu
        print("  Correcting neuropil and preprocessing baseline...", flush=True)
        dF = suite2p.extraction.preprocess(
                F=dF,
                baseline=ops['baseline'],
                win_baseline=ops['win_baseline'],
                sig_baseline=ops['sig_baseline'],
                fs=ops['fs'],
                prctile_baseline=ops['prctile_baseline'],
                batch_size=ops['batch_size'],
                device=torch.device('cpu'),
            )
        # Identify spikes
        print("  Deconvolving spikes...", flush=True)
        spks = suite2p.extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])



    outputs = dict(zip(OUTPUT_NAMES, (
        F, Fneu, F_chan2, Fneu_chan2, iscell, ops, spks, stat,
    )))
    backup = wd / ("backup_before_cellpose_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir()
    print(f"  Backing up existing outputs to {backup.name}...", flush=True)
    for name in OUTPUT_NAMES:
        if (wd / name).exists():
            shutil.copy2(wd / name, backup / name)
    print("  Saving calculated outputs...", flush=True)
    for name, value in outputs.items():
        if value is not None:
            np.save(wd / name, value)
        elif (wd / name).exists():
            # Remove stale second-channel arrays only after backing them up.
            (wd / name).unlink()
    print(f"  Saved {len(stat)} ROIs, {F.shape[1]} frames; backup: {backup.name}", flush=True)


def main():
    # Suite2p sends extraction batch progress to its INFO logger rather than
    # directly to the terminal. Enable it so long runs do not appear stalled.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("suite2p").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-path", type=Path, default=SUBJECT_PATH)
    parser.add_argument("--seg-name", default=SEG_NAME, help="Exact mask filename inside each plane0")
    parser.add_argument("--dry-run", action="store_true", help="List available inputs without writing outputs")
    args = parser.parse_args()
    if Path(args.seg_name).name != args.seg_name:
        parser.error("--seg-name must be a filename, not a path")
    ready, problems = discover_runs(args.subject_path, args.seg_name)
    for problem in problems:
        print(f"MISSING: {problem}", flush=True)
    print(f"Found {len(ready)} ready runs across sessions 216–231.", flush=True)
    failures, completed = [], 0
    for index, plane in enumerate(ready, 1):
        print(f"[{index}/{len(ready)}] {plane.relative_to(args.subject_path)}", flush=True)
        if args.dry_run:
            continue
        try:
            convert_run(plane, args.seg_name)
            completed += 1
        except Exception as exc:
            failures.append((plane, str(exc)))
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"{'Preview' if args.dry_run else 'Finished'}: {len(ready)} ready, "
          f"{completed} completed, {len(problems)} missing, {len(failures)} failed.", flush=True)
    for plane, error in failures:
        print(f"FAILED: {plane}: {error}")
    return 1 if problems or failures or not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
