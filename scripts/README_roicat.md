# ROICaT session tracking

These scripts use the installed `roicat` conda environment (tested against ROICaT 1.7.11) and the official `pipeline_tracking` custom-data API:
https://roicat.readthedocs.io/en/latest/roicat.html#roicat.pipelines.pipeline_tracking

## Select sessions

Edit `scripts/configs/roicat_71R_example.json`. The configuration selects all 68 runs across the 16 71R sessions from 216 through 231 inclusive, plane0, ordered by session then run. Each run is treated as a separate recording by ROICaT; matching is performed jointly across all 68 recordings. Add entries in the desired session order to track more sessions. Only listed entries are loaded. Compare the same animal and anatomical field/plane; run names alone do not ensure the same field.

- `root`: subject directory containing `ses-*` folders.
- `sessions`: explicit session, run, and plane for each recording. Each entry can override `image_file` or `roi_file`.
- `image_file`: single-frame mean image for alignment. Default is the native raw green TIFF. A red structural image can be used if it has exactly the same pixel coordinates as the green masks.
- `roi_file`: Cellpose `_seg.npy`, a 2D label-mask `.npy`, or Suite2p `stat.npy`. Only load trusted object files.
- `cells_only`: for Suite2p only; requires `iscell.npy` and selects rows whose first column is 1. Cellpose uses all positive labels.
- `um_per_pixel`: **approximately 0.988 µm/pixel**, the arithmetic mean of the x and y scales. The reported FOV is interpreted as **466 µm in x × 500 µm in y**, across the exported **501 × 478 pixels**, giving **0.930 µm/pixel in x** and **1.046 µm/pixel in y**. Both axis scales are recorded in the configuration. This workflow supplies ROICaT one isotropic approximation; it does not resample images or footprints to correct anisotropy. Confirm acquisition calibration/cropping if the FOV dimensions refer to the uncropped acquisition. All sessions must share the calibration.
- `use_gpu`: false by default for this Mac.

The default alignment is phase-correlation translation followed by DeepFlow nonrigid registration. This accommodates both global and local offsets. Set `roicat_overrides.alignment.fit_nonrigid.method` to `null` for translation only. Both stages use session index 0 as their template. Review registration and match-quality diagnostics: deformation and cell similarity can produce incorrect matches. These scripts do not reuse manual notebook shifts; ROICaT registers original images itself.

## Run

From the repository root:

```bash
conda activate roicat

# Validate source files, ROI counts, coordinates, and matching image dimensions.
python scripts/prepare_roicat_sessions.py --config scripts/configs/roicat_71R_example.json

# Validate the complete ROICaT input object without tracking or downloading models.
python scripts/run_roicat_sessions.py --config scripts/configs/roicat_71R_example.json --output results/roicat/71R_216_231_all_runs --validate-only

# Run tracking and save results. Choose a new output directory for every run.
python scripts/run_roicat_sessions.py --config scripts/configs/roicat_71R_example.json --output results/roicat/71R_216_231_all_runs
```

If conda activation is unavailable, use `/Users/megyoung/anaconda3/envs/roicat/bin/python` instead of `python`. The full run may download pretrained ROInet weights and can take substantial time on CPU. No session source files are changed. An existing output directory is refused; after a failed run, choose a new directory and retain the failed run's logs/files for diagnosis.

For offline input preparation or to avoid repeatedly loading network-drive data:

```bash
python scripts/prepare_roicat_sessions.py --config scripts/configs/roicat_71R_example.json --output results/roicat/prepared_71R_216_231_all_runs
python scripts/run_roicat_sessions.py --prepared results/roicat/prepared_71R_216_231_all_runs --output results/roicat/tracked_71R_216_231_all_runs
```

Prepared data captures the masks at preparation time. Prepare again after curation changes. `--um-per-pixel VALUE` overrides the saved configuration at tracking time.

## Outputs

- `roi_matches.csv`: one row per input ROI, including session/run, original mask label or zero-based Suite2p stat index, and ROICaT cluster ID. The same nonnegative cluster ID across sessions identifies a proposed cell match. **-1 means unassigned, not one shared cell.** Cluster IDs apply only within a run and are not confidence scores.
- `match_summary.json`: ROI, cluster, multi-session cluster, and unassigned counts.
- `input_manifest.json`: exact file paths, session ordering, original IDs, effective pixel scale, and ROICaT version.
- `requested_parameters.json`: full pipeline configuration requested.
- ROICaT's native `sessions.tracking.*` outputs: cluster labels/quality metrics, parameters, full results, and run data.
- `visualization/`: the pipeline's alignment and clustering diagnostic images/animations.

Inputs are sparse ROI footprints; no imaging movies are loaded. Cellpose footprints are uniform within each mask, while Suite2p footprints retain `lam` weights when available. Both are normalized to unit sum per ROI. No implicit resizing or cropping is performed.

## Manual alignment after segmentation curation

The adapted `prepare_roicat_inputs_meg.py`, `preview_roicat_mask_shift.py`, and
`manual_shift_roicat_inputs_meg.py` live together in `scripts/`. These versions
use this project's prepared sparse footprints and mean images, not Arya's
`stat.npy`/`ops.npy` folder format. Cellpose labels and Suite2p original indices
are preserved. Source segmentation and TIFF files are never edited.

Run from the repository root, with the `roicat` conda environment active:

```bash
# Prepare again after saving curated masks. Output directories must be new.
python scripts/prepare_roicat_inputs_meg.py \
  --config scripts/configs/roicat_71R_example.json \
  --out-root results/roicat/curated_inputs

# First pass, with the existing automatic registration settings.
python scripts/run_roicat_sessions.py \
  --prepared results/roicat/curated_inputs \
  --output results/roicat/curated_first_pass

# Example candidate only: replace the recording and offsets after inspection.
python scripts/preview_roicat_mask_shift.py \
  --input-root results/roicat/curated_inputs \
  --reference-session ses-216_date-20260730 --reference-run run-01 \
  --session ses-217_date-20260731 --run run-01 \
  --dx 0 --dy 0 \
  --out results/roicat/previews/ses217_run01_dx0_dy0.png
```

Positive dx moves right; positive dy moves down. Previews show original prepared
coordinates, not first-pass ROICaT's aligned coordinates. Use a consistent
reference across previews. Plane defaults to plane0; `--reference-plane` and
`--plane` select other planes. Keeping or deleting previews does not apply shifts.

Create a CSV (for example `results/roicat/chosen_shifts.csv`) with one row per
recording needing a shift. The following zero offset is a format example, not an
estimated correction:

```csv
session,run,dx,dy
ses-217_date-20260731,run-01,0,0
```

The optional `plane` column disambiguates multiple planes. Unknown or duplicate
recordings and noninteger shifts are rejected. Recordings not listed are copied
unchanged. Partial edge clipping is reported and surviving footprint weights are
renormalized. A shift removing any whole ROI is rejected so trace/mask identities
cannot silently disappear; choose a smaller shift or revisit the FOV selection.

```bash
python scripts/manual_shift_roicat_inputs_meg.py \
  --input-root results/roicat/curated_inputs \
  --out-root results/roicat/curated_shifted_inputs \
  --shift-csv results/roicat/chosen_shifts.csv --apply

python scripts/run_roicat_sessions.py \
  --prepared results/roicat/curated_shifted_inputs \
  --output results/roicat/curated_shifted_tracking
```

Manual shifts move the mean image and footprints together before ROICaT's normal
registration. The shifted manifest records offsets and clipped pixel counts;
source paths in its session records remain provenance for the original files.
Always use `--prepared` to track the shifted inputs: using `--config` would reload
the unshifted originals. Revised offsets must be applied to the original prepared
folder, not a previously shifted folder. Review the final registration diagnostics
and matches before analysis.
