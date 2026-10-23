# nLightG2 notebooks

## The new set: `notebooks/nlight/`

Run them in order. Each one starts with `RECORDING = '…'`, any name from
`configs/nlight_recordings.json`, plus the parameters it uses. Notebooks 02–04
compute every analysis under every baseline definition in `SPEC`: the current
60 s rolling p50, a fixed full-recording p10, and 180/300/600 s rolling
windows. That way each result can be read under each baseline choice.

| Notebook | What it does |
|---|---|
| `00_extract_traces` | Reads Suite2p `ops.npy` + `data.bin`. Writes raw whole-FOV F and tile F (plus the ΔF/F the old scripts read) to the recording's `export_root`. Never writes into a Suite2p folder on the share. |
| `01_baseline_check` | Sarah's questions: raw whole-FOV F before ΔF/F, current vs fixed/long baselines, whether slow fluctuations are preserved (slow-component correlation and amplitude ratio), whether relative amplitudes are preserved (block gains). |
| `02_whole_recording` | Whole-FOV raw F and ΔF/F per baseline, tile heatmaps per baseline, zoomed window with tiles + whole-FOV side by side. What running removes, and saving running-regressed ΔF/F. |
| `03_event_aligned` | Grating 1 / Grating 2 / Reward: whole-FOV mean ± SEM per baseline (original and prestimulus-subtracted), tile heatmaps (rows = baseline, columns = event), response table, pupil and speed aligned to the same events. |
| `04_pupil_and_running` | Tile–pupil correlation maps per baseline, whole-FOV lag curves for pupil diameter and derivative, ΔF/F tuning by pupil size, running fraction / bouts / speed histogram, running vs stationary ΔF/F, run-onset and run-offset aligned ΔF/F. Then the pupil analyses again after regressing out running (lagged speed + running indicator; in-sample and held-out R², fitted kernel) and on stationary frames only. |
| `05_compare_recordings` | Airpuff vs no airpuff, corridor2 passive viewing vs baseline, corridor2 vs passive: identical baselines, plane-mean ΔF/F, and whether each difference survives every baseline. |

**`regress_running` flag** (notebooks 02–04): set `regress_running = True` to run every
analysis in that notebook on running-regressed ΔF/F. Running is regressed out of ΔF/F, not out
of pupil. The model is lagged running speed plus a running/stationary indicator, fitted
separately for each baseline method and each tile, and is defined once in `RUNNING_MODEL`
(threshold, bout length, lags, cross-validation). Originals stay available as
`s['wf_original']` and each tile's `dff_original`. Notebook 02 shows what running removes
(traces, kernels, variance explained, heatmaps before/after) and can save the regressed ΔF/F to
`<derived_root>/running_regressed/`. Notebook 03 overlays original and regressed event responses.
Regressed traces are missing within the lag range of the recording edges and wherever speed is
missing.

**Recordings** are listed in `configs/nlight_recordings.json`: Suite2p root,
export root and behaviour table for each. `{meg1}` is the share path. Override
it with the `NLIGHT_MEG1_ROOT` environment variable if the share is mounted
somewhere else. To add a recording, add an entry. `corridor2 baseline` has no
behaviour table listed yet; add its path if it has one.

**Caching:** tile ΔF/F under long rolling windows is slow, so it is cached in
`~/.cache/nlight` (or `NLIGHT_CACHE_DIR`). Deleting that folder only costs time.

## What the old notebooks did, and where it lives now

`scripts/nlightg2.ipynb` (4-plane passive / 2x / puff) and
`scripts/nlightg2_passive2.ipynb` (corridor2, single plane) run the same
sequence. passive2 reuses the existing registration and reads behaviour from
`all_data.pkl → aligned_data`.

| # | Old section | Now |
|---|---|---|
| 1 | Read ScanImage metadata; detect 1 vs 4 planes; per-plane frame rate | Not ported (registration stage). Still in the old notebook. |
| 2 | Consolidate TIFF parts into complete volumes (drop last partial volume) | Not ported. The behaviour loader reads its `*complete_volumes.json` record to allow exactly one extra table row. |
| 3 | Two-pass rigid Suite2p registration | Not ported. |
| 4 | Square-tile F, brightness exclusion, rolling ΔF/F, `F/F0/dff.npy` exports | `00` (`nlight.traces`). Same files, tested to reproduce `extract_big2p_tiles.py`. |
| 5 | All-tile ΔF/F heatmaps (grid order, viridis, shared scale) | `02`, per baseline |
| 6 | Whole-image ROI: average pixels first, then F₀ and ΔF/F | `00` (extract), `02` (plot) |
| 7 | Raw whole-FOV F and stable-baseline comparison (fixed p10, 180/300 s) | `01`, extended with slow/fast preservation metrics and block gains |
| 8 | Separate whole-image ROI plot and zoomed tile heatmaps (e.g. minutes 1–2 / 4–5) | `02`, zoom section (`ZOOM_MIN`) |
| 9 | Registered multi-plane video with tile grid | Not ported; old cell still works |
| 10 | Tile baseline sensitivity: rolling p20 vs Suite2p maximin at 30/60/120 s | Whole-FOV version in `01`. Tile-level maximin not ported (`compare_tile_baselines.py` still runs on the exports). |
| 11 | No-puff vs puff paired tile heatmaps | `05` (whole-FOV, all baselines). Paired tile heatmaps not ported. |
| 12 | Event-aligned tile ΔF/F: Grating 1 (Stim 2), Grating 2 (Stim 4), Reward | `03`, per baseline, plus whole-FOV means |
| 13 | Original vs per-trial prestimulus-subtracted (−2 to 0 s, 80% valid) | `03`, same rule, matched trials |
| 14 | Zero-lag tile–pupil correlation maps + whole-image scatter | `04` maps per baseline; whole-FOV r table |
| 15 | Histogram of tile–pupil correlations | `04` shows the median per map; histogram not ported |
| 16 | Circular-shift significance test + BH correction | Not ported (`test_pupil_correlations.py` still runs on the exports) |
| 17 | Trial z-scores, pupil-state split, Gaussian GLM with blocked CV | Not ported (`analyze_passive_pupil_events.py`) |
| 18 | Pupil diameter / derivative lag curves (mean ± IQR across tiles) | `04`, whole-FOV, per baseline |
| 19 | Lag curves for significant tiles; significant-tile outlined maps | Not ported (needs 16) |
| 20 | Per-tile lag search 0–16 s with shift-corrected peaks; peak-lag histogram | Not ported (`test_pupil_lag_range.py`, `plot_positive_peak_lags.py`) |
| 21 | Event heatmaps sorted by pupil correlation | Not ported |
| 22 | Pupil diameter aligned to events | `03` (also running speed) |
| 23 | Whole-image ΔF/F tuning by pupil size, per plane and plane-averaged | `04` (plane-mean), per baseline |
| 24 | Inspection cells (`value_counts`, `all_data.pkl` keys, TIFF frame counts) | Dropped |
| 25 | Videos: cropped, reference-style, 2×/3× zoom, 4/8/20/40-frame smoothing, GRAB | Not ported; old cells still work |
| 26 | Passive vs airpuff combined-plane overview (FOV trace + matched tiles) | `05` |
| — | Locomotion (Sarah's running question) | New in `04` |
| — | Pupil analyses with running regressed out / stationary only | New in `04` |
| — | Running-regressed ΔF/F everywhere (`regress_running`), plus export | New in `02`–`04` |

**Running the not-yet-ported scripts:** they read the same export files. Pass
them the recording's `export_root`, and for corridor2 a DataFrame copy of
`all_data.pkl → aligned_data`, as passive2 did with `passive2_aligned_data.pkl`.
