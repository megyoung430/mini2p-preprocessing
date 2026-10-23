# ΔF/F and frame-by-frame behavior exports

Use the mini2p-preprocessing environment. From this repository:

```bash
conda activate mini2p-preprocessing
cd /Users/megyoung/mini2p-preprocessing

# All runs in sessions 216–231: calculate and save fractional ΔF/F beside F/Fneu.
python scripts/run_recording_exports.py dff --save-to-plane

# Optional: list the runs eligible for alignment without creating full files.
python scripts/run_recording_exports.py align --dry-run

# Export one full frame-table PKL per confirmed recording.
python scripts/run_recording_exports.py align --write-full
```

To regenerate previews only, omit --write-full. For one recording:

```bash
python scripts/calculate_run_dff.py --session 216 --run 4 --save-to-plane
python scripts/align_run_dff_behavior.py --session 216 --run 4
python scripts/align_run_dff_behavior.py --session 216 --run 4 --write-full
```

## Saved ΔF/F

Exactly the check notebook's whole-recording baseline calculation:

- F_corrected = F − ops['neucoeff'] × Fneu (default coefficient 0.7).
- F0 = each ROI's 20th percentile of corrected F across the whole recording.
- dff = (F_corrected − F0) / F0.
- Nonpositive F0 or any nonfinite corrected input makes that ROI's ΔF/F NaN.

These are fractional ΔF/F values, not raw fluorescence and not z-scores.
No smoothing, red-reference regression, artifact interpolation, or trial baseline
subtraction is applied. Both channels and all ROIs retain their original order.
With --save-to-plane, each run's suite2p/plane0 gains:

- dff.npy / dff_chan2.npy (ROIs × frames)
- dff_F0.npy / dff_F0_chan2.npy
- dff_valid.npy / dff_valid_chan2.npy
- dff_metadata.json

Existing F.npy, Fneu.npy, stat.npy and ops.npy are not changed. Rerunning replaces
these derived dff files. Local copies are also saved under
output/frame_behavior/ses-N_run-NN/dff/ with functional_/channel2_ prefixes.

## CSV matching

The default mapping is output/frame_behavior/behavior_matches_sessions216-231_reviewed.csv.
Use --matches-csv PATH to select another audited mapping. Only CONFIRMED rows
are aligned. Missing entries, timestamp-only candidates, and no-overlap entries
are skipped before reading traces or producing an aligned PKL. Duplicate CSV
rows and missing paths in a confirmed row raise errors rather than silently skip.
The selected pair is revalidated with rsync and acquisition timestamps; filenames
are never matched by behavior run order. Inputs no longer passing validation fail.

The TIFF acquisition epoch is interpreted in Europe/London, TSV info/start_time
in UTC. Single-run CLI options permit overriding these timezone assumptions.
The source TIFF comes from source_tiff.json or the phase.json preprocessing log.
The CSV chooses the behavior TSV and HDF5 clock file.

## Frame-table PKL

Each SUBJECT/ses-N_date-YYYYMMDD/behav/run-NN/imaging_frames_behavior.pkl is a pandas
DataFrame. One row per imaging frame, with:

- frame, imaging_time_s, behavior_time_s, alignment_valid
- behavior_state: most recent state entry at or before that frame
- event:NAME / state:NAME: counts assigned to nearest covered frame
- events: list of records preserving exact event timestamps and signed offsets
- trial, blocks, trials_in_block, choice, chosen_rank, reward_pulses,
  chosen_magnitude, and other outputs of magnitude-bandit-analysis extract_trials
- dictionary-derived columns such as reward_magnitude_by_arm.A1,
  reward_magnitude_by_rank.best, choices_by_tower.A1 and choices_by_rank.best
- task.*: additional raw logged trial variables, with nested dictionaries flattened
- trial_index, trial_start_s, trial_end_s, trial_logged_s, trial_outcome_logged
- dff_functional_roi_0000 etc. and dff_channel2_roi_0000 etc.

Trial columns are produced by the actual extract_trials implementation in
/Users/megyoung/magnitude-bandit-analysis. Its port-finding exclusions, aliases,
and derived features are retained. Canonical records are matched back to the
raw print rows by their content, not by blindly pairing row numbers.

Each completed trial labels frames from its preceding wait_init entry up to the
next wait_init (or recording end). This is retrospective annotation: eventual
choice/reward values also label pre-choice frames within that trial. They are
not evidence that reward occurred then. trial_outcome_logged is false before
the print record and true afterward. Use timestamped state/event columns for
event timing. Incomplete trials without print records remain unknown, and
ambiguous trial boundaries stop export rather than shift trial assignments.

Uncovered frames have unknown task variables and NaN event counts, not zeros.
A single extra unclocked final imaging frame is retained with NaN timing;
other frame-count mismatches fail. No extrapolation beyond matched sync pulses.

The pickle contains fractional ΔF/F. The alignment script recomputes using the
same default calculation function instead of trusting potentially stale cache
files. Use the default baseline settings in both stages for identical values.
Source paths, ROI classification, baseline settings and alignment QC are in
DataFrame.attrs. Load only trusted pickle files:

```python
import pandas as pd
frames = pd.read_pickle('/Volumes/behrens/meg/3x3_field_magnitude_bandit/derivatives/cohort-01/sub-03_id-MY_71_R/ses-216_date-20260730/behav/run-04/imaging_frames_behavior.pkl')
print(frames[['frame', 'trial', 'choice', 'reward_pulses', 'dff_functional_roi_0000']].head())
```

## Batch behavior

The launcher discovers actual run folders in sessions 216–231. Change the range
with --first-session and --last-session. It continues after a per-run failure,
saves batch_dff_report.json or batch_align_report.json, and returns nonzero if
any run fails. Skipped unconfirmed alignments are reported separately. Existing
full PKLs for skipped runs are not deleted or replaced. --dry-run writes a separate
batch report and does not create data outputs.

Alignment previews and metadata are saved alongside the PKL in the session/behav/run folder. An explicit --output-dir overrides this location. Batch reports remain local by default.
