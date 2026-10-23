# How the arrays were generated, and the comparison with session 177

The following is established from the **current repository code** (`Convert_seg_to_stat`, `convert_seg_to_stat_216_231.py`, `calculate_run_dff.py`), not a complete historical execution log.

1. **Inputs already exist.** The converter reads `ops.npy`, registered int16 `data.bin`, and (when present) `data_chan2.bin`. Registration, destriping and distortion correction happened upstream; this converter does not perform them.
2. **Cellpose labels → `stat`.** Each positive mask label becomes one ROI: `ypix`, `xpix`, `npix`, uniform `lam=1`, and mean-coordinate `med`. `roi_stats` fills in Suite2p ROI properties. Rows follow ascending positive label order; IDs need not correspond between runs.
3. **ROI and neuropil masks.** `create_masks` uses saved ops or fallbacks: `lam_percentile=50`, `allow_overlap=False`, inner exclusion radius 2 pixels, at least 350 neuropil pixels, `circular_neuropil=False`. Normalized ROI weights extract mean fluorescence; surrounding neuropil masks supply the background trace.
4. **Registered movies → fluorescence.** `extraction_wrapper` reads frames in batches on CPU using the same spatial masks for both channels. It produces raw ROI `F.npy`, surrounding `Fneu.npy`, and their `_chan2` equivalents. These saved arrays are NOT neuropil-subtracted. Check `functional_chan` rather than assuming file channel 1 always means physical green.
5. **Classification.** Suite2p's classifier produces `iscell.npy`; the converter does not discard non-cell rows from the fluorescence arrays.
6. **Neuropil correction for deconvolution.** `Fc = F − neucoeff × Fneu`. Missing flat ops entries receive defaults: `neucoeff=0.7`, baseline `maximin`, window 60 seconds, smoothing sigma 10 frames, percentile 8, `fs=7.52 Hz`, `tau=1 s`, batch size 500. Existing entries win. Defaults do not establish the actual acquisition settings.
7. **Baseline preprocessing → spikes.** `suite2p.extraction.preprocess(Fc, ...)` feeds `oasis`, whose result is saved as `spks.npy`. The inspected installed Suite2p source subtracts its estimated baseline, without dividing by F0. Neither converter saves its intermediate corrected trace, F0, or dF/F. `spks` is a deconvolution output, not dF/F.
8. **This comparison notebook calculates F0 and dF/F later.** It applies the explicit SciPy formula above to saved fluorescence. Red-reference regression and artifact treatment are further analysis steps, not part of mask conversion.
9. **The newer percentile export is separate.** `calculate_run_dff.py` computes one F0 per ROI: the 20th percentile of Fc over the whole recording, then `(Fc−F0)/F0`. Nonpositive F0 or nonfinite input makes the ROI's dF/F NaN. It can save `dff.npy`, `dff_F0.npy`, `dff_valid.npy`, channel-2 equivalents and `dff_metadata.json`. It applies no red regression, artifact replacement or smoothing. Therefore those saved dff files should not be expected to equal this notebook's maximin outputs.

| Aspect | Session 177 converter | Sessions 216–231 converter |
|---|---|---|
| Input path | Current active assignment is `destriped_distortion_uncorrected/suite2p/plane0`; the earlier striped assignment is overwritten | Each `funcimg/run-*/suite2p/plane0` |
| Segmentation | `meanImg_ch1_seg.npy` | Each run's `meanImg_green_functional_contrast_seg.npy` |
| Extraction and deconvolution | Same core calls/settings in current code | Same core calls/settings in current code |
| Label selection | `unique(masks)[1:]`, assumes background zero exists | Explicitly selects positive labels and validates masks |
| Channels and validation | Opens both binaries unconditionally | Supports one channel; checks frame sizes/counts and missing two-channel binary |
| Saving | Overwrites eight output files | Backs up eight existing outputs, then replaces them |

Different segmentation, registered input movies, saved ops, or Suite2p versions can change the values even when the core algorithm is shared. Folder names alone do not prove how the 216–231 movies were preprocessed. The converter does not update existing downstream dff files: if F was re-extracted later, previously saved dff may be stale and must be regenerated. The original comparison notebook also mixed its explicit normalized formula with installed `preprocess` in later method plots; this copy uses the normalized formula consistently.



### Saved settings inspected during notebook creation

Session 216/run-01 (420 ROIs × 4079 frames), session 231/run-03 (480 × 10609), and session 177's destriped folder (254 × 14001) all had `fs=7.52`, `neucoeff=0.7`, `baseline=maximin`, `win_baseline=60`, `sig_baseline=10`, `prctile_baseline=8`, `functional_chan=2`, and two channels. **`tau` was 1.0 s in the two newer runs and 0.4 s in session 177.** Tau changes spike deconvolution, not F extraction or the F0/dF/F formula here. These are spot checks, not a claim that all runs have identical ops.

Both sampled newer runs also have `dff_metadata.json` recording the whole-recording 20th-percentile method. The session-177 folder did not have that metadata file. In the acquisition notebook, physical channel 2 is green/functional and channel 1 is red/structural: `data.bin`/`F.npy` correspond to functional green, whereas the `_chan2` output suffix denotes the other Suite2p channel. Thus file-channel numbering in the comparison plots is not the acquisition-channel numbering.
