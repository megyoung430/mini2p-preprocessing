# nLightG2 preprocessing (`src/nlight`)

```bash
uv sync                # creates the environment from pyproject.toml / uv.lock
uv run pytest          # synthetic-data tests; no lab data needed
```

| module | layer | does |
|---|---|---|
| `baseline.py` | analysis | rolling / fixed percentile F₀, ΔF/F, bad-frame masking |
| `tiles.py` | analysis | tile grid, tile means, raw whole-FOV mean, brightness screen |
| `traces.py` | analysis + thin I/O | `TraceSettings`; `compute_*` (pure) and `extract_*` (plane folder → exports) |
| `baseline_comparison.py` | analysis | candidate baselines; slow-fluctuation and relative-amplitude preservation metrics |
| `suite2p_io.py` | I/O | read `ops.npy` / registered binary; write the exports downstream scripts read |
| `recording_comparison.py` | analysis | plane-mean ΔF/F under shared baselines; cross-recording table, paired differences |
| `behavior.py` | I/O + analysis | row-aligned behaviour table, event onsets, pupil/speed |
| `events.py` | analysis | event windows, trial mean/SEM, prestimulus subtraction |
| `state.py` | analysis | pupil lag correlation, smoothing, tuning, running state |
| `analyses.py` | analysis | the above applied per baseline method |
| `session.py` | I/O | recordings config, loading a recording, cached tile ΔF/F per baseline |
| `analysis_plots.py` | plotting | panels for notebooks 02–04 |
| `plotting.py` | plotting | panel functions taking an `ax`; composite baseline-check figure |
| `workflows.py` | glue | plane-level entry points used by scripts and notebooks |

Notebooks: `notebooks/nlight/00…05`. See `NOTEBOOKS.md` for what each does and how the old
notebooks map onto them. Recordings are configured in `configs/nlight_recordings.json`.
Command-line equivalents: `scripts/nlightg2_extract_traces.py`,
`scripts/nlightg2_whole_fov_baseline_check.py`, `scripts/nlightg2_compare_recordings.py`.

Suite2p folders are only read. Writing exports into a Suite2p detection folder is refused,
because the tile `F.npy` would overwrite Suite2p's ROI `F.npy`.
