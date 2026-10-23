"""Whole-FOV dF/F of several recordings under identical baseline definitions.

    uv run python scripts/nlightg2_compare_recordings.py results/comparisons/airpuff \
        --recording "no puff=/Volumes/.../raw_data_2x/suite2p" \
        --recording "puff=/Volumes/.../raw_data_2x_puff/suite2p"

Each ``--recording`` is ``name=folder``, where the folder holds ``planeN/``
subfolders with ``whole_image_roi_dff.npz`` (a Suite2p root made by the old
notebook, or an ``out_root`` from nlightg2_extract_traces.py). Inputs are only
read; the figure and CSVs go to ``out_dir``.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nlight import plotting, recording_comparison, workflows  # noqa: E402


def name_and_path(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected name=folder, got {text!r}")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--recording", type=name_and_path, action="append", required=True)
    parser.add_argument("--window", type=float, default=60.0, help="current rolling window (s)")
    parser.add_argument("--percentile", type=float, default=50.0, help="rolling percentile")
    parser.add_argument("--fixed-percentiles", type=float, nargs="+", default=[10])
    parser.add_argument("--windows", type=float, nargs="+", default=[180, 300, 600])
    parser.add_argument("--slow-timescale", type=float, default=10.0)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    if len(args.recording) < 2:
        parser.error("give at least two --recording arguments")
    spec = recording_comparison.BaselineSpec(args.window, args.percentile,
                                             tuple(args.fixed_percentiles), tuple(args.windows))
    plotting.use_lab_style()
    out = workflows.save_recording_comparison(dict(args.recording), args.out_dir, title=args.title,
                                              spec=spec, slow_timescale_s=args.slow_timescale)
    print("saved", out)


if __name__ == "__main__":
    main()
