"""Raw whole-FOV fluorescence and baseline-sensitivity check for each plane.

    uv run python scripts/nlightg2_whole_fov_baseline_check.py results/passive2

``export_root`` is the ``out_root`` given to nlightg2_extract_traces.py. Reads
the saved traces; writes a figure, metrics.csv and block_gains.csv to
``<export_root>/<planeN>/whole_fov_baseline_check/``.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nlight import plotting, suite2p_io, workflows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--fixed-percentiles", type=float, nargs="+", default=[10])
    parser.add_argument("--windows", type=float, nargs="+", default=[180, 300, 600],
                        help="longer rolling-baseline windows to compare (s)")
    parser.add_argument("--slow-timescale", type=float, default=10.0,
                        help="rolling-mean width defining the slow component (s)")
    parser.add_argument("--blocks", type=int, default=5,
                        help="contiguous blocks for the amplitude-gain check")
    args = parser.parse_args()
    plotting.use_lab_style()
    for plane_dir in suite2p_io.plane_dirs(args.export_root, marker="whole_image_roi_dff.npz"):
        out = workflows.save_baseline_check(
            plane_dir, fixed_percentiles=tuple(args.fixed_percentiles),
            long_windows_s=tuple(args.windows), slow_timescale_s=args.slow_timescale,
            n_blocks=args.blocks)
        print("saved", out)


if __name__ == "__main__":
    main()
