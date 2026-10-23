"""Tile and whole-FOV F / F0 / dF/F for every registered plane under a Suite2p root.

    uv run python scripts/nlightg2_extract_traces.py /Volumes/.../suite2p results/passive2 \
        --tile-size 32 --baseline-window 60 --baseline-percentile 50

Reads the Suite2p root; writes the exports the event, pupil and video scripts
read to ``<out_root>/<planeN>/``. The Suite2p folder is not modified.
"""
import argparse
from pathlib import Path

from nlight import suite2p_io, traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite2p_root", type=Path, help="Suite2p output (read only)")
    parser.add_argument("out_root", type=Path, help="where <planeN>/ exports are written")
    parser.add_argument("--fs", type=float, default=None,
                        help="verified per-plane rate (Hz); default: ops['fs']")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--exclude-below-percentile", type=float, default=0.0)
    parser.add_argument("--baseline-window", type=float, default=60.0, help="seconds")
    parser.add_argument("--baseline-percentile", type=float, default=20.0)
    args = parser.parse_args()
    settings = traces.TraceSettings(
        tile_size_px=args.tile_size, exclude_below_percentile=args.exclude_below_percentile,
        baseline_window_s=args.baseline_window, baseline_percentile=args.baseline_percentile)
    for plane_dir in suite2p_io.plane_dirs(args.suite2p_root):
        traces.extract_tiles(plane_dir, settings, args.fs, args.out_root)
        print("saved", traces.extract_whole_fov(plane_dir, settings, args.fs, args.out_root))


if __name__ == "__main__":
    main()
