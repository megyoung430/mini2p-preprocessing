"""Prepare curated Cellpose/Suite2p inputs for the local ROICaT runner."""
from pathlib import Path
import argparse
from prepare_roicat_sessions import load_config, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/roicat_71R_example.json')
    parser.add_argument('--out-root', required=True, type=Path, help='New prepared input directory.')
    args = parser.parse_args()
    prepare(load_config(args.config), args.out_root)
    print(f'Prepared inputs: {args.out_root}')


if __name__ == '__main__':
    main()
