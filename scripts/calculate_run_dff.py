"""Calculate the same whole-recording percentile ΔF/F as check_dff_ses216_run01.

Example: python scripts/calculate_run_dff.py --session 216 --run 4
Writes derived arrays locally; never changes plane0 inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np

SUBJECT = Path('/Volumes/behrens/meg/3x3_field_magnitude_bandit/derivatives/cohort-01/sub-03_id-MY_71_R')
OUTPUT = Path(__file__).resolve().parents[1] / 'output' / 'frame_behavior'


def find_plane(subject, session, run):
    matches = sorted(subject.glob(f'ses-{session}_*/funcimg/run-{run:02d}/suite2p/plane0'))
    if len(matches) != 1:
        raise ValueError(f'Expected one plane folder; found {matches}')
    return matches[0]


def calculate_dff(F, Fneu, coefficient, percentile=20, min_f0=0):
    F = np.asarray(F, dtype=np.float64)
    Fneu = np.asarray(Fneu, dtype=np.float64)
    if F.ndim != 2 or F.shape != Fneu.shape or 0 in F.shape:
        raise ValueError(f'Expected matching ROI × frame arrays: {F.shape}, {Fneu.shape}')
    if not 0 <= percentile <= 100 or not np.isfinite(min_f0) or min_f0 < 0 or not np.isfinite(coefficient):
        raise ValueError('Invalid baseline parameters')
    corrected = F - coefficient * Fneu
    finite = np.isfinite(corrected).all(axis=1)
    F0 = np.full((len(F), 1), np.nan)
    F0[finite] = np.percentile(corrected[finite], percentile, axis=1, keepdims=True)
    valid = finite & (F0[:, 0] > min_f0)
    dff = np.full_like(corrected, np.nan)
    np.divide(corrected - F0, F0, out=dff, where=valid[:, None])
    return {'dff': dff, 'F0': F0, 'valid': valid}


def calculate_run(plane, percentile=20, min_f0=0):
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    stat = np.load(plane / 'stat.npy', allow_pickle=True)
    iscell = np.load(plane / 'iscell.npy')
    if iscell.shape != (len(stat), 2):
        raise ValueError('iscell/stat ROI counts differ')
    results = {}
    for label, suffix in [('functional', ''), ('channel2', '_chan2')]:
        paths = [plane / f'{prefix}{suffix}.npy' for prefix in ('F', 'Fneu')]
        if label == 'channel2' and not any(p.exists() for p in paths):
            continue
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError(f'Incomplete fluorescence pair: {paths}')
        results[label] = calculate_dff(*(np.load(p) for p in paths),
                                      float(ops.get('neucoeff', 0.7)), percentile, min_f0)
        if results[label]['dff'].shape[0] != len(stat):
            raise ValueError('Fluorescence/stat ROI counts differ')
    if len({r['dff'].shape for r in results.values()}) != 1:
        raise ValueError('Channel shapes differ')
    metadata = dict(plane=str(plane), baseline='whole-recording percentile',
                    percentile=percentile, min_f0=min_f0, neucoeff=float(ops.get('neucoeff', .7)),
                    fs=float(ops['fs']), functional_chan=ops.get('functional_chan'))
    return results, metadata, iscell


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=int, required=True)
    p.add_argument('--run', type=int, required=True)
    p.add_argument('--subject-dir', type=Path, default=SUBJECT)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--percentile', type=float, default=20)
    p.add_argument('--min-f0', type=float, default=0)
    p.add_argument('--save-to-plane', action='store_true', help='Also save dff.npy/dff_chan2.npy alongside F.npy (does not alter F/Fneu)')
    a = p.parse_args()
    plane = find_plane(a.subject_dir, a.session, a.run)
    results, metadata, iscell = calculate_run(plane, a.percentile, a.min_f0)
    out = a.output_dir / f'ses-{a.session}_run-{a.run:02d}' / 'dff'
    out.mkdir(parents=True, exist_ok=True)
    for label, values in results.items():
        for key, value in values.items():
            np.save(out / f'{label}_{key}.npy', value)
        print(label, values['dff'].shape, 'valid baselines:', int(values['valid'].sum()))
    np.save(out / 'iscell.npy', iscell)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    if a.save_to_plane:
        for label, values in results.items():
            suffix = '' if label == 'functional' else '_chan2'
            np.save(plane / f'dff{suffix}.npy', values['dff'])
            np.save(plane / f'dff_F0{suffix}.npy', values['F0'])
            np.save(plane / f'dff_valid{suffix}.npy', values['valid'])
        (plane / 'dff_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
        print('Saved fractional ΔF/F beside F.npy:', plane)
    print(out)


if __name__ == '__main__':
    main()
