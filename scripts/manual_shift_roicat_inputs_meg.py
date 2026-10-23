"""Shift prepared ROICaT footprints and FOV images, preserving original ROI IDs."""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def load_inputs(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    images, footprints = [], []
    for i, record in enumerate(manifest['sessions']):
        im = np.load(root / f'fov_{i:03d}.npy', allow_pickle=False)
        sf = sparse.load_npz(root / f'footprints_{i:03d}.npz').tocsr()
        if im.ndim != 2 or sf.shape != (len(record['roi_ids']), im.size):
            raise ValueError(f'Invalid prepared input dimensions at recording {i}')
        images.append(im)
        footprints.append(sf)
    return manifest, images, footprints


def resolve_record(manifest, session, run=None, plane=None):
    matches = [i for i, r in enumerate(manifest['sessions'])
               if r['session'] == session and (not run or r['run'] == run)
               and (not plane or Path(r['plane']).name == plane)]
    if len(matches) != 1:
        raise ValueError(f'{session}/{run or "*"}/{plane or "*"}: expected one recording, found {len(matches)}; specify run and plane.')
    return matches[0]


def shift_record(image, footprint, dx, dy):
    h, w = image.shape
    coo = footprint.tocoo()
    y, x = coo.col // w + dy, coo.col % w + dx
    keep = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    shifted = sparse.csr_matrix((coo.data[keep], (coo.row[keep], y[keep] * w + x[keep])), shape=coo.shape)
    sums = np.asarray(shifted.sum(axis=1)).ravel()
    if np.any(sums <= 0):
        raise ValueError(f'Shift would remove entire ROI(s), row indices {np.flatnonzero(sums <= 0).tolist()}; choose a smaller shift.')
    shifted = (sparse.diags(1 / sums) @ shifted).tocsr().astype(np.float32)
    out = np.full_like(image, np.nanmedian(image))
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = image[y0:y1, x0:x1]
    return out, shifted, int((~keep).sum())


def apply_shifts(input_root, out_root, shift_csv):
    input_root, out_root = Path(input_root).resolve(), Path(out_root).resolve()
    if out_root == input_root or input_root in out_root.parents or out_root.exists():
        raise ValueError('Output must be a new directory outside the input directory.')
    manifest, images, footprints = load_inputs(input_root)
    if manifest.get('manual_shifts'):
        raise ValueError('These inputs are already shifted; apply revised offsets to the original prepared inputs.')
    with Path(shift_csv).open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'session', 'run', 'dx', 'dy'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires session,run,dx,dy columns; plane is optional.')
        rows = list(reader)
    if not rows:
        raise ValueError('Shift CSV is empty.')
    applied, seen = [], set()
    for row in rows:
        i = resolve_record(manifest, row['session'], row['run'], row.get('plane'))
        if i in seen:
            raise ValueError(f'Duplicate shift for recording {i}')
        seen.add(i)
        dx, dy = int(row['dx']), int(row['dy'])
        images[i], footprints[i], clipped = shift_record(images[i], footprints[i], dx, dy)
        applied.append(dict(index=i, session=row['session'], run=row['run'], dx=dx, dy=dy, clipped_pixels=clipped))
        print(f'{row["session"]}/{row["run"]}: dx={dx}, dy={dy}, clipped footprint pixels={clipped}')
    manifest = copy.deepcopy(manifest)
    manifest['manual_shifts'] = applied
    manifest['prepared_source'] = str(input_root)
    out_root.mkdir(parents=True, exist_ok=False)
    for i, (im, sf) in enumerate(zip(images, footprints)):
        np.save(out_root / f'fov_{i:03d}.npy', im, allow_pickle=False)
        sparse.save_npz(out_root / f'footprints_{i:03d}.npz', sf)
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (out_root / 'manual_shifts_applied.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(applied[0]))
        writer.writeheader()
        writer.writerows(applied)
    print(f'Saved shifted inputs: {out_root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--out-root', required=True, type=Path)
    parser.add_argument('--shift-csv', required=True, type=Path)
    parser.add_argument('--apply', required=True, action='store_true')
    args = parser.parse_args()
    apply_shifts(args.input_root, args.out_root, args.shift_csv)


if __name__ == '__main__':
    main()
