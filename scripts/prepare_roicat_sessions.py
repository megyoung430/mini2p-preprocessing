"""Validate selected sessions and prepare sparse ROICaT inputs without running tracking."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from scipy import sparse


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    root = Path(config['root']).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    config['root'] = str(root.resolve())
    sessions = config['sessions']
    if len(sessions) < 2:
        raise ValueError('Choose at least two sessions.')
    keys = [(s['session'], s.get('run', 'run-01'), s.get('plane', 'plane0')) for s in sessions]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate session/run/plane entries.')
    return config


def load_image(path):
    with Image.open(path) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError(f'Expected a single-frame image: {path}')
        image = np.asarray(im).copy()
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].astype(float) @ [0.2126, 0.7152, 0.0722]
    if image.ndim != 2 or not np.isfinite(image).all() or np.ptp(image) == 0:
        raise ValueError(f'Expected a finite, nonconstant 2D image: {path}')
    return image.astype(np.float32)


def load_footprints(path, shape, cells_only=True):
    """Return CSR footprints and original IDs; never merge overlapping Suite2p ROIs."""
    raw = np.load(path, allow_pickle=True)
    if path.name == 'stat.npy':
        ids = np.arange(len(raw))
        if cells_only:
            flags = np.load(path.with_name('iscell.npy'), allow_pickle=False)
            if flags.ndim != 2 or flags.shape[0] != len(raw) or flags.shape[1] < 1:
                raise ValueError(f'iscell.npy does not match {path}')
            ids = ids[flags[:, 0] == 1]
        rows, cols, values = [], [], []
        for row, roi_id in enumerate(ids):
            roi = raw[roi_id]
            y, x = np.asarray(roi['ypix']), np.asarray(roi['xpix'])
            if y.ndim != 1 or x.shape != y.shape or not y.size:
                raise ValueError(f'Invalid coordinates: {path}, ROI {roi_id}')
            if not (np.isfinite(y).all() and np.isfinite(x).all() and
                    np.equal(y, np.floor(y)).all() and np.equal(x, np.floor(x)).all()):
                raise ValueError(f'Noninteger coordinates: {path}, ROI {roi_id}')
            if np.any((y < 0) | (y >= shape[0]) | (x < 0) | (x >= shape[1])):
                raise ValueError(f'ROI outside image: {path}, ROI {roi_id}')
            weights = np.asarray(roi.get('lam', np.ones(y.size)), dtype=np.float32)
            if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError(f'Invalid ROI weights: {path}, ROI {roi_id}')
            rows.extend([row] * len(y)); cols.extend(y.astype(int) * shape[1] + x.astype(int))
            values.extend(weights)
        sf = sparse.csr_matrix((values, (rows, cols)), shape=(len(ids), int(np.prod(shape))), dtype=np.float32)
    else:
        obj = raw.item() if raw.shape == () else raw
        masks = np.asarray(obj['masks'] if isinstance(obj, dict) else obj)
        if masks.shape != shape or not np.isfinite(masks).all() or np.any(masks < 0) or not np.equal(masks, np.floor(masks)).all():
            raise ValueError(f'Expected nonnegative integer labels with shape {shape}: {path}')
        ids = np.unique(masks[masks > 0]).astype(np.int64)
        pixels = np.flatnonzero(masks.ravel() > 0)
        rows = np.searchsorted(ids, masks.ravel()[pixels])
        sf = sparse.csr_matrix((np.ones(len(pixels), np.float32), (rows, pixels)),
                              shape=(len(ids), masks.size))
    if len(ids) == 0:
        raise ValueError(f'No ROIs selected: {path}')
    sf = sparse.diags(1 / np.asarray(sf.sum(axis=1)).ravel()) @ sf
    return sf.tocsr().astype(np.float32), ids


def prepare(config, output=None):
    images, footprints, records = [], [], []
    for index, entry in enumerate(config['sessions']):
        plane = Path(config['root']) / entry['session'] / 'funcimg' / entry.get('run', 'run-01') / 'suite2p' / entry.get('plane', 'plane0')
        image_path = plane / entry.get('image_file', config['image_file'])
        roi_path = plane / entry.get('roi_file', config['roi_file'])
        image = load_image(image_path)
        if images and image.shape != images[0].shape:
            raise ValueError(f'Image shapes differ: {image.shape} vs {images[0].shape}; no implicit resizing.')
        sf, ids = load_footprints(roi_path, image.shape, config.get('cells_only', True))
        records.append(dict(index=index, session=entry['session'], run=entry.get('run', 'run-01'),
                            plane=str(plane), image_path=str(image_path), roi_path=str(roi_path),
                            roi_id_kind='suite2p_stat_index' if roi_path.name == 'stat.npy' else 'mask_label',
                            roi_ids=ids.tolist(), n_rois=len(ids), shape=list(image.shape)))
        images.append(image); footprints.append(sf)
        print(f"{index}: {entry['session']} / {entry.get('run', 'run-01')}: {len(ids)} ROIs, {image.shape}")
    manifest = dict(config=config, sessions=records)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for i, (im, sf) in enumerate(zip(images, footprints)):
            np.save(output / f'fov_{i:03d}.npy', im, allow_pickle=False)
            sparse.save_npz(output / f'footprints_{i:03d}.npz', sf)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return images, footprints, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='New directory for prepared arrays; omit for validation only.')
    args = parser.parse_args()
    prepare(load_config(args.config), args.output)
    print('Input validation passed.')


if __name__ == '__main__':
    main()
