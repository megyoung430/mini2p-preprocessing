"""Preview a translation between two prepared ROICaT recordings; no input edits."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from manual_shift_roicat_inputs_meg import load_inputs, resolve_record, shift_record


def mask(sf, shape):
    return np.asarray(sf.sum(axis=0)).reshape(shape) > 0


def overlay(ref, moving):
    rgb = np.repeat((ref * .75)[..., None], 3, axis=2)
    rgb[..., 0] = np.maximum(rgb[..., 0], moving)
    rgb[..., 1:] *= 1 - .5 * moving[..., None]
    return rgb


def centroids(sf, shape):
    y, x = np.indices(shape)
    total = np.asarray(sf.sum(axis=1)).ravel()
    return np.asarray(sf @ x.ravel()).ravel() / total, np.asarray(sf @ y.ravel()).ravel() / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', required=True, type=Path)
    p.add_argument('--reference-session', required=True)
    p.add_argument('--reference-run', required=True)
    p.add_argument('--reference-plane', default='plane0')
    p.add_argument('--session', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--plane', default='plane0')
    p.add_argument('--dx', required=True, type=int, help='Positive = right.')
    p.add_argument('--dy', required=True, type=int, help='Positive = down.')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    manifest, images, footprints = load_inputs(a.input_root)
    ri = resolve_record(manifest, a.reference_session, a.reference_run, a.reference_plane)
    mi = resolve_record(manifest, a.session, a.run, a.plane)
    if images[ri].shape != images[mi].shape:
        raise ValueError('Reference and moving image dimensions differ.')
    _, shifted, clipped = shift_record(images[mi], footprints[mi], a.dx, a.dy)
    shape = images[ri].shape
    ref = mask(footprints[ri], shape)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(overlay(ref, mask(footprints[mi], shape)))
    axes[0].set_title('Before: reference grey, moving red')
    axes[1].imshow(overlay(ref, mask(shifted, shape)))
    axes[1].set_title(f'After dx={a.dx}, dy={a.dy}; clipped pixels={clipped}')
    axes[2].imshow(images[ri], cmap='gray')
    axes[2].scatter(*centroids(footprints[ri], shape), s=10, c='cyan', label='reference')
    axes[2].scatter(*centroids(shifted, shape), s=10, c='red', label='shifted')
    axes[2].legend()
    axes[2].set_title('Centroids on reference mean image')
    for ax in axes:
        ax.set_xlim(-.5, shape[1] - .5)
        ax.set_ylim(shape[0] - .5, -.5)
        ax.axis('off')
    fig.suptitle(f'{a.session}/{a.run} → {a.reference_session}/{a.reference_run}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180)
    plt.close(fig)
    print(a.out)


if __name__ == '__main__':
    main()
