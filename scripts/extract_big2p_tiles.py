"""Tile fluorescence and rolling dF/F export alongside registered Suite2p binaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def extract_plane_tiles(plane_dir, fs, tile_size=16, batch_frames=500, exclude_below_percentile=10.0, baseline_window_s=60.0, baseline_percentile=20.0):

    plane_dir=Path(plane_dir)

    if tile_size < 1 or int(tile_size)!=tile_size or batch_frames < 1 or int(batch_frames)!=batch_frames:
        raise ValueError('Tile size and batch size must be positive integers')
    if not 0 <= exclude_below_percentile < 100:
        raise ValueError('Brightness percentile must be in [0, 100); 0 disables filtering')
    if not np.isfinite(fs) or fs<=0 or not np.isfinite(baseline_window_s) or baseline_window_s<=0:
        raise ValueError('Frame rate and baseline window must be positive and finite')
    if not 0<=baseline_percentile<=100:
        raise ValueError('Baseline percentile must be in [0, 100]')

    # Load the relevant Suite2p ops file and registered binary and validate the crop region.
    tile_size, batch_frames=int(tile_size), int(batch_frames)
    ops = np.load(plane_dir/'ops.npy', allow_pickle=True).item()

    movie_path = plane_dir/'data.bin'
    if not movie_path.is_file():
        movie_path=Path(ops['reg_file'])
    
    nt, h, w = int(ops['nframes']), int(ops['Ly']), int(ops['Lx'])
    y0, y1 = map(int,ops['yrange'])
    x0, x1 = map(int,ops['xrange'])
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError('Invalid crop')

    # Compute tile grid
    nr, nc = (y1-y0)//tile_size, (x1-x0)//tile_size
    if min(nr,nc) < 1:
        raise ValueError('No complete tiles')

    # Get the pixel boundaries for each tile
    bounds= np.array([[y0+r*tile_size, y0+(r+1)*tile_size, x0+c*tile_size, x0+(c+1)*tile_size] for r in range(nr) for c in range(nc)])


    F_all = np.empty((nr*nc,nt), np.float32)
    movie = np.memmap(movie_path, dtype=np.int16, mode='r', shape=(nt,h,w))
    for start in range(0, nt, batch_frames):
        stop = min(nt, start+batch_frames)
        block = np.asarray(movie[start:stop,y0:y0+nr*tile_size,x0:x0+nc*tile_size], dtype=np.float32)
        # Group pixels into frames x tile rows x pixels in tile height x tile columns x pixels in tile width, 
        # then average over the pixel axes to get mean tile fluorescence. Then,
        # flatten the tile rows and columns into a single tile axis, and transpose to get tiles x frames.
        F_all[:,start:stop] = block.reshape(stop-start,nr,tile_size,nc,tile_size).mean(axis=(2,4)).reshape(stop-start,-1).T
    del movie

    # Exclude bad frames
    bad = np.asarray(ops.get('badframes', np.zeros(nt,bool)), dtype=bool).ravel()
    if len(bad)!=nt or bad.all():
        raise ValueError('Invalid/all-bad frame mask')

    # Absolute fluorescence means raw intensity level, not abs(F) or abs(dff).
    # Calculate the mean fluorescence of each tile over all non-bad frames, then exclude tiles below the specified percentile.
    brightness = np.nanmean(F_all[:,~bad],axis=1)
    finite = np.isfinite(brightness)
    if not finite.any():
        raise ValueError('No finite tile brightness values')
    cutoff = float(np.percentile(brightness[finite], exclude_below_percentile))
    keep = finite & ((brightness>=cutoff) if exclude_below_percentile > 0 else True)
    ids = np.flatnonzero(keep)
    F = F_all[keep]

    # Compute rolling baseline and dF/F for each retained tile. 
    # The baseline is the specified percentile of a centered rolling window, ignoring bad frames and NaNs. 
    # The dF/F is computed as (F-F0)/F0, with NaN where F0 is non-positive or either F or F0 is NaN.
    window = max(3, int(round(baseline_window_s*fs)))
    window += window%2 == 0
    minimum = max(1, int(np.ceil(.25*min(window, nt))))
    F0 = np.full_like(F, np.nan)
    for i, trace in enumerate(F):
        a = trace.copy()
        a[bad | ~np.isfinite(a)] = np.nan
        # Calculate the baseline fluorescence as a 60s rolling window percentile, ignoring NaNs and bad frames. 
        # The rolling window is centered and requires a minimum number of valid samples.
        F0[i] = pd.Series(a).rolling(window, center=True, min_periods=minimum).quantile(baseline_percentile/100).to_numpy(dtype=np.float32)

    # Compute dF/F, with NaN where F0 is non-positive or either F or F0 is NaN.
    dff = np.full_like(F, np.nan)
    np.divide(F-F0, F0, out=dff, where=np.isfinite(F)&np.isfinite(F0)&(F0>0))
    dff[:,bad] = np.nan
    time_s = np.arange(nt)/fs

    # Save the results to disk. The retained tiles are saved in a single compressed NPZ file, along with the time axis and bad frame mask. 
    # The original tile IDs and bounds are also saved for auditing the exclusion.
    for name, array in [('F',F), ('F0',F0), ('dff',dff), ('tile_ids',ids), ('tile_bounds_yxyx',bounds[keep])]:
        np.save(plane_dir/f'{name}.npy', array)
    np.savez_compressed(plane_dir/'tile_fluorescence_dff.npz', F=F, F0=F0, dff=dff, time_s=time_s, badframes=bad, tile_ids=ids, tile_bounds_yxyx=bounds[keep])
    # Keep all brightness values and original bounds to audit the exclusion.
    np.savez_compressed(plane_dir/'tile_selection.npz', brightness=brightness, keep_mask=keep, all_tile_bounds_yxyx=bounds)
    metadata=dict(registered_binary = str(movie_path),
                  ops_path = str(plane_dir/'ops.npy'),
                  array_axes = ['retained_tiles','frames'],
                  tile_size_px = tile_size,
                  grid_shape = [nr,nc],
                  original_tile_count = len(keep),
                  retained_tile_count = int(keep.sum()),
                  brightness_exclusion_percentile = float(exclude_below_percentile),
                  brightness_threshold = float(cutoff),
                  brightness_statistic = 'time mean of raw fluorescence over non-bad frames, per plane; not abs(F)',
                  exclusion_rule = 'exclude brightness strictly below cutoff; ties retained; 0 disables filtering',
                  tile_ids = 'original row-major grid IDs, aligned with rows of F/F0/dff',
                  fs_hz = float(fs),
                  baseline_window_s_requested = float(baseline_window_s),
                  baseline_window_frames = window,
                  baseline_window_s_effective = window/fs,
                  baseline_percentile = float(baseline_percentile),
                  baseline_centered = True,
                  baseline_min_valid_samples = minimum,
                  bad_frame_policy = 'exclude baseline samples; dff NaN; preserve time axis',
                  dff_units = 'fraction',
                  neuropil_subtraction = False)
    (plane_dir/'tile_fluorescence_dff.json').write_text(json.dumps(metadata, indent=2))
    print(f'{plane_dir.name}: retained {keep.sum()}/{len(keep)} tiles; brightness cutoff {cutoff:.4g}', flush=True)
    return plane_dir/'tile_fluorescence_dff.npz'

def label_tile_axis(ax, plane_dir):
    """Map compact retained rows back to original spatial-grid tile IDs."""
    path = Path(plane_dir)/'tile_ids.npy'
    if not path.exists():
        return
    ids = np.load(path)
    loc = np.unique(np.linspace(0, len(ids)-1, min(7,len(ids)), dtype=int))
    ax.set_yticks(loc, labels=ids[loc])
    ax.set_ylabel('Tile ID (retained tiles)')
