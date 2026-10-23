"""Run ROICaT tracking on explicitly selected sessions and export original-ID matches."""
from pathlib import Path
import argparse
import csv
import importlib.metadata
import json
import os
import tempfile
import numpy as np
from scipy import sparse
from prepare_roicat_sessions import load_config, prepare


def build_data(images, footprints, um_per_pixel):
    if not isinstance(um_per_pixel, (float, int)) or isinstance(um_per_pixel, bool) or not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError('Set um_per_pixel to your measured positive pixel size in the config, or pass --um-per-pixel.')
    import roicat
    data = roicat.data_importing.Data_roicat(verbose=True)
    data.set_FOV_images(images)
    data.set_spatialFootprints(footprints, um_per_pixel=float(um_per_pixel))
    data.transform_spatialFootprints_to_ROIImages(out_height_width=(36, 36))
    if not data.check_completeness(verbose=False)['tracking']:
        raise ValueError('ROICaT input object is incomplete.')
    return data


def merge_params(defaults, overrides):
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f'Unknown ROICaT parameter: {key}')
        if isinstance(value, dict) and isinstance(defaults[key], dict):
            merge_params(defaults[key], value)
        else:
            defaults[key] = value


def export_matches(labels_by_session, sessions, output):
    if len(labels_by_session) != len(sessions):
        raise ValueError('Result session count differs from input manifest.')
    rows = []
    for labels, session in zip(labels_by_session, sessions):
        if len(labels) != session['n_rois']:
            raise ValueError('Result ROI count differs from input manifest.')
        for local_index, (roi_id, label) in enumerate(zip(session['roi_ids'], labels)):
            rows.append(dict(session_index=session['index'], session=session['session'], run=session['run'],
                             roi_index=local_index, original_roi_id=roi_id, roi_id_kind=session['roi_id_kind'],
                             cluster_id=int(label)))
    with (output / 'roi_matches.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = {}
    for row in rows:
        if row['cluster_id'] >= 0:
            counts.setdefault(row['cluster_id'], set()).add(row['session_index'])
    summary = dict(n_sessions=len(sessions), n_rois=len(rows), n_clusters=len(counts),
                   n_clusters_in_multiple_sessions=sum(len(v) > 1 for v in counts.values()),
                   n_unassigned=sum(r['cluster_id'] < 0 for r in rows))
    (output / 'match_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path)
    source.add_argument('--prepared', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New output directory (existing directories are refused).')
    parser.add_argument('--um-per-pixel', type=float, help='Override the config pixel size.')
    parser.add_argument('--validate-only', action='store_true', help='Build and check ROICaT input; do not download models or track.')
    args = parser.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'roicat-matplotlib'))
    if args.output.exists():
        parser.error('Output directory already exists; choose a fresh directory.')
    if args.config:
        config = load_config(args.config)
        images, footprints, manifest = prepare(config)
    else:
        manifest = json.loads((args.prepared / 'manifest.json').read_text())
        config = manifest['config']
        images = [np.load(args.prepared / f'fov_{i:03d}.npy', allow_pickle=False) for i in range(len(manifest['sessions']))]
        footprints = [sparse.load_npz(args.prepared / f'footprints_{i:03d}.npz') for i in range(len(images))]
    scale = args.um_per_pixel if args.um_per_pixel is not None else config.get('um_per_pixel')
    data = build_data(images, footprints, scale)
    import roicat
    params = roicat.util.get_default_parameters(pipeline='tracking')
    merge_params(params, config.get('roicat_overrides', {}))
    params['general']['use_GPU'] = config.get('use_gpu', False)
    params['general']['random_seed'] = 0
    params['results_saving']['dir_save'] = str(args.output.resolve())
    params['results_saving']['prefix_name_save'] = 'sessions'
    if args.validate_only:
        print('ROICaT data validation passed. No tracking or output writing performed.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest['effective_um_per_pixel'] = scale
    manifest['roicat_version'] = importlib.metadata.version('roicat')
    (args.output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'requested_parameters.json').write_text(json.dumps(params, indent=2) + '\n')
    # The official pipeline saves full results, aligned ROIs/images, and diagnostic figures.
    results, run_data, params_used = roicat.pipelines.pipeline_tracking(params=params, custom_data=data)
    summary = export_matches(results['clusters']['labels_bySession'], manifest['sessions'], args.output)
    print(json.dumps(summary, indent=2))
    print(f'Results saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
