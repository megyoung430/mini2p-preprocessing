"""Match raw TIFF timestamps + rsync, preview a frame table; save full PKL only with --write-full.

python scripts/align_run_dff_behavior.py --session 216 --run 4
Preview is the default. Requires numpy, pandas, tifffile, h5py.
TIFF epoch and filename dates are Europe/London; TSV info/start_time is UTC by default.
The shared rsync train must corroborate timestamps within --timestamp-tolerance seconds.
No behavior-run-number matching and no alignment from file modification times.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import h5py
import tifffile
from calculate_run_dff import SUBJECT, OUTPUT, find_plane, calculate_run
from align_scanimage_pycontrol import (read_scanimage_clocks, read_pycontrol,
    match_rsync_pulses, interpolate_with_nan, nearest_frame_indices)
from frame_trial_variables import attach_trial_variables

MATCH_CSV = OUTPUT / 'behavior_matches_sessions216-231_reviewed.csv'


def confirmed_csv_row(csv_path, session, run):
    mapping = pd.read_csv(csv_path)
    if mapping.duplicated(['session', 'run']).any():
        raise ValueError('Match CSV contains duplicate session/run rows')
    rows = mapping[(mapping.session == session) & (mapping.run == run)]
    if rows.empty or rows.iloc[0]['status'] != 'CONFIRMED':
        return None
    row = rows.iloc[0]
    if any(pd.isna(row.get(k)) for k in ['behavior_path', 'h5', 'imaging_tiff']):
        raise ValueError('Confirmed CSV row has missing source paths')
    return row


def match_csv_sources(row, tiff, nframes, args):
    if tiff.name != row['imaging_tiff']:
        raise ValueError('CSV imaging TIFF differs from current preprocessing provenance')
    tsv = Path(row['behavior_path'])
    if args.behavior_dir:
        tsv = args.behavior_dir / tsv.name
    h5 = tiff.parent / row['h5']
    clocks = read_scanimage_clocks(h5)
    if len(clocks['frame_times_scanimage_s']) not in (nframes, nframes-1):
        raise ValueError('CSV-selected H5 frame count does not match current fluorescence')
    raw, sync = read_pycontrol(tsv)
    ia, ib, errors = match_rsync_pulses(sync, clocks['sync_times_scanimage_s'])
    if len(ia) < 10:
        raise ValueError('CSV-selected recording no longer meets the 10-pulse requirement')
    scan, py = clocks['sync_times_scanimage_s'][ib], sync[ia]
    slope, intercept = np.polyfit(scan, py, 1)
    residual = np.abs(slope*scan+intercept-py)
    start = tiff_start(tiff, args.imaging_timezone)
    bs = behavior_start(raw, args.behavior_timezone)
    delta = bs.timestamp()+slope*clocks['frame_times_scanimage_s'][0]+intercept-start.timestamp()
    if abs(delta)>args.timestamp_tolerance or abs(slope-1)>.005 or residual.max()>.05:
        raise ValueError('CSV-selected pair failed timestamp/rsync revalidation')
    qc = dict(matched_pulses=len(ia), timestamp_error_s=delta,
              median_chunk_rms_ms=float(np.median(errors)*1000),
              max_affine_residual_ms=float(residual.max()*1000))
    return (tsv,raw,bs,h5,clocks,scan,py,qc), start


def source_tiff(plane):
    """Resolve explicit preprocessing provenance, never infer source from run order."""
    run_dir = plane.parents[1]
    manifest = run_dir / 'source_tiff.json'
    if manifest.is_file():
        return Path(json.loads(manifest.read_text())['source_tiff'])
    paths = {Path(json.loads(p.read_text())['input_tiff'])
             for p in run_dir.glob('*.phase.json') if not p.name.startswith('.')}
    if len(paths) != 1:
        raise ValueError(f'Expected one source TIFF in phase provenance: {paths}')
    return paths.pop()


def tiff_start(path, timezone):
    with tifffile.TiffFile(path) as tif:
        description = tif.pages[0].description
    m = re.search(r'epoch\s*=\s*\[([^\]]+)\]', description)
    if not m:
        raise ValueError(f'No acquisition epoch in {path}')
    y, mo, d, h, mi, sec = map(float, m[1].split())
    return datetime(int(y), int(mo), int(d), int(h), int(mi), tzinfo=ZoneInfo(timezone)) + timedelta(seconds=sec)


def behavior_start(raw, timezone):
    rows = raw.loc[raw['type'].eq('info') & raw['subtype'].eq('start_time'), 'content']
    if len(rows) != 1:
        raise ValueError('TSV must contain one info/start_time')
    value = datetime.fromisoformat(rows.iloc[0])
    return value.replace(tzinfo=ZoneInfo(timezone)) if value.tzinfo is None else value


def match_sources(tiff, behav_dir, nframes, fs, args):
    start = tiff_start(tiff, args.imaging_timezone)
    behavior = []
    audit = []
    for path in sorted(behav_dir.glob('*.tsv')):
        if path.name.startswith('.'):
            continue
        try:
            raw, sync = read_pycontrol(path)
            bs = behavior_start(raw, args.behavior_timezone)
            duration = pd.to_numeric(raw['time'], errors='coerce').max()
            # Overlap actual time intervals, not run IDs or the nearest filename.
            if bs.timestamp() <= start.timestamp() + nframes/fs and bs.timestamp()+duration >= start.timestamp():
                behavior.append((path, raw, sync, bs))
        except (ValueError, KeyError) as exc:
            audit.append({'tsv': str(path), 'rejected': str(exc)})
    matches = []
    for h5 in sorted(tiff.parent.glob('*.h5')):
        if h5.name.startswith('.'):
            continue
        with h5py.File(h5, 'r') as f:
            duration = f['FrameClock'].size / float(f.attrs['samplerate'])
        if not nframes/fs - 5 <= duration <= nframes/fs + 30:
            continue
        clocks = read_scanimage_clocks(h5)
        nf = len(clocks['frame_times_scanimage_s'])
        if nf not in (nframes, nframes-1):
            continue
        for path, raw, sync, bs in behavior:
            try:
                ia, ib, errors = match_rsync_pulses(sync, clocks['sync_times_scanimage_s'])
                if len(ia) < 10:
                    raise ValueError('Fewer than 10 matched pulses')
                scan = clocks['sync_times_scanimage_s'][ib]
                py = sync[ia]
                slope, intercept = np.polyfit(scan, py, 1)
                residual = np.abs(np.polyval([slope, intercept], scan)-py)
                inferred_first = bs.timestamp() + slope*clocks['frame_times_scanimage_s'][0] + intercept
                delta = inferred_first - start.timestamp()
                if abs(delta) > args.timestamp_tolerance or abs(slope-1) > .005 or residual.max() > .05:
                    raise ValueError(f'Failed timestamp/drift check: delta={delta:.3f}s, slope={slope:.6f}, max residual={residual.max():.4f}s')
                matches.append((path, raw, bs, h5, clocks, scan, py,
                                dict(matched_pulses=len(ia), timestamp_error_s=delta,
                                     median_chunk_rms_ms=float(np.median(errors)*1000),
                                     max_affine_residual_ms=float(residual.max()*1000))))
            except ValueError as exc:
                audit.append({'tsv': str(path), 'h5': str(h5), 'rejected': str(exc)})
    if len(matches) != 1:
        raise ValueError(f'Expected one timestamp + rsync match, found {len(matches)}. Audit: {audit}')
    return matches[0], start, audit


def frame_behavior(raw, frame_times):
    """Events assigned to nearest covered frame; repeats counted; exact event times retained."""
    valid = np.isfinite(frame_times)
    data = {'frame': np.arange(len(frame_times)), 'behavior_time_s': frame_times,
            'alignment_valid': valid}
    rows = raw[raw['type'].isin(['event', 'state'])].copy()
    rows['time'] = pd.to_numeric(rows['time'], errors='coerce')
    rows = rows[np.isfinite(rows['time'])].sort_values('time', kind='stable')
    rows['frame'] = nearest_frame_indices(rows['time'].to_numpy(), frame_times)
    records = [[] for _ in frame_times]
    for (kind, name), group in rows.groupby(['type', 'content'], sort=False):
        values = np.zeros(len(frame_times), dtype=float)
        indices = group.loc[group['frame'] >= 0, 'frame'].to_numpy(dtype=int)
        np.add.at(values, indices, 1)
        values[~valid] = np.nan  # Unknown, not absence of behavior.
        data[f'{kind}:{name}'] = values
    for row in rows.itertuples():
        if row.frame >= 0:
            records[row.frame].append(dict(type=row.type, name=row.content,
                                          time_s=float(row.time), offset_s=float(row.time-frame_times[row.frame])))
    data['events'] = records
    states = rows[rows['type'].eq('state')]
    active = np.full(len(frame_times), None, dtype=object)
    if len(states):
        idx = np.searchsorted(states['time'].to_numpy(), frame_times[valid], side='right') - 1
        positions = np.flatnonzero(valid)
        good = idx >= 0
        active[positions[good]] = states['content'].to_numpy()[idx[good]]
    data['behavior_state'] = active
    return pd.DataFrame(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=int, required=True)
    p.add_argument('--run', type=int, required=True)
    p.add_argument('--subject-dir', type=Path, default=SUBJECT)
    p.add_argument('--behavior-dir', type=Path)
    p.add_argument('--matches-csv', type=Path, default=MATCH_CSV)
    p.add_argument('--behavior-project', type=Path, default=Path('/Users/megyoung/magnitude-bandit-analysis'))
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Optional export root; default is SUBJECT/ses-*/behav/run-XX')
    p.add_argument('--imaging-timezone', default='Europe/London')
    p.add_argument('--behavior-timezone', default='UTC')
    p.add_argument('--timestamp-tolerance', type=float, default=5)
    p.add_argument('--write-full', action='store_true', help='Explicitly save the full frame DataFrame as PKL')
    a = p.parse_args()
    mapping = confirmed_csv_row(a.matches_csv, a.session, a.run)
    if mapping is None:
        print(f'SKIP session {a.session} run {a.run:02d}: no CONFIRMED match in {a.matches_csv}')
        return
    plane = find_plane(a.subject_dir, a.session, a.run)
    tiff = source_tiff(plane)
    # Calculate directly to avoid consuming stale cached ΔF/F from another configuration.
    results, metadata, iscell = calculate_run(plane)
    nframes = results['functional']['dff'].shape[1]
    match, start = match_csv_sources(mapping, tiff, nframes, a)
    tsv, raw, bs, h5, clocks, scan, py, qc = match
    scan_times = clocks['frame_times_scanimage_s']
    frame_times = interpolate_with_nan(scan_times, scan, py)
    if len(frame_times) == nframes-1:
        frame_times = np.append(frame_times, np.nan)
        scan_times = np.append(scan_times, np.nan)
    table = frame_behavior(raw, frame_times)
    table.insert(1, 'imaging_time_s', scan_times)
    table, trial_metadata = attach_trial_variables(table, raw, tsv, a.behavior_project)
    trace_columns = {f'dff_{label}_roi_{roi:04d}': values['dff'][roi]
                     for label, values in results.items() for roi in range(len(iscell))}
    table = pd.concat([table, pd.DataFrame(trace_columns)], axis=1)
    table.attrs = dict(metadata, session=a.session, run=a.run, tiff=str(tiff), h5=str(h5),
                       behavior_tsv=str(tsv), imaging_start=start.isoformat(),
                       behavior_start=bs.isoformat(), qc=qc, match_csv=str(a.matches_csv),
                       **trial_metadata,
                       iscell=iscell.tolist(),
                       valid_baselines={k:v['valid'].tolist() for k,v in results.items()},
                       event_assignment='nearest covered frame; counts preserve repeats; events retains exact timestamps',
                       unclocked_final_frame='retained with NaN timestamps if one extra trace frame',
                       trace_units='fractional delta F/F; no z-scoring')
    out = (a.output_dir / f'ses-{a.session}_run-{a.run:02d}' if a.output_dir
           else plane.parents[3] / 'behav' / f'run-{a.run:02d}')
    out.mkdir(parents=True, exist_ok=True)
    # Include first rows, reward-state events, and other non-sync behavioral events.
    event_rows = np.flatnonzero([any(e['name'] != 'rsync' for e in es) for es in table['events']])
    reward_rows = np.flatnonzero(table.get('state:choice', pd.Series(np.zeros(nframes))).fillna(0).to_numpy()>0)
    indices = np.unique(np.r_[np.arange(min(5,nframes)), event_rows[:5], reward_rows[:3]])
    preview = table.iloc[indices].copy()
    preview.to_csv(out / 'frame_table_preview.csv', index=False)
    cols = ['frame','imaging_time_s','behavior_time_s','alignment_valid','behavior_state']
    cols += [c for c in ['state:choice','event:poke_1','event:rsync'] if c in table]
    cols += [c for c in ['trial','blocks','trials_in_block','choice','chosen_rank',
                         'reward_pulses','chosen_magnitude','trial_outcome_logged'] if c in table]
    cols += list(trace_columns)[:3]
    compact = preview[cols]
    (out / 'frame_table_preview.html').write_text(compact.to_html(index=False, float_format=lambda x:f'{x:.4f}'))
    summary = dict(table.attrs, n_frames=len(table), n_columns=len(table.columns),
                   aligned_frames=int(table.alignment_valid.sum()), columns=table.columns.tolist())
    (out / 'alignment_preview_metadata.json').write_text(json.dumps(summary,indent=2,default=str))
    print(f'Matched TIFF: {tiff.name}\nMatched H5: {h5.name}\nMatched TSV: {tsv.name}')
    print(json.dumps(qc,indent=2))
    print(f'Table: {table.shape}; covered frames: {table.alignment_valid.sum()}')
    print(compact.to_string(index=False))
    print(f'Preview: {out / "frame_table_preview.html"}')
    if a.write_full:
        table.to_pickle(out / 'imaging_frames_behavior.pkl')
        print(f'Saved full PKL: {out / "imaging_frames_behavior.pkl"}')
    else:
        print('Preview only. Full PKL NOT created. Use --write-full after reviewing.')


if __name__ == '__main__':
    main()
