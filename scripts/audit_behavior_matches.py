"""Read-only timestamp/rsync matching audit for MY_71_R sessions 216–231."""
import csv
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import align_run_dff_behavior as align
from calculate_run_dff import SUBJECT, OUTPUT

# Reuse session inputs across runs; cached clocks contain only extracted edges.
align.read_pycontrol = lru_cache(maxsize=32)(align.read_pycontrol)
align.read_scanimage_clocks = lru_cache(maxsize=32)(align.read_scanimage_clocks)
args = SimpleNamespace(imaging_timezone='Europe/London', behavior_timezone='UTC', timestamp_tolerance=5)
if not SUBJECT.is_dir():
    raise SystemExit(f'Subject folder unavailable. Mount the behrens drive first: {SUBJECT}')
rows = []
out = OUTPUT / 'behavior_matches_sessions216-231.csv'
out.parent.mkdir(parents=True, exist_ok=True)
previous = {}
if out.exists():
    with out.open() as f:
        previous = {(int(r['session']), int(r['run'])):r for r in csv.DictReader(f)
                    if r['status']=='CONFIRMED'}
fields = ['session','run','status','imaging_tiff','imaging_start','behavior_tsv','h5',
          'n_frames','matched_pulses','timestamp_error_s','median_chunk_rms_ms',
          'max_affine_residual_ms','behavior_path','detail']
for session in range(216,232):
    planes = sorted(SUBJECT.glob(f'ses-{session}_*/funcimg/run-*/suite2p/plane0'))
    if not planes:
        rows.append(dict(session=session, run='', status='MISSING_SESSION',
                         detail='No run plane0 folders found'))
    for plane in planes:
        run = int(plane.parents[1].name.split('-')[1])
        row = dict(session=session,run=run,status='UNRESOLVED')
        try:
            if (session, run) in previous:
                rows.append(previous[(session,run)])
                print(f'{session} run-{run:02d}: previously CONFIRMED',flush=True)
                continue
            tiff = align.source_tiff(plane)
            row['imaging_tiff'] = tiff.name
            row['imaging_start'] = align.tiff_start(tiff,args.imaging_timezone).isoformat()
            ops = np.load(plane/'ops.npy',allow_pickle=True).item()
            F = np.load(plane/'F.npy',mmap_mode='r')
            nframes = F.shape[1]
            del F
            row['n_frames'] = nframes
            match, start, audit = align.match_sources(tiff,tiff.parent.parent/'behav',nframes,float(ops['fs']),args)
            path, raw, bs, h5, clocks, scan, py, qc = match
            row.update(status='CONFIRMED',behavior_tsv=path.name,behavior_path=str(path),h5=h5.name,**qc)
        except Exception as exc:
            row['detail'] = str(exc)
        rows.append(row)
        with out.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        print(f"{session} run-{run:02d}: {row['status']} {row.get('behavior_tsv',row.get('detail',''))}",flush=True)
with out.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f'Saved {len(rows)} runs: {out}',flush=True)
