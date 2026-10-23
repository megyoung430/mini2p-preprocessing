"""Run ΔF/F or CSV-guided alignment for every recording in a session range.

python scripts/run_recording_exports.py dff --save-to-plane
python scripts/run_recording_exports.py align --write-full
Default sessions: 216–231. Alignment skips non-CONFIRMED/missing CSV rows.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from calculate_run_dff import SUBJECT, OUTPUT
from align_run_dff_behavior import MATCH_CSV, confirmed_csv_row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['dff','align'])
    p.add_argument('--first-session',type=int,default=216)
    p.add_argument('--last-session',type=int,default=231)
    p.add_argument('--subject-dir',type=Path,default=SUBJECT)
    p.add_argument('--output-dir',type=Path,default=None,
                   help='Override data export root; alignment defaults to each session/behav/run')
    p.add_argument('--matches-csv',type=Path,default=MATCH_CSV)
    p.add_argument('--save-to-plane',action='store_true')
    p.add_argument('--write-full',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='List eligible runs without calculation or export')
    a=p.parse_args()
    if not a.subject_dir.is_dir(): p.error('Subject directory unavailable; mount the drive first.')
    if a.first_session>a.last_session: p.error('Invalid session range')
    reports=[]
    for session in range(a.first_session,a.last_session+1):
        planes=sorted(a.subject_dir.glob(f'ses-{session}_*/funcimg/run-*/suite2p/plane0'))
        if not planes:
            reports.append(dict(session=session,status='FAILED',reason='No run folders found'))
        for plane in planes:
            run=int(plane.parents[1].name.split('-')[1])
            if a.stage=='align' and confirmed_csv_row(a.matches_csv,session,run) is None:
                print(f'SKIP {session} run-{run:02d}: no CONFIRMED CSV match',flush=True)
                reports.append(dict(session=session,run=run,status='SKIPPED',reason='No confirmed CSV match'))
                continue
            script='calculate_run_dff.py' if a.stage=='dff' else 'align_run_dff_behavior.py'
            command=[sys.executable,str(Path(__file__).with_name(script)),
                     '--session',str(session),'--run',str(run),'--subject-dir',str(a.subject_dir)]
            if a.output_dir:
                command+=['--output-dir',str(a.output_dir)]
            if a.stage=='dff' and a.save_to_plane: command+=['--save-to-plane']
            if a.stage=='align':
                command+=['--matches-csv',str(a.matches_csv)]
                if a.write_full: command+=['--write-full']
            print(f'RUN {session} run-{run:02d}: {a.stage}',flush=True)
            if a.dry_run:
                reports.append(dict(session=session,run=run,status='PLANNED'))
                continue
            result=subprocess.run(command,check=False)
            reports.append(dict(session=session,run=run,status='OK' if result.returncode==0 else 'FAILED',returncode=result.returncode))
    report_dir = a.output_dir or OUTPUT
    report_dir.mkdir(parents=True,exist_ok=True)
    suffix='_dry_run' if a.dry_run else ''
    (report_dir/f'batch_{a.stage}{suffix}_report.json').write_text(json.dumps(reports,indent=2))
    counts={s:sum(r['status']==s for r in reports) for s in ['OK','SKIPPED','FAILED','PLANNED']}
    print(counts)
    return 1 if counts['FAILED'] or not reports else 0

if __name__=='__main__':
    raise SystemExit(main())
