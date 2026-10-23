"""Summarize an audit and label timestamp-only candidates without claiming rsync confirmation."""
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from align_run_dff_behavior import source_tiff,tiff_start,behavior_start
from calculate_run_dff import SUBJECT,OUTPUT,find_plane

source=OUTPUT/'behavior_matches_sessions216-231.csv'
df=pd.read_csv(source).fillna('')
df['timestamp_candidate']=''
df['note']='Verified by acquisition timestamps and rsync pulses.'
for i,row in df[df.status!='CONFIRMED'].iterrows():
    try:
        plane=find_plane(SUBJECT,int(row.session),int(row.run))
        tiff=source_tiff(plane)
        start=tiff_start(tiff,'Europe/London')
        ops=np.load(plane/'ops.npy',allow_pickle=True).item()
        n=np.load(plane/'F.npy',mmap_mode='r').shape[1]
        candidates=[]
        for tsv in sorted((tiff.parent.parent/'behav').glob('*.tsv')):
            if tsv.name.startswith('.'): continue
            raw=pd.read_csv(tsv,sep='\t');bs=behavior_start(raw,'UTC')
            duration=pd.to_numeric(raw.time,errors='coerce').max()
            if bs.timestamp()<=start.timestamp()+n/float(ops['fs']) and bs.timestamp()+duration>=start.timestamp():
                pulses=((raw.type=='event')&(raw.subtype=='sync')&(raw.content=='rsync')).sum()
                candidates.append((tsv.name,int(pulses)))
        df.loc[i,'timestamp_candidate']='; '.join(c[0] for c in candidates)
        if len(candidates)==1:
            df.loc[i,'status']='TIMESTAMP_ONLY'
            df.loc[i,'note']=f'Unique time overlap; {candidates[0][1]} behavior sync pulses. Not confirmed by the rsync audit.'
        elif not candidates:
            df.loc[i,'status']='NO_TIME_OVERLAP'
            df.loc[i,'note']='No behavior TSV interval overlaps the original TIFF acquisition.'
        else:
            df.loc[i,'note']='Multiple timestamp candidates; no unique confirmed alignment.'
    except Exception as exc:
        df.loc[i,'note']=str(exc)
reviewed=OUTPUT/'behavior_matches_sessions216-231_reviewed.csv'
df.to_csv(reviewed,index=False)
cols=['session','run','status','behavior_tsv','timestamp_candidate','imaging_tiff','h5','matched_pulses','note']
view=df[cols].copy()
html='''<!doctype html><meta charset="utf-8"><title>MY_71_R behavior matching</title>
<style>body{font:14px system-ui;margin:32px;color:#18212a}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#e9eff4}tr:hover{background:#f2f7fa}input{padding:10px;width:400px;margin:16px 0}</style>
<h1>MY_71_R — sessions 216–231</h1><p>One row per imaging run. CONFIRMED requires both absolute acquisition timing and shared rsync pulses. Timestamp-only candidates are not confirmed matches. TIFF epochs use Europe/London; TSV start_time uses UTC. Behavior filenames retain their original local-time names.</p>'''
html+=f'<p>{len(df)} runs. '+escape(str(df.status.value_counts().to_dict()))+'</p>'
html+='<input id="q" placeholder="Filter by session, run, filename or status">'
html+=view.to_html(index=False,border=0,table_id='matches')
html+='''<script>document.getElementById('q').addEventListener('input',e=>{for(const r of document.querySelectorAll('#matches tbody tr'))r.hidden=!r.innerText.toLowerCase().includes(e.target.value.toLowerCase())})</script>'''
(OUTPUT/'behavior_matches_sessions216-231.html').write_text(html)
md=['# MY_71_R behavior matches — sessions 216–231','', '| Session | Run | Status | Behavior TSV / candidate |', '|---|---|---|---|']
for r in df.itertuples():
    md.append(f'| {r.session} | {int(r.run):02d} | {r.status} | {r.behavior_tsv or r.timestamp_candidate or "No overlapping file"} |')
(OUTPUT/'behavior_matches_sessions216-231.md').write_text('\n'.join(md)+'\n')
print(df.status.value_counts().to_string())
print(df[df.status!='CONFIRMED'][['session','run','status','timestamp_candidate','note']].to_string(index=False))
print(reviewed)
