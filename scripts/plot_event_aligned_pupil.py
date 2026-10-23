"""Raw pupil diameter aligned to the saved tile-event windows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Shared nLightG2 plot typography.
import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

from plot_passive_event_aligned_tiles import event_onsets, aligned_trials


def plot_pupil(root,table_path):
    root=Path(root);table=pd.read_pickle(table_path)
    sources=sorted((root/'event_aligned').glob('plane[0-9]*_event_aligned_dff.npz'))
    if not sources:raise ValueError('Run tile event alignment first')
    plane=sources[0].name.split('_')[0]
    with np.load(root/plane/'tile_fluorescence_dff.npz') as z:times=z['time_s']
    nt=len(times);dt=float(np.median(np.diff(times)))
    if len(table)!=nt:
        manifests=list((root/'input').glob('*complete_volumes.json'))
        if len(manifests)!=1:raise ValueError('Cannot verify pupil alignment')
        m=json.loads(manifests[0].read_text())
        if len(table)!=nt+int(m['omitted_final_pages']>0) or m['output_pages']!=nt*len(sources):raise ValueError('Table row count mismatch')
    pupil=pd.to_numeric(table.pupil_diameter.iloc[:nt],errors='coerce').to_numpy(dtype=float,copy=True)
    pupil[(pupil<=0)|~np.isfinite(pupil)]=np.nan
    with np.load(sources[0]) as z:t=z['time_from_event_s']
    offsets=np.rint(t/dt).astype(int)
    if not np.allclose(t,offsets*dt):raise ValueError('Event timing does not match imaging')
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True,constrained_layout=True)
    arrays={'time_from_event_s':t};rows=[]
    for ax,(label,key,column,code) in zip(axes,[('Grating 1','grating_1','Stim',2),('Grating 2','grating_2','Stim',4),('Reward','reward','Reward',1)]):
        trials,mean,count,used=aligned_trials(pupil[None,:],event_onsets(table[column].iloc[:nt],code),offsets)
        trials=trials[:,0,:];mean=mean[0];count=count[0]
        variance=np.full(len(t),np.nan)
        np.divide(np.nansum((trials-mean)**2,axis=0),count-1,out=variance,where=count>1)
        sem=np.sqrt(variance/np.maximum(count,1))
        ax.plot(t,trials.T,color='teal',alpha=.12,lw=.65)
        ax.fill_between(t,mean-sem,mean+sem,color='teal',alpha=.25,label='±SEM across trials')
        ax.plot(t,mean,color='black',lw=2,label='Trial mean');ax.axvline(0,color='.45',ls='--',lw=1)
        ax.set(title=f'{label} · {len(used)} events',xlabel='Time from onset (s)',xlim=(t[0],t[-1]))
        ax.spines[['top','right']].set_visible(False)
        arrays.update({key+'_trials':trials,key+'_mean':mean,key+'_sem':sem,key+'_valid_count':count,key+'_onset_rows':used})
        for j,time in enumerate(t):rows.append(dict(event=key,time_s=time,mean_diameter=mean[j],sem_diameter=sem[j],n_valid_trials=count[j]))
    axes[0].set_ylabel('Pupil diameter (recorded units)');axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Event-aligned pupil diameter · faint lines: individual trials',fontsize=12)
    out=root/'event_aligned';path=out/'pupil_diameter_event_aligned.png'
    fig.savefig(path,dpi=180);plt.close(fig)
    np.savez_compressed(out/'pupil_diameter_event_aligned.npz',**arrays)
    pd.DataFrame(rows).to_csv(out/'pupil_diameter_event_aligned_summary.csv',index=False)
    (out/'pupil_diameter_event_aligned_settings.json').write_text(json.dumps(dict(table=str(table_path),window_s=[float(t[0]),float(t[-1])],alignment='User-specified table row = imaging volume; one pupil trace, not duplicated across planes',signal='Raw diameter in recorded units; nonpositive/nonfinite excluded; no smoothing or baseline subtraction',shading='Sample SD / sqrt(finite trial count) per time point; not animal SEM; trial dependence may affect interpretation'),indent=2))
    return path
