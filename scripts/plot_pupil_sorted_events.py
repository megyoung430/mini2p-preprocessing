"""Matched event heatmaps ordered by full-recording zero-lag pupil correlation."""
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

from plot_passive_event_aligned_tiles import subtract_trial_baseline, event_color_limit
from plot_pupil_lag_correlations import lag_correlations


def plot_sorted(root,table_path,baseline_start_s=-2,min_valid_fraction=.8, color_limit_percent=20, color_percentile=98):
    root=Path(root);out=root/'event_aligned';sources=sorted(out.glob('plane[0-9]*_event_aligned_dff.npz'))
    if not sources:raise FileNotFoundError('Run event alignment first')
    table=pd.read_pickle(table_path)
    with np.load(root/sources[0].name.split('_')[0]/'tile_fluorescence_dff.npz') as z:nt=len(z['time_s'])
    if len(table)!=nt:
        manifests=list((root/'input').glob('*complete_volumes.json'))
        if len(manifests)!=1:raise ValueError('Cannot verify row alignment')
        m=json.loads(manifests[0].read_text())
        if len(table)!=nt+int(m['omitted_final_pages']>0) or m['output_pages']!=nt*len(sources):raise ValueError('Table length mismatch')
    pupil=pd.to_numeric(table.pupil_diameter.iloc[:nt],errors='coerce').to_numpy(dtype=float,copy=True)
    pupil[(pupil<=0)|~np.isfinite(pupil)]=np.nan
    fig,axes=plt.subplots(len(sources),6,figsize=(28,3.6*len(sources)),constrained_layout=True,squeeze=False)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray');records=[]
    for row,source in enumerate(sources):
        plane=source.name.split('_')[0]
        with np.load(root/plane/'tile_fluorescence_dff.npz') as z:
            ids=z['tile_ids'];f=z['dff'].copy();f[:,z['badframes'].astype(bool)]=np.nan
        with np.load(root/plane/'tile_selection.npz') as z:
            if not np.array_equal(ids,np.flatnonzero(z['keep_mask'])):raise ValueError('Tile selection mismatch')
        r,n=lag_correlations(f,pupil,[0]);r=r[:,0];n=n[:,0]
        order=np.argsort(-np.where(np.isfinite(r),r,-np.inf),kind='stable')
        for rank,i in enumerate(order):records.append(dict(plane=plane,rank_top_to_bottom=rank+1,tile_id=int(ids[i]),pearson_r=r[i],n_pairs=int(n[i])))
        with np.load(source) as saved:
            if not np.array_equal(saved['tile_ids'],ids):raise ValueError('Event tiles are stale; rerun event-alignment cell')
            t=saved['time_from_event_s'];dt=float(np.median(np.diff(t)))
            for j,(label,key) in enumerate([('Grating 1','grating_1'),('Grating 2','grating_2'),('Reward','reward')]):
                original,centered,_,_,_=subtract_trial_baseline(saved[key+'_trials'],t,baseline_start_s,min_valid_fraction)
                for k,(name,values) in enumerate([('Original ΔF/F₀',original),('Prestimulus-subtracted',centered)]):
                    count=np.isfinite(values).sum(0);mean=np.full(count.shape,np.nan)
                    np.divide(np.nansum(values,axis=0),count,out=mean,where=count>0)
                    ax=axes[row,j*2+k]
                    im=ax.imshow(np.ma.masked_invalid(mean[order]*100),origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,vmin=-20,vmax=20,extent=[t[0]-dt/2,t[-1]+dt/2,len(ids)-.5,-.5])
                    ax.axvline(0,color='.3',ls='--',lw=.8)
                    ticks=np.unique(np.linspace(0,len(ids)-1,min(5,len(ids)),dtype=int))
                    ax.set_yticks(ticks,labels=[f'{ids[order[q]]} ({r[order[q]]:+.2f})' for q in ticks])
                    ax.set(title=f'{plane} · {label}\n{name}',xlabel='Time from onset (s)',ylabel='Tile ID (pupil r)',xlim=(t[0],t[-1]))
    images=[ax.images[0] for ax in axes.flat]
    limit=event_color_limit([image.get_array() for image in images], color_limit_percent, color_percentile)
    for image in images: image.set_clim(-limit, limit)
    fig.colorbar(im,ax=list(axes.flat),label='ΔF/F₀ (%) / prestimulus change (percentage points)',extend='both',shrink=.85)
    fig.suptitle('Tiles ordered by pupil correlation: most positive at top, most negative at bottom\nOriginal versus per-trial prestimulus subtraction; identical tile order across events',fontsize=16)
    path=out/'gratings_reward_pupil_sorted_comparison.png';fig.savefig(path,dpi=160);plt.close(fig)
    pd.DataFrame(records).to_csv(out/'pupil_sorted_tile_order.csv',index=False)
    (out/'pupil_sorted_settings.json').write_text(json.dumps(dict(table=str(table_path),sort='Full-recording raw pupil diameter vs saved dff, zero-lag Pearson r, descending within each plane; undefined correlations last; no significance filtering',baseline_start_s=baseline_start_s,min_valid_fraction=min_valid_fraction,limits=[-limit,limit],color_percentile=color_percentile,alignment='User-specified table row = imaging volume'),indent=2))
    return path
