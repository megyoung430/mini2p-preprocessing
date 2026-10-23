"""Descriptive whole-image fluorescence tuning by simultaneous pupil diameter."""
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


def plot_tuning(root, table_path, n_bins=10, min_samples=30):
    root = Path(root)
    table = pd.read_pickle(table_path)
    
    planes = sorted(p for p in root.glob('plane[0-9]*') if (p/'whole_image_roi_dff.npz').exists())
    if not planes:
        raise ValueError('No whole-image ROI traces')

    if int(n_bins) != n_bins or n_bins<2 or min_samples<2:
        raise ValueError('Invalid bin/sample parameters')
    
    with np.load(planes[0]/'whole_image_roi_dff.npz') as z:
        times = z['time_s']
        nt = len(times)

    if len(table) != nt:
        manifests = list((root/'input').glob('*complete_volumes.json'))
        if len(manifests)!=1:
            raise ValueError('Cannot verify pupil alignment')
        m = json.loads(manifests[0].read_text())
        if len(table)!=nt+int(m['omitted_final_pages']>0) or m['output_pages']!=nt*len(planes):
            raise ValueError('Table length mismatch')

    pupil = pd.to_numeric(table.pupil_diameter.iloc[:nt], errors='coerce').to_numpy(dtype=float, copy=True)
    pupil[(pupil<=0)|~np.isfinite(pupil)] = np.nan
    finite = pupil[np.isfinite(pupil)]

    if len(finite)<min_samples or np.ptp(finite)<=0:
        raise ValueError('Insufficient variable pupil data')

    edges=np.linspace(finite.min(),finite.max(),int(n_bins)+1)
    bins=np.clip(np.searchsorted(edges,pupil,side='right')-1,0,n_bins-1)
    centers=(edges[:-1]+edges[1:])/2
    fig,axes=plt.subplots(2,len(planes),figsize=(4.2*len(planes),6.4),sharex=True,sharey='row',squeeze=False,constrained_layout=True,gridspec_kw={'height_ratios':[3,1]})
    rows=[];saved={'pupil_diameter':pupil,'time_s':times,'bin_edges':edges}
    for col,plane in enumerate(planes):
        with np.load(plane/'whole_image_roi_dff.npz') as z:
            if not np.allclose(z['time_s'],times):raise ValueError('Plane timing mismatch')
            y=z['dff'].astype(float)*100;bad=z['badframes'].astype(bool)
        valid=np.isfinite(y)&np.isfinite(pupil)&~bad
        means=np.full(n_bins,np.nan);sd=np.full(n_bins,np.nan);counts=np.zeros(n_bins,int)
        for b in range(n_bins):
            values=y[valid&(bins==b)];count=len(values);counts[b]=count
            mean=float(np.mean(values)) if count else np.nan
            spread=float(np.std(values,ddof=1)) if count>1 else np.nan
            if count>=min_samples:means[b]=mean;sd[b]=spread
            rows.append(dict(plane=plane.name,bin_index=b,pupil_left=edges[b],pupil_right=edges[b+1],pupil_center=centers[b],n_samples=count,mean_dff_percent=mean,sd_dff_percent=spread,sem_dff_percent=spread/np.sqrt(count) if count>1 else np.nan,plotted=count>=min_samples))
        sem=sd/np.sqrt(np.maximum(counts,1))
        ax=axes[0,col];ax.fill_between(centers,means-sem,means+sem,color='teal',alpha=.18,label='±SEM across paired frames')
        ax.plot(centers,means,'o-',color='teal',lw=1.8,ms=4,label='Mean ΔF/F₀');ax.axhline(0,color='.65',ls='--',lw=.7)
        ax.set(title=plane.name)
        axes[1,col].bar(centers,counts,width=np.diff(edges)*.9,color=np.where(counts>=min_samples,'teal','lightgray'))
        axes[1,col].axhline(min_samples,color='.4',ls=':',lw=.8)
        axes[1,col].set(xlabel='Pupil diameter (recorded units)')
        saved[plane.name+'_dff_percent']=np.where(valid,y,np.nan)
    axes[0,0].set_ylabel('Whole-image ΔF/F₀ (%)');axes[1,0].set_ylabel('Paired samples')
    axes[0,0].legend(fontsize=8,frameon=False)
    # fig.suptitle(f'Global fluorescence versus pupil size · {n_bins} equal-width bins · zero lag',fontsize=12)
    out=root/'pupil_tuning';out.mkdir(exist_ok=True);path=out/'whole_image_pupil_tuning.png'
    fig.savefig(path,dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'whole_image_pupil_tuning.csv',index=False)
    np.savez_compressed(out/'whole_image_pupil_tuning_samples.npz',**saved)
    (out/'settings.json').write_text(json.dumps(dict(table=str(table_path),n_bins=n_bins,min_samples=min_samples,bin_edges=edges.tolist(),signal='Saved whole-image ROI dff per plane, multiplied by 100',binning='Shared equal-width pupil bins over full positive finite range; last includes maximum; all valid paired frames, no lag or smoothing',alignment='User table row=imaging volume assumption; no within-volume plane offsets',shading='Sample SD / sqrt(number of valid paired frames) per pupil bin; temporal autocorrelation is not corrected and may understate uncertainty',limitations='Descriptive association. Consecutive frames are correlated; stimulus, reward, movement, time and baseline processing may affect the relationship. No causal tuning or NE concentration calibration.'),indent=2))
    return path

def plot_plane_average(root):
    """Average simultaneous planes first, then bin; frames are the SEM unit."""
    out=Path(root)/'pupil_tuning'
    settings=json.loads((out/'settings.json').read_text())
    with np.load(out/'whole_image_pupil_tuning_samples.npz') as z:
        pupil=z['pupil_diameter'];edges=z['bin_edges']
        keys=sorted(k for k in z.files if k.endswith('_dff_percent'))
        if not keys:raise ValueError('No saved plane traces; run tuning cell first')
        traces=np.stack([z[k] for k in keys])
    valid=np.isfinite(pupil)&np.all(np.isfinite(traces),axis=0)
    average=np.full(len(pupil),np.nan);average[valid]=traces[:,valid].mean(axis=0)
    n_bins=len(edges)-1;minimum=settings['min_samples']
    membership=np.clip(np.searchsorted(edges,pupil,side='right')-1,0,n_bins-1)
    centers=(edges[:-1]+edges[1:])/2
    rows=[];means=np.full(n_bins,np.nan);sems=means.copy();counts=np.zeros(n_bins,int)
    for b in range(n_bins):
        values=average[valid&(membership==b)];n=len(values);counts[b]=n
        mean=float(values.mean()) if n else np.nan
        sd=float(values.std(ddof=1)) if n>1 else np.nan
        sem=sd/np.sqrt(n) if n>1 else np.nan
        if n>=minimum:means[b]=mean;sems[b]=sem
        rows.append(dict(bin_index=b,pupil_left=edges[b],pupil_right=edges[b+1],pupil_center=centers[b],n_samples=n,mean_dff_percent=mean,sd_dff_percent=sd,sem_dff_percent=sem,plotted=n>=minimum))
    fig,axes=plt.subplots(2,1,figsize=(7,6),sharex=True,constrained_layout=True,gridspec_kw={'height_ratios':[3,1]})
    axes[0].fill_between(centers,means-sems,means+sems,color='teal',alpha=.25,label='±SEM across time samples')
    axes[0].plot(centers,means,'o-',color='teal',lw=2,label=f'Mean across {len(keys)} planes')
    axes[0].set(ylabel='Mean whole-image ΔF/F₀ (%)')
    # axes[0].legend(frameon=False,fontsize=9)
    axes[1].bar(centers,counts,width=np.diff(edges)*.9,color=np.where(counts>=minimum,'teal','lightgray'))
    axes[1].axhline(minimum,color='.4',ls=':',lw=.8)
    axes[1].set(xlabel='Pupil diameter (recorded units)',ylabel='Paired samples')
    # fig.suptitle('Pupil-size tuning: average across planes\nMean planes at each time point, then bin by pupil diameter')
    path=out/'all_planes_pupil_tuning.png';fig.savefig(path,dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'all_planes_pupil_tuning.csv',index=False)
    np.savez_compressed(out/'all_planes_pupil_tuning_samples.npz',pupil_diameter=pupil,mean_dff_percent=average,bin_edges=edges)
    (out/'all_planes_settings.json').write_text(json.dumps(dict(plane_keys=keys,method='Equal-weight mean of plane dff at each time point; require all planes finite, then bin by simultaneous pupil diameter using original shared edges',sem='SD of plane-averaged fluorescence samples within bin / sqrt(n time samples); not SEM across planes; temporal correlation may understate uncertainty',min_samples=minimum),indent=2))
    return path
