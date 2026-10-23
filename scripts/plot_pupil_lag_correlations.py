"""Descriptive lag curves for retained tiles, without circular wrapping."""
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

from scipy.signal import savgol_filter


def lag_correlations(traces, predictor, shifts, minimum=30):
    result=[];counts=[]
    for shift in shifts:
        # Positive lag means fluorescence follows pupil: corr(F[t+lag], P[t]).
        x=traces[:,shift:] if shift>0 else traces[:,:shift] if shift<0 else traces
        y=predictor[:-shift] if shift>0 else predictor[-shift:] if shift<0 else predictor
        good=np.isfinite(x)&np.isfinite(y)[None,:];n=good.sum(1)
        xx=np.where(good,x,0);yy=np.where(good,y[None,:],0)
        xc=np.where(good,xx-xx.sum(1)[:,None]/np.maximum(n,1)[:,None],0)
        yc=np.where(good,yy-yy.sum(1)[:,None]/np.maximum(n,1)[:,None],0)
        den=np.sqrt((xc*xc).sum(1)*(yc*yc).sum(1))
        r=np.full(len(x),np.nan);np.divide((xc*yc).sum(1),den,out=r,where=(n>=minimum)&(den>0))
        result.append(r);counts.append(n)
    return np.array(result).T,np.array(counts).T


def plot_lags(root,table_path,max_lag_s=3,smoothing_s=.5,minimum=30,positive_significant_only=False,fdr_alpha=.05):
    root=Path(root);table=pd.read_pickle(table_path)
    planes=sorted(p for p in root.glob('plane[0-9]*') if (p/'tile_fluorescence_dff.npz').exists())
    if not planes:raise ValueError('No saved tile traces')
    with np.load(planes[0]/'tile_fluorescence_dff.npz') as z:times=z['time_s'];nt=len(times)
    if len(table)!=nt:
        manifests=list((root/'input').glob('*complete_volumes.json'))
        if len(manifests)!=1:raise ValueError('Cannot verify table alignment')
        m=json.loads(manifests[0].read_text())
        if len(table)!=nt+int(m['omitted_final_pages']>0) or m['output_pages']!=nt*len(planes):raise ValueError('Table length mismatch')
    dt=float(np.median(np.diff(times)))
    if max_lag_s<=0 or max_lag_s>=times[-1]-times[0] or smoothing_s<0:raise ValueError('Invalid lag/smoothing window')
    pupil=pd.to_numeric(table.pupil_diameter.iloc[:nt],errors='coerce').to_numpy(dtype=float,copy=True)
    pupil[(pupil<=0)|~np.isfinite(pupil)]=np.nan
    smooth=np.full(nt,np.nan);derivative=np.full(nt,np.nan)
    window=max(3,int(round(smoothing_s/dt))|1) if smoothing_s else 0
    valid=np.isfinite(pupil);edges=np.diff(np.r_[False,valid,False].astype(int))
    for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        segment=pupil[start:stop]
        if window:
            if len(segment)<window:continue
            smooth[start:stop]=savgol_filter(segment,window,2)
            derivative[start:stop]=savgol_filter(segment,window,2,deriv=1,delta=dt)
            # Discard filter-edge estimates; never interpolate across missing pupil.
            half=window//2
            smooth[start:start+half]=np.nan;smooth[stop-half:stop]=np.nan
            derivative[start:start+half]=np.nan;derivative[stop-half:stop]=np.nan
        elif len(segment)>=3:
            smooth[start:stop]=segment;derivative[start+1:stop-1]=np.gradient(segment,dt)[1:-1]
    shifts=np.arange(-int(round(max_lag_s/dt)),int(round(max_lag_s/dt))+1);lags=shifts*dt
    selection=None
    if positive_significant_only:
        if not 0<fdr_alpha<1:raise ValueError('Invalid FDR threshold')
        selection=pd.read_csv(root/'pupil_correlation/tile_pupil_significance.csv')
        if selection.duplicated(['plane','tile_id']).any():raise ValueError('Duplicate significance rows')
    out=root/('pupil_lag_correlation_positive_significant' if positive_significant_only else 'pupil_lag_correlation');out.mkdir(exist_ok=True)
    selected_rows=[]
    fig,axes=plt.subplots(1,len(planes),figsize=(4.2*len(planes),4.5),sharey=True,squeeze=False,constrained_layout=True)
    rows=[];summary=[]
    for ax,plane in zip(axes.flat,planes):
        with np.load(plane/'tile_fluorescence_dff.npz') as z:
            traces=z['dff'].copy();ids=z['tile_ids'];bad=z['badframes'].astype(bool)
            if not np.allclose(z['time_s'],times):raise ValueError('Plane timing mismatch')
        with np.load(plane/'tile_selection.npz') as z:
            if not np.array_equal(ids,np.flatnonzero(z['keep_mask'])):raise ValueError('Tile selection mismatch')
        traces[:,bad]=np.nan
        if selection is not None:
            plane_selection=selection[selection.plane==plane.name].set_index('tile_id')
            if set(plane_selection.index)!=set(ids):raise ValueError('Significance results do not match retained tiles; rerun significance cell')
            plane_selection=plane_selection.loc[ids]
            current_r,_=lag_correlations(traces,pupil,[0],minimum)
            if not np.allclose(current_r[:,0],plane_selection.pearson_r,atol=1e-7,equal_nan=True):
                raise ValueError('Significance results are stale; rerun pupil correlation and significance cells')
            keep=(plane_selection.pearson_r.to_numpy()>0)&(plane_selection.q_bh.to_numpy()<=fdr_alpha)
            selected_rows.extend(plane_selection.loc[keep].reset_index().to_dict('records'))
            traces=traces[keep];ids=ids[keep]
        if not len(ids):
            ax.set(title=f'{plane.name} · 0 selected tiles',xlabel='Fluorescence lag (s)')
            ax.text(.5,.5,'No qualifying tiles',transform=ax.transAxes,ha='center')
            continue
        for name,predictor,color in [('Pupil diameter',smooth,'black'),('Pupil derivative',derivative,'#9290ad')]:
            r,n=lag_correlations(traces,predictor,shifts,minimum)
            mean=np.nanmean(r,axis=0);lo,hi=np.nanpercentile(r,[25,75],axis=0)
            ax.plot(lags,mean,color=color,lw=2,label=name);ax.fill_between(lags,lo,hi,color=color,alpha=.22)
            for j,lag in enumerate(lags):
                summary.append(dict(plane=plane.name,predictor=name,lag_s=lag,mean_r=mean[j],q25_r=lo[j],q75_r=hi[j]))
                for i,tile in enumerate(ids):rows.append(dict(plane=plane.name,tile_id=int(tile),predictor=name,lag_s=lag,pearson_r=r[i,j],n_pairs=n[i,j]))
        ax.axhline(0,color='.65',ls='--',lw=1);ax.axvline(0,color='.65',ls='--',lw=1)
        ax.set(title=f'{plane.name} · {len(ids)} tiles',xlabel='Fluorescence lag (s)');ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Correlation coefficient');axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle(('Positive pupil correlation, BH q ≤ '+str(fdr_alpha)+'; ' if positive_significant_only else '')+'mean across tiles\nShading = tile interquartile range; positive lag: fluorescence follows pupil',fontsize=12)
    path=out/'pupil_lag_correlations.png';fig.savefig(path,dpi=180);plt.close(fig)
    if selection is not None:pd.DataFrame(selected_rows).to_csv(out/'selected_tiles.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'tile_lag_correlations.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'lag_summary.csv',index=False)
    pd.DataFrame(dict(time_s=times,pupil_raw=pupil,pupil_smoothed=smooth,pupil_derivative=derivative)).to_csv(out/'pupil_predictors.csv',index=False)
    (out/'settings.json').write_text(json.dumps(dict(positive_significant_only=positive_significant_only,fdr_alpha=fdr_alpha,selection='Existing two-sided circular-shift test, pooled BH q <= alpha and raw zero-lag r > 0. Selection and visualization share data; not independent validation.' if positive_significant_only else 'All retained tiles',table=str(table_path),lag_definition='corr(F(t+lag), pupil(t)); positive fluorescence follows pupil',smoothing_s_requested=smoothing_s,savgol_window_frames=window,polynomial_order=2,dt_s=dt,max_lag_s=max_lag_s,minimum_pairs=minimum,shading='25th–75th percentile across retained tiles, not confidence interval',alignment='User-specified row=volume; no within-volume plane offset',limitations='One recording; descriptive correlations, no lag significance or causal claims. Source image alone does not establish its shading, lag sign, or preprocessing. Missing pupil gaps are not interpolated; no circular wrap.'),indent=2))
    return path
