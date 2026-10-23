"""Per-tile nonwrapping 0–16 s pupil lag search with shift max-statistic control."""
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

from matplotlib.patches import Rectangle
from test_pupil_correlations import all_shift_correlations,bh_adjust


def scan(root,table_path,max_lag_s=16,alpha=.05):
    root=Path(root);table=pd.read_pickle(table_path)
    planes=sorted(p for p in root.glob('plane[0-9]*') if (p/'tile_fluorescence_dff.npz').exists())
    with np.load(planes[0]/'tile_fluorescence_dff.npz') as z:times=z['time_s'];nt=len(times)
    if len(table)!=nt:
        paths=list((root/'input').glob('*complete_volumes.json'))
        if len(paths)!=1:raise ValueError('Cannot verify table length')
        m=json.loads(paths[0].read_text())
        if len(table)!=nt+int(m['omitted_final_pages']>0) or m['output_pages']!=nt*len(planes):raise ValueError('Table length mismatch')
    dt=float(np.median(np.diff(times)))
    if not 0<max_lag_s<times[-1]/2 or not 0<alpha<1:raise ValueError('Invalid range/alpha')
    kmax=int(np.floor(max_lag_s/dt));lags=np.arange(kmax+1)*dt
    pupil=pd.to_numeric(table.pupil_diameter.iloc[:nt],errors='coerce').to_numpy(dtype=float,copy=True)
    pupil[(pupil<=0)|~np.isfinite(pupil)]=np.nan
    out=root/'pupil_lag_search';out.mkdir(exist_ok=True)
    rows=[];curves={};grids={}
    for plane in planes:
        with np.load(plane/'tile_fluorescence_dff.npz') as z:
            x=z['dff'].astype(float);ids=z['tile_ids'];bad=z['badframes'].astype(bool)
            if not np.allclose(times,z['time_s']):raise ValueError('Plane timing differs')
        with np.load(plane/'tile_selection.npz') as z:
            if not np.array_equal(ids,np.flatnonzero(z['keep_mask'])):raise ValueError('Tile selection mismatch')
        x[:,bad]=np.nan;x[:,:kmax]=np.nan
        plane_r=[]
        for start in range(0,len(x),64):
            r=all_shift_correlations(x[start:start+64],pupil)
            observed=r[:,:kmax+1];plane_r.append(observed)
            # Re-run the entire lag search at every circular pupil offset.
            nullmax=np.full_like(r,-np.inf)
            for k in range(kmax+1):nullmax=np.maximum(nullmax,np.nan_to_num(np.abs(np.roll(r,-k,axis=1)),nan=-np.inf))
            for j,a in enumerate(observed):
                tile=int(ids[start+j]);valid=np.isfinite(a)
                if not valid.any():
                    rows.append(dict(plane=plane.name,tile_id=tile,peak_r=np.nan,peak_lag_s=np.nan,max_r=np.nan,max_r_lag_s=np.nan,min_r=np.nan,min_r_lag_s=np.nan,p_lag_corrected=np.nan));continue
                peak=int(np.nanargmax(np.abs(a)));hi=int(np.nanargmax(a));lo=int(np.nanargmin(a))
                null=nullmax[j];ok=np.isfinite(null)
                p=float(np.sum(null[ok]>=abs(a[peak])-1e-12)/ok.sum())
                rows.append(dict(plane=plane.name,tile_id=tile,peak_r=a[peak],peak_lag_s=lags[peak],max_r=a[hi],max_r_lag_s=lags[hi],min_r=a[lo],min_r_lag_s=lags[lo],p_lag_corrected=p))
        curves[plane.name]=np.concatenate(plane_r);grids[plane.name]=json.loads((plane/'tile_fluorescence_dff.json').read_text())['grid_shape']
        np.savez_compressed(out/f'{plane.name}_lag_curves.npz',tile_ids=ids,lag_s=lags,pearson_r=curves[plane.name])
        print('Completed',plane.name,flush=True)
    df=pd.DataFrame(rows);df['q_bh']=bh_adjust(df.p_lag_corrected);df['significant']=df.q_bh<=alpha
    df.to_csv(out/'tile_peak_lag_statistics.csv',index=False)
    fig,axes=plt.subplots(len(planes),3,figsize=(13,3.5*len(planes)),constrained_layout=True,squeeze=False)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray');lagmap=plt.get_cmap('viridis').copy();lagmap.set_bad('lightgray')
    for row,plane in enumerate(planes):
        d=df[df.plane==plane.name];shape=grids[plane.name]
        for col,field in enumerate(['peak_r','peak_lag_s']):
            a=np.full(shape,np.nan);values=d[field].to_numpy().copy()
            if col==1:values[~d.significant.to_numpy()]=np.nan
            a.flat[d.tile_id.to_numpy()]=values
            im=axes[row,col].imshow(np.ma.masked_invalid(a),cmap=cmap if col==0 else lagmap,vmin=-1 if col==0 else 0,vmax=1 if col==0 else max_lag_s,interpolation='nearest')
            if col==0:
                for tile in d.loc[d.significant,'tile_id']:
                    y,x=divmod(int(tile),shape[1]);axes[row,col].add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,edgecolor='black',lw=.6))
            axes[row,col].set(title=f'{plane.name}: '+('Strongest |r| (signed); significant outlined' if col==0 else 'Peak lag: significant tiles only'),xlabel='Tile column',ylabel='Tile row')
            fig.colorbar(im,ax=axes[row,col],label='Pearson r' if col==0 else 'Lag (s)',shrink=.8)
        for sign,color,label in [(1,'firebrick','Positive peak'),(-1,'steelblue','Negative peak')]:
            values=d.loc[d.significant&(np.sign(d.peak_r)==sign),'peak_lag_s']
            axes[row,2].hist(values,bins=np.linspace(0,max_lag_s,17),histtype='step',lw=1.6,color=color,label=f'{label}: n={len(values)}')
        axes[row,2].set(xlabel='Peak fluorescence lag (s)',ylabel='Significant tiles',title=plane.name);axes[row,2].legend(fontsize=8)
    fig.suptitle(f'Pupil lag search 0–{max_lag_s:g} s · max-statistic shift test + pooled BH q ≤ {alpha}\nPositive lag means fluorescence follows pupil; peak = largest absolute correlation')
    fig.savefig(out/'pupil_peak_lag_maps.png',dpi=160);plt.close(fig)
    summary=df.groupby('plane').agg(tested=('tile_id','size'),significant=('significant','sum'))
    summary.to_csv(out/'summary.csv')
    (out/'settings.json').write_text(json.dumps(dict(max_lag_s=max_lag_s,actual_max_lag_s=float(lags[-1]),lag_step_s=dt,alpha=alpha,pupil='Raw diameter, no smoothing or derivative',support='Same imaging samples for all lags; first max-lag frames omitted. Observed lags do not wrap.',statistic='Largest absolute Pearson r across all tested lags; null repeats same maximization at every circular pupil offset; empirical p includes identity offset.',multiple_testing='BH across all retained tiles in all planes, after within-tile lag-search correction',alignment='User row=volume assumption; no plane-specific within-volume offsets',limitations='Exploratory shift-null stationarity assumption. Peak lags selected on same data, not causal delays or independently validated latencies. Boundary maxima may extend outside range. Stimulus-response latency is distinct from pupil lag. No stimulus/speed adjustment.'),indent=2))
    print(summary.to_string());return out/'pupil_peak_lag_maps.png'
