"""Exploratory circular-shift tests for zero-lag tile/pupil correlations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def all_shift_correlations(x,y,minimum=30):
    """Pairwise-finite Pearson r for every circular shift of y; rows are tiles."""
    x=np.atleast_2d(np.asarray(x,float));y=np.asarray(y,float)
    mx=np.isfinite(x).astype(float);my=np.isfinite(y).astype(float)
    x=np.where(mx,x-np.nanmean(x,axis=1,keepdims=True),0)
    y=np.where(my,y-np.nanmean(y),0)
    n=x.shape[1]
    def cross(a,b):return np.fft.irfft(np.fft.rfft(a,axis=-1)*np.conj(np.fft.rfft(b,axis=-1)),n=n,axis=-1)
    count=np.rint(cross(mx,my))
    sx=cross(x,my);sy=cross(mx,y)
    xy=cross(x,y);xx=cross(x*x,my);yy=cross(mx,y*y)
    safe=np.maximum(count,1)
    numerator=xy-sx*sy/safe
    denom=np.sqrt(np.maximum(xx-sx*sx/safe,0)*np.maximum(yy-sy*sy/safe,0))
    r=np.full_like(numerator,np.nan)
    np.divide(numerator,denom,out=r,where=(count>=minimum)&(denom>1e-12))
    return np.clip(r,-1,1)

def bh_adjust(p):
    p=np.asarray(p,float);out=np.full_like(p,np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])];m=len(order)
    if m:out[order]=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1].clip(0,1)
    return out

def test_pupil_correlations(root,table_path,alpha=.05):
    root,table_path=Path(root),Path(table_path)
    table=pd.read_pickle(table_path)
    base=pd.read_csv(root/'pupil_correlation/tile_pupil_correlations.csv')
    rows=[];details=[]
    for plane in sorted(base.plane.unique()):
        folder=root/plane
        with np.load(folder/'tile_fluorescence_dff.npz') as z:
            x=z['dff'].astype(float);ids=z['tile_ids'];bad=z['badframes'].astype(bool);times=z['time_s']
        nt=x.shape[1]
        if len(table)!=nt:
            files=list((root/'input').glob('*complete_volumes.json'))
            if len(files)!=1:raise ValueError('Cannot validate table alignment')
            meta=json.loads(files[0].read_text())
            if len(table)!=nt+int(meta['omitted_final_pages']>0):raise ValueError('Table length mismatch')
        y=pd.to_numeric(table.pupil_diameter.iloc[:nt],errors='coerce').to_numpy(dtype=float,copy=True)
        y[(y<=0)|~np.isfinite(y)]=np.nan
        x[:,bad]=np.nan
        # Imaging bad-frame exclusions remain fixed; pupil gaps shift with pupil.
        # This equals the earlier pairwise exclusion for the observed zero-lag r.
        fs=1/np.median(np.diff(times));far=np.minimum(np.arange(nt),nt-np.arange(nt))/fs>=30
        for start in range(0,len(x),100):
            r=all_shift_correlations(x[start:start+100],y)
            for j in range(len(r)):
                obs=r[j,0];null=r[j,1:];valid=np.isfinite(null)
                p=(1+np.sum(np.abs(null[valid])>=abs(obs)-1e-12))/(1+valid.sum()) if np.isfinite(obs) else np.nan
                far_null=r[j,far];far_null=far_null[np.isfinite(far_null)]
                pfar=(1+np.sum(np.abs(far_null)>=abs(obs)-1e-12))/(1+len(far_null)) if np.isfinite(obs) else np.nan
                tile=int(ids[start+j]);expected=base[(base.plane==plane)&(base.tile_id==tile)].pearson_r.iloc[0]
                np.testing.assert_allclose(obs,expected,atol=1e-7,equal_nan=True)
                rows.append(dict(plane=plane,tile_id=tile,pearson_r=obs,p_shift=p,p_shift_excluding_30s=pfar,
                                 n_null_shifts=int(valid.sum())))
        details.append(dict(plane=plane,nframes=nt,fs_hz=float(fs)))
        print('Tested:',plane,flush=True)
    df=pd.DataFrame(rows)
    df['q_bh']=bh_adjust(df.p_shift)
    df['significant_fdr05']=df.q_bh<=alpha
    df['q_bh_excluding_30s']=bh_adjust(df.p_shift_excluding_30s)
    out=root/'pupil_correlation'
    df.to_csv(out/'tile_pupil_significance.csv',index=False)
    summary=df.groupby('plane').agg(tested=('tile_id','size'),significant=('significant_fdr05','sum'),
        min_q=('q_bh','min'),max_r=('pearson_r','max'),min_r=('pearson_r','min'))
    summary.to_csv(out/'significance_summary.csv')
    (out/'significance_settings.json').write_text(json.dumps(dict(alpha=alpha,planes=details,
        primary_test='Two-sided absolute Pearson r vs all nonzero integer circular shifts of pupil, shifting missing pupil mask with signal; fixed imaging masks; (1+exceedances)/(1+valid shifts)',
        multiple_testing='Benjamini-Hochberg across all finite tile tests pooled over all detected planes',
        sensitivity='Secondary shift test excludes offsets within 30 seconds of zero, in either direction; not the primary result',
        caveats='Exploratory within-recording evidence conditional on row alignment and shift-null stationarity. Autocorrelation retained but wraparound and drift can affect null validity. Shared stimulus/reward/movement effects are not controlled. Tiles are not independent animal replicates.',
        validation='All zero-lag correlations reproduce the earlier CSV; FFT correlations checked against direct masked Pearson for every shift on synthetic arrays'),indent=2))
    print(summary.to_string());print('TOTAL',int(df.significant_fdr05.sum()),'/',len(df))
    print('30s sensitivity total:',int((df.q_bh_excluding_30s<=alpha).sum()))
    print(df.sort_values('q_bh').head(12).to_string(index=False))
    return summary
