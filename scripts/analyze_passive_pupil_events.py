"""Trial metrics, pupil-state comparisons and temporal-CV Gaussian models for passive tiles."""
from pathlib import Path
import json
import hashlib
import warnings
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

EVENTS={'grating_1':2,'grating_2':4}
LABELS={'grating_1':'Grating 1','grating_2':'Grating 2','reward':'Reward'}

def mean_valid(a,axis):
    count=np.isfinite(a).sum(axis=axis)
    out=np.full(count.shape,np.nan,dtype=float)
    np.divide(np.nansum(a,axis=axis),count,out=out,where=count>0)
    return out

def event_basis(nt,onsets,fs,bins):
    cols=[];names=[];owners=[]
    for name,events in onsets.items():
        for a,b in zip(bins[:-1],bins[1:]):
            v=np.zeros(nt)
            for row in events:
                lo=int(row)+int(np.ceil(a*fs));hi=int(row)+int(np.ceil(b*fs))
                v[max(0,lo):min(nt,hi)]+=1
            cols.append(v);names.append(f'{name}_{a:g}_{b:g}s');owners.append(name)
    return np.column_stack(cols),names,owners

def make_design(basis,pupil,speed,train,include_speed=True):
    pmean=float(np.mean(pupil[train]));pstd=float(np.std(pupil[train]))
    if pstd<=0:raise ValueError('Pupil is constant in training interval')
    p=(pupil-pmean)/pstd
    parts=[np.ones((len(p),1)),basis,p[:,None],basis*p[:,None]]
    stats={'pupil_mean':pmean,'pupil_sd':pstd}
    if include_speed:
        smean=float(np.mean(speed[train]));ssd=float(np.std(speed[train]))
        parts.append(((speed-smean)/(ssd if ssd>0 else 1))[:,None])
        stats.update(speed_mean=smean,speed_sd=ssd)
    return np.column_stack(parts),stats

def score_r2(y,pred):
    sst=np.sum((y-y.mean(axis=0))**2,axis=0)
    out=np.full(sst.shape,np.nan)
    np.divide(np.sum((y-pred)**2,axis=0),sst,out=out,where=sst>0)
    return 1-out

def fit_models(Y,basis,pupil,speed,valid,times,names,owners,include_speed=True,nfolds=5,gap_s=60):
    """Shared observed rows; blocked held-out folds with a temporal buffer."""
    q=basis.shape[1];nt=len(times)
    columns=['intercept']+names+['pupil_z']+[f'{x}:pupil_z' for x in names]
    if include_speed:columns+=['speed_z']
    groups={}
    for name in ['grating_1','grating_2','reward']:
        ix=[j for j,owner in enumerate(owners) if owner==name]
        groups[name]=[1+j for j in ix]+[q+2+j for j in ix]
    groups['sensory']=groups['grating_1']+groups['grating_2']
    groups['pupil']=[q+1]+list(range(q+2,2*q+2))
    groups['interactions']=list(range(q+2,2*q+2))
    if include_speed:groups['speed']=[len(columns)-1]
    models={'full':np.arange(len(columns))}
    models.update({name:np.array([i for i in range(len(columns)) if i not in removed]) for name,removed in groups.items()})
    X,stats=make_design(basis,pupil,speed,valid,include_speed)
    yy=Y[valid];rank=int(np.linalg.matrix_rank(X[valid]));condition=float(np.linalg.cond(X[valid]))
    coef=np.linalg.lstsq(X[valid],yy,rcond=None)[0]
    fitted={};cv={k:np.full(Y.shape,np.nan) for k in models}
    fitted['full']=score_r2(yy,X[valid]@coef)
    for key,ix in models.items():
        if key!='full':
            b=np.linalg.lstsq(X[valid][:,ix],yy,rcond=None)[0]
            fitted[key]=score_r2(yy,X[valid][:,ix]@b)
    fold_details=[]
    for fold,test_indices in enumerate(np.array_split(np.arange(nt),nfolds)):
        lo,hi=times[test_indices[0]],times[test_indices[-1]]
        test=valid & (times>=lo)&(times<=hi)
        train=valid & ((times<lo-gap_s)|(times>hi+gap_s))
        if train.sum()<3*len(columns) or test.sum()<10:
            raise ValueError('Too few training/test samples; reduce folds/gap or model complexity')
        xx,_=make_design(basis,pupil,speed,train,include_speed)
        fold_details.append(dict(fold=fold,start_s=float(lo),stop_s=float(hi),train_n=int(train.sum()),test_n=int(test.sum()),rank=int(np.linalg.matrix_rank(xx[train]))))
        for key,ix in models.items():
            b=np.linalg.lstsq(xx[train][:,ix],Y[train],rcond=None)[0]
            cv[key][test]=xx[test][:,ix]@b
    cvscores={k:score_r2(Y[valid],v[valid]) for k,v in cv.items()}
    result={'r2_fit':fitted['full'],'r2_cv':cvscores['full']}
    for key in groups:
        result['delta_r2_fit_'+key]=fitted['full']-fitted[key]
        result['delta_r2_cv_'+key]=cvscores['full']-cvscores[key]
    return result,coef,columns,dict(rank=rank,n_columns=len(columns),condition_number=condition,
        n_valid=int(valid.sum()),normalization=stats,folds=fold_details,removed_columns={k:[columns[i] for i in ix] for k,ix in groups.items()})

def analyze(root,table_path,pre_s=10,post_s=20,min_valid_fraction=.8,pupil_split_quantile=.5,
            include_speed=True,bins=(0,2,5,10,20),cv_folds=5,cv_gap_s=60):
    root,table_path=Path(root),Path(table_path)
    if pre_s<=0 or post_s<=0 or not 0<min_valid_fraction<=1 or not 0<pupil_split_quantile<1:
        raise ValueError('Invalid epoch or baseline/pupil parameters')
    table=pd.read_pickle(table_path)
    planes=sorted(p for p in root.glob('plane[0-9]*') if (p/'tile_fluorescence_dff.npz').exists())
    if not planes:raise ValueError('No tile traces')
    with np.load(planes[0]/'tile_fluorescence_dff.npz') as z:times=z['time_s'];nt=len(times)
    if len(table)!=nt:
        paths=list((root/'input').glob('*complete_volumes.json'))
        if len(paths)!=1:raise ValueError('Cannot verify alignment')
        m=json.loads(paths[0].read_text())
        if len(table)!=nt+int(m['omitted_final_pages']>0) or m['output_pages']!=nt*len(planes):raise ValueError('Unverified row alignment')
    extra=len(table)-nt;table=table.iloc[:nt].copy()
    pupil=pd.to_numeric(table.pupil_diameter,errors='coerce').to_numpy(dtype=float,copy=True)
    pupil[(pupil<=0)|~np.isfinite(pupil)]=np.nan
    speed=pd.to_numeric(table.Speed_Absolute,errors='coerce').to_numpy(dtype=float) if include_speed else np.zeros(nt)
    fs=1/np.median(np.diff(times))
    offsets=np.arange(-int(round(pre_s*fs)),int(round(post_s*fs))+1);relative=offsets/fs
    premask=relative<0;postmask=(relative>=0)&(relative<post_s)
    npre=max(2,int(np.ceil(min_valid_fraction*premask.sum())))
    npost=max(2,int(np.ceil(min_valid_fraction*postmask.sum())))
    onsets={name:event_onsets(table.Stim,code) for name,code in EVENTS.items()};onsets['reward']=event_onsets(table.Reward,1)
    basis,names,owners=event_basis(nt,onsets,fs,bins)
    threshold=float(np.nanquantile(pupil,pupil_split_quantile))
    out=root/'pupil_event_analysis';out.mkdir(exist_ok=True)
    metrics=[];slopes=[];state_rows=[];glm_rows=[];coef_rows=[];event_info=[];qc={};fig_data=[];model_maps=[];non_event=[]
    for plane in planes:
        with np.load(plane/'tile_fluorescence_dff.npz') as z:
            F=z['dff'];ids=z['tile_ids'];bad=z['badframes'].astype(bool)
            if len(z['time_s'])!=nt or not np.allclose(z['time_s'],times):raise ValueError('Plane time axes differ')
        F=F.copy();F[:,bad]=np.nan
        grid=json.loads((plane/'tile_fluorescence_dff.json').read_text())['grid_shape']
        result_arrays={'relative_time_s':relative,'tile_ids':ids}
        for event,starts in onsets.items():
            trials,raw_mean,raw_count,used=aligned_trials(F,starts,offsets)
            if not len(used):raise ValueError(f'No complete windows: {event}')
            pre=trials[...,premask];bc=np.isfinite(pre).sum(-1)
            mu=mean_valid(pre,-1)
            var=np.full(mu.shape,np.nan)
            np.divide(np.nansum((pre-mu[...,None])**2,axis=-1),bc-1,out=var,where=bc>=npre)
            sd=np.sqrt(var);mu[bc<npre]=np.nan
            sd[(sd<=1e-6)|~np.isfinite(sd)]=np.nan
            zz=(trials-mu[...,None])/sd[...,None]
            zmean=mean_valid(zz,0)
            raw_matched=np.where(np.isfinite(sd)[...,None],trials,np.nan)
            raw_matched_mean=mean_valid(raw_matched,0)
            post=trials[...,postmask];zpost=zz[...,postmask]
            enough=np.isfinite(post).sum(-1)>=npost
            means=mean_valid(post,-1);means[~enough]=np.nan
            peak=np.max(np.where(np.isfinite(post),post,-np.inf),axis=-1);peak[~enough]=np.nan
            trough=np.min(np.where(np.isfinite(post),post,np.inf),axis=-1);trough[~enough]=np.nan
            post_count=np.isfinite(post).sum(-1)
            std=np.full(means.shape,np.nan)
            np.divide(np.nansum((post-means[...,None])**2,axis=-1),post_count-1,out=std,where=enough)
            std=np.sqrt(std)
            arg=np.argmax(np.where(np.isfinite(post),post,-np.inf),axis=-1)
            latency=relative[postmask][arg];latency[~enough]=np.nan
            zm=mean_valid(zpost,-1);zm[~enough]=np.nan
            zp=np.max(np.where(np.isfinite(zpost),zpost,-np.inf),axis=-1);zp[~np.isfinite(zp)|~enough]=np.nan
            zstd=std/sd;zrange=(peak-trough)/sd
            # Trial pupil state uses the same pre-event window; no state label from post-response data.
            pwin=pupil[used[:,None]+offsets[None,:]][:,premask]
            pcount=np.isfinite(pwin).sum(-1);pmean=mean_valid(pwin,-1);pmean[pcount<npre]=np.nan
            state=np.where(np.isfinite(pmean),np.where(pmean>=threshold,'large_pupil','small_pupil'),'unknown')
            other_before=[];other_after=[]
            for trial,row in enumerate(used):
                before=sum(int(np.sum((times[other]>=times[row]-pre_s)&(times[other]<times[row]))) for other in onsets.values())
                after=sum(int(np.sum((times[other]>times[row])&(times[other]<times[row]+post_s))) for other in onsets.values())
                other_before.append(before);other_after.append(after)
                event_info.append(dict(plane=plane.name,event=event,onset_row=int(row),onset_s=float(times[row]),
                    table_time_s=float(table.Time.iloc[row]),other_events_pre=before,other_events_post=after,
                    pupil_pre_mean=float(pmean[trial]),pupil_state=state[trial]))
                for i,tile in enumerate(ids):
                    metrics.append(dict(plane=plane.name,event=event,onset_row=int(row),tile_id=int(tile),
                        pupil_state=state[trial],pupil_pre_mean=float(pmean[trial]),baseline_mean=float(mu[trial,i]),
                        baseline_sd=float(sd[trial,i]),baseline_valid_n=int(bc[trial,i]),post_valid_n=int(post_count[trial,i]),
                        mean_dff=float(means[trial,i]),peak_dff=float(peak[trial,i]),std_dff=float(std[trial,i]),
                        range_dff=float(peak[trial,i]-trough[trial,i]),latency_to_peak_s=float(latency[trial,i]),
                        mean_z=float(zm[trial,i]),peak_z=float(zp[trial,i]),std_z=float(zstd[trial,i]),range_z=float(zrange[trial,i])))
            result_arrays.update({event+'_onset_rows':used,event+'_mean_dff':raw_matched_mean,event+'_mean_z':zmean,
                event+'_z_valid_trial_count':np.isfinite(zz).sum(0),event+'_baseline_mean':mu,event+'_baseline_sd':sd,
                event+'_pupil_state':state})
            fig_data.append((plane.name,event,ids,raw_matched_mean*100,zmean,len(used)))
            state_means=[]
            for value in ['small_pupil','large_pupil']:
                selected=means[state==value];scount=np.isfinite(selected).sum(0)
                sm=mean_valid(selected,0);sm[scount<2]=np.nan;state_means.append(sm)
                for i,tile in enumerate(ids):state_rows.append(dict(plane=plane.name,event=event,tile_id=int(tile),
                    pupil_state=value,n_valid_trials=int(scount[i]),mean_dff=float(sm[i])))
            a,b=state_means;ok=np.isfinite(a)&np.isfinite(b)
            slope=intercept=rr=np.nan
            if ok.sum()>=3 and np.std(a[ok])>0:
                slope,intercept=np.linalg.lstsq(np.column_stack([a[ok],np.ones(ok.sum())]),b[ok],rcond=None)[0]
                rr=np.corrcoef(a[ok],b[ok])[0,1]
            slopes.append(dict(plane=plane.name,event=event,slope_large_vs_small=float(slope),intercept=float(intercept),
                r=float(rr),n_tiles=int(ok.sum()),small_trials=int((state=='small_pupil').sum()),large_trials=int((state=='large_pupil').sum())))
        np.savez_compressed(out/f'{plane.name}_event_means.npz',**result_arrays)
        # Event-free describes timing only, not confirmed spontaneous biological events.
        free=~(basis>0).any(1)&~bad
        for i,tile in enumerate(ids):
            a=F[i,free];a=a[np.isfinite(a)]
            non_event.append(dict(plane=plane.name,tile_id=int(tile),event_free_seconds=len(a)/fs,
                mean_dff=float(np.mean(a)) if len(a) else np.nan,std_dff=float(np.std(a,ddof=1)) if len(a)>1 else np.nan))
        valid=np.isfinite(pupil)&~bad&np.all(np.isfinite(F),axis=0)
        if include_speed:valid &= np.isfinite(speed)
        if valid.sum()<100:raise ValueError('Too few common model samples')
        models,coef,columns,diagnostic=fit_models(F.T,basis,pupil,speed,valid,times,names,owners,include_speed,cv_folds,cv_gap_s)
        diagnostic['nonfinite_or_bad_rows_excluded']=int(nt-valid.sum());qc[plane.name]=diagnostic
        for i,tile in enumerate(ids):
            glm_rows.append(dict(plane=plane.name,tile_id=int(tile),**{k:float(v[i]) for k,v in models.items()}))
            for j,name in enumerate(columns):coef_rows.append(dict(plane=plane.name,tile_id=int(tile),term=name,beta=float(coef[j,i])))
        model_maps.append((plane.name,ids,grid,models))
        print('Completed:',plane.name,'model rank',diagnostic['rank'],'/',diagnostic['n_columns'],flush=True)
    trialdf=pd.DataFrame(metrics);trialdf.to_csv(out/'event_trial_tile_metrics.csv',index=False)
    metrics_cols=['mean_dff','peak_dff','std_dff','range_dff','latency_to_peak_s','mean_z','peak_z','std_z','range_z']
    tile_summary=trialdf.groupby(['plane','event','tile_id'])[metrics_cols].mean().reset_index()
    tile_summary.to_csv(out/'event_tile_summary.csv',index=False)
    tile_summary.groupby(['plane','event'])[metrics_cols].mean().to_csv(out/'event_plane_summary.csv')
    pd.DataFrame(state_rows).to_csv(out/'pupil_state_tile_means.csv',index=False)
    pd.DataFrame(slopes).to_csv(out/'pupil_state_slopes.csv',index=False)
    pd.DataFrame(event_info).to_csv(out/'trial_alignment_and_overlap.csv',index=False)
    pd.DataFrame(non_event).to_csv(out/'event_free_summary.csv',index=False)
    modeldf=pd.DataFrame(glm_rows);modeldf.to_csv(out/'glm_tile_metrics.csv',index=False)
    modeldf.groupby('plane').mean(numeric_only=True).drop(columns='tile_id').to_csv(out/'glm_plane_summary.csv')
    pd.DataFrame(coef_rows).to_csv(out/'glm_coefficients.csv',index=False)
    qc['analysis']=dict(table=str(table_path),table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        table_final_rows_omitted=extra,alignment='row i = volume/frame i per user; no within-volume plane offset',
        fs_hz=float(fs),pre_s=pre_s,post_s=post_s,baseline_min_valid=npre,post_min_valid=npost,z_sd_floor=1e-6,
        event_codes=EVENTS,event_window_bins_s=bins,include_speed_control=include_speed,
        pupil_split_quantile=pupil_split_quantile,pupil_threshold=threshold,
        state_definition='Per-trial mean pupil in prestim window; split at full-session pupil quantile; >=2 usable trials/state/tile required for slopes',
        model='Gaussian identity-link linear model: intercept + event lag-bin counts + standardized pupil + each event-bin*pupil + optional standardized speed',
        cv='Contiguous blocked folds; exclude neighboring train samples within gap; standardization fit on train rows; pooled out-of-fold R2 on same observations for full/reduced models',
        cv_folds=cv_folds,cv_gap_s=cv_gap_s,
        group_drop='Drop a predictor together with its interactions; interaction-only group drops interactions alone. Group contributions overlap and do not sum to total R2.',
        inference='Descriptive one-recording results; no mouse-level KS/t/Wilcoxon/Friedman tests or animal effect sizes. No spontaneous-event detector inferred from unspecified criteria.',
        units='dff and changes are fractional; z is unitless; plots convert raw dff to percent',
        caveats='Closely spaced events overlap pre/post windows. Baseline estimation uses saved dff; CV is on these precomputed traces, not a full preprocessing holdout. Negative CV R2/delta R2 retained. No causal interpretation.')
    (out/'analysis_settings.json').write_text(json.dumps(qc,indent=2))
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray')
    fig,axes=plt.subplots(len(planes),6,figsize=(28,3.5*len(planes)),constrained_layout=True,squeeze=False)
    for k,(plane,event,ids,raw,zmean,n) in enumerate(fig_data):
        row,col=divmod(k,3)
        for j,(values,limit,label) in enumerate([(raw,20,'ΔF/F₀ (%)'),(zmean,3,'Prestimulus z')]):
            ax=axes[row,col*2+j]
            im=ax.imshow(np.ma.masked_invalid(values),origin='upper',aspect='auto',interpolation='nearest',
                cmap=cmap,vmin=-limit,vmax=limit,extent=[relative[0]-.5/fs,relative[-1]+.5/fs,len(ids)-.5,-.5])
            ax.axvline(0,color='.3',ls='--',lw=.7);ax.set(xlabel='Seconds from onset',ylabel='Retained tile ID',title=f'{plane} · {LABELS[event]} · n={n}\n{label}')
            loc=np.unique(np.linspace(0,len(ids)-1,5,dtype=int));ax.set_yticks(loc,labels=ids[loc])
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=plt.Normalize(-20,20),cmap=cmap),ax=list(axes[:,::2].flat),label='Trial mean ΔF/F₀ (%)',shrink=.8,extend='both')
    fig.colorbar(ScalarMappable(norm=plt.Normalize(-3,3),cmap=cmap),ax=list(axes[:,1::2].flat),label='Trial mean z-score',shrink=.8,extend='both')
    fig.suptitle('10-second prestimulus z-score; 20-second responses — overlapping events retained',fontsize=16)
    fig.savefig(out/'event_raw_and_z_heatmaps.png',dpi=150);plt.close(fig)
    fields=['r2_fit','r2_cv','delta_r2_cv_sensory','delta_r2_cv_reward','delta_r2_cv_pupil','delta_r2_cv_interactions']
    if include_speed:fields+=['delta_r2_cv_speed']
    fig,axes=plt.subplots(len(planes),len(fields),figsize=(4*len(fields),3.4*len(planes)),constrained_layout=True,squeeze=False)
    delta_vals=np.concatenate([np.abs(modeldf[f].dropna()) for f in fields if f.startswith('delta')]);dl=max(.01,float(np.percentile(delta_vals,99)))
    for row,(plane,ids,grid,models) in enumerate(model_maps):
        for col,field in enumerate(fields):
            a=np.full(grid,np.nan);a.flat[ids]=models[field]
            lim=1 if field.startswith('r2') else dl
            im=axes[row,col].imshow(np.ma.masked_invalid(a),cmap=cmap,vmin=-lim,vmax=lim,interpolation='nearest')
            axes[row,col].set(title=f'{plane}\n{field}',xlabel='Tile column',ylabel='Tile row')
    fig.colorbar(ScalarMappable(norm=plt.Normalize(-1,1),cmap=cmap),ax=list(axes[:,:2].flat),label='R²',shrink=.8,extend='both')
    fig.colorbar(ScalarMappable(norm=plt.Normalize(-dl,dl),cmap=cmap),ax=list(axes[:,2:].flat),label='Held-out ΔR²',shrink=.8,extend='both')
    fig.suptitle('Gaussian event–pupil model: fit and held-out contributions; gray = excluded tiles',fontsize=16)
    fig.savefig(out/'glm_spatial_maps.png',dpi=150);plt.close(fig)
    print(modeldf.groupby('plane')[fields].median().to_string())
    plot_pupil_event_metrics(out,pupil,onsets,relative,offsets,npre,npost,post_s)
    print('Output:',out)
    return out


def plot_pupil_event_metrics(out,pupil,onsets,relative,offsets,npre,npost,post_s):
    """One pupil response per event/trial, never duplicated as independent plane data."""
    out=Path(out);pre=relative<0;post=(relative>=0)&(relative<post_s)
    rows=[];arrays={'relative_time_s':relative}
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True,squeeze=False)
    for col,(event,starts) in enumerate(onsets.items()):
        indices=starts[:,None]+offsets[None,:]
        good=(indices[:,0]>=0)&(indices[:,-1]<len(pupil));used=starts[good]
        trials=pupil[indices[good]]
        count=np.isfinite(trials[:,pre]).sum(1);mu=mean_valid(trials[:,pre],1)
        var=np.full(mu.shape,np.nan)
        np.divide(np.nansum((trials[:,pre]-mu[:,None])**2,axis=1),count-1,out=var,where=count>=npre)
        mu[count<npre]=np.nan;sd=np.sqrt(var);sd[(sd<=1e-9)|~np.isfinite(sd)]=np.nan
        z=(trials-mu[:,None])/sd[:,None]
        change=(trials-mu[:,None])/mu[:,None]*100
        for i,row in enumerate(used):
            enough=np.isfinite(trials[i,post]).sum()>=npost
            a=trials[i,post];za=z[i,post];ch=change[i,post]
            if enough:
                peak=float(np.nanmax(a));trough=float(np.nanmin(a));lat=float(relative[post][np.nanargmax(a)])
            else:peak=trough=lat=np.nan
            rows.append(dict(event=event,onset_row=int(row),baseline_diameter=float(mu[i]),baseline_sd=float(sd[i]),
                mean_diameter=float(np.nanmean(a)) if enough else np.nan,peak_diameter=peak,
                std_diameter=float(np.nanstd(a,ddof=1)) if enough else np.nan,range_diameter=peak-trough,
                mean_z=float(np.nanmean(za)) if enough and np.isfinite(sd[i]) else np.nan,
                peak_z=float(np.nanmax(za)) if enough and np.isfinite(sd[i]) else np.nan,
                mean_percent_change=float(np.nanmean(ch)) if enough and np.isfinite(mu[i]) else np.nan,
                latency_to_peak_s=lat))
        arrays.update({event+'_trials':trials,event+'_z_trials':z,event+'_onset_rows':used})
        for r,(a,label) in enumerate([(trials,'Pupil diameter (recorded units)'),(z,'Prestimulus pupil z-score')]):
            mean=mean_valid(a,0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);lo,hi=np.nanpercentile(a,[10,90],axis=0)
            ax=axes[r,col];ax.fill_between(relative,lo,hi,alpha=.2,color='teal',label='10th–90th percentile across trials')
            ax.plot(relative,mean,color='teal',lw=1.2,label='Trial mean');ax.axvline(0,color='.3',ls='--',lw=.8)
            ax.set(title=f'{LABELS[event]} · {len(used)} events',xlabel='Seconds from onset',ylabel=label)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Pupil responses aligned to events — 10-second prestimulus reference')
    fig.savefig(out/'pupil_event_responses.png',dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'pupil_event_trial_metrics.csv',index=False)
    np.savez_compressed(out/'pupil_event_trials.npz',**arrays)
