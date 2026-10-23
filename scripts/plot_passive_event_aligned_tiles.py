"""Event-aligned tile dff from an explicitly row-aligned behavioral table."""
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


def event_color_limit(arrays, limit_percent=None, percentile=98):
    """Shared symmetric display limit in percent dF/F; never changes data."""
    if limit_percent is not None:
        limit = float(limit_percent)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError('Color limit must be positive and finite')
        return limit
    if not 0 < percentile <= 100:
        raise ValueError('Color percentile must be in (0, 100]')
    values = np.concatenate([np.ma.asarray(a).compressed().ravel() for a in arrays])
    values = np.abs(values[np.isfinite(values)])
    if not values.size:
        raise ValueError('No finite event responses to scale')
    return max(float(np.percentile(values, percentile)), 1e-6)


def event_onsets(values, code):
    active = np.asarray(values) == code
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])

def aligned_trials(dff, onsets, offsets):
    indices = np.asarray(onsets)[:, None] + np.asarray(offsets)[None, :]
    valid = (indices[:, 0] >= 0) & (indices[:, -1] < dff.shape[1])
    used = np.asarray(onsets)[valid]
    trials = dff[:, indices[valid]].transpose(1, 0, 2)
    count = np.isfinite(trials).sum(axis=0)
    mean = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(np.nansum(trials, axis=0), count, out=mean, where=count > 0)
    return trials, mean, count, used

def plot_event_aligned_tiles(root, table_path, grating_codes, pre_s=2.0, post_s=5.0, onset_position=None, grating_groups=('gr_2', 'gr_4'), color_limit_percent=20, color_percentile=98):
    root, table_path = Path(root), Path(table_path)
    if set(grating_codes) != {'Grating 1', 'Grating 2'} or len(set(grating_codes.values())) != 2:
        raise ValueError('Provide distinct Stim codes for Grating 1 and Grating 2')
    if pre_s < 0 or post_s <= 0:
        raise ValueError('Require pre_s >= 0 and post_s > 0')
    table = pd.read_pickle(table_path)
    index_source = isinstance(table, dict)
    if not index_source:
        for col in ['Time', 'Stim', 'Reward']:
            if col not in table:raise ValueError(f'Missing {col} column')
    plane_dirs = sorted(p for p in root.glob('plane[0-9]*') if p.is_dir())
    plane_data=[]
    for p in plane_dirs:
        with np.load(p/'tile_fluorescence_dff.npz') as z:
            plane_data.append((p, z['dff'], z['time_s'], z['tile_ids']))
    if not plane_data:raise ValueError('No saved tile traces')
    nt=len(plane_data[0][2]); times=plane_data[0][2]
    if nt < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Saved imaging time axis must be finite and strictly increasing')
    for p, dff, t, ids in plane_data:
        if dff.shape != (len(ids),nt) or not np.allclose(t,times):
            raise ValueError(f'Inconsistent frame axes in {p}')
    if index_source:
        from load_nlight_event_indices import load_event_onsets
        loaded = load_event_onsets(table_path, nt, onset_position, grating_groups)
        events = {'Grating 1': loaded['grating_1'], 'Grating 2': loaded['grating_2'], 'Reward': loaded['reward']}
        dropped_rows = 0
    else:
        if len(table) != nt:
            summaries=list((root/'input').glob('*complete_volumes.json'))
            if len(summaries)!=1:raise ValueError('Cannot verify extra table row without page-count metadata')
            summary=json.loads(summaries[0].read_text())
            expected_extra=int(summary['omitted_final_pages']>0)
            if len(table)!=nt+expected_extra or summary['output_pages']!=nt*len(plane_data):
                raise ValueError('Table length does not match complete volumes plus final partial volume')
        dropped_rows=len(table)-nt
        table=table.iloc[:nt].copy()
        if not np.isfinite(table.Time).all() or not (np.diff(table.Time)>0).all():
            warnings.warn(
                'Behavior Time contains nonfinite or nonincreasing timestamps. '
                'Using the explicitly row-aligned event positions and saved imaging '
                'time axis; behavioral rows are not sorted or dropped, and original '
                'Time values are retained in event_onsets.csv.',
                RuntimeWarning, stacklevel=2)
    fs=1/np.median(np.diff(times))
    offsets=np.arange(-int(round(pre_s*fs)),int(round(post_s*fs))+1)
    relative_s=offsets/fs
    if not index_source:
        events={name:event_onsets(table.Stim,code) for name,code in grating_codes.items()}
        events['Reward']=event_onsets(table.Reward,1)
    if any(len(indices)==0 for indices in events.values()):raise ValueError('One condition has no events')
    out=root/'event_aligned';out.mkdir(exist_ok=True)
    (out/'event_source.json').write_text(json.dumps(dict(source=str(table_path),grating_groups=grating_groups if index_source else grating_codes,onset_position=onset_position,index_source=index_source),indent=2))
    manifest=[];panels=[];event_counts={}
    for plane_dir,dff,times,ids in plane_data:
        arrays={'time_from_event_s':relative_s,'tile_ids':ids}
        for name,onsets in events.items():
            trials, mean, count, used=aligned_trials(dff,onsets,offsets)
            if not len(used):raise ValueError(f'No complete windows for {name}')
            key=name.lower().replace(' ','_')
            arrays.update({key+'_trials':trials,key+'_mean':mean,key+'_valid_trial_count':count,
                           key+'_onset_rows':used})
            panels.append((plane_dir.name,name,mean*100,ids,len(used),len(onsets)))
            event_counts[name]={'detected':len(onsets),'complete_windows':len(used)}
        np.savez_compressed(out/f'{plane_dir.name}_event_aligned_dff.npz',**arrays)
    for name,onsets in events.items():
        for row in onsets:
            manifest.append(dict(event=name,table_row=int(row),suite2p_frame=int(row),
                table_time_s=float(table.Time.iloc[row]) if not index_source else None,imaging_time_s=float(times[row]),
                full_window=bool(row+offsets[0]>=0 and row+offsets[-1]<nt)))
    pd.DataFrame(manifest).to_csv(out/'event_onsets.csv',index=False)
    limit=event_color_limit([m for _,_,m,_,_,_ in panels], color_limit_percent, color_percentile)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray')
    fig,axes=plt.subplots(len(plane_data),3,figsize=(18,3.5*len(plane_data)),
                          constrained_layout=True,squeeze=False)
    for ax,(plane,name,mean,ids,nused,ntotal) in zip(axes.flat,panels):
        im=ax.imshow(np.ma.masked_invalid(mean),aspect='auto',origin='upper',interpolation='nearest',
            cmap=cmap,vmin=-limit,vmax=limit,
            extent=[relative_s[0]-.5/fs,relative_s[-1]+.5/fs,len(ids)-.5,-.5])
        ax.axvline(0,color='white',ls='--',lw=1)
        ticks=np.unique(np.linspace(0,len(ids)-1,min(6,len(ids)),dtype=int))
        ax.set_yticks(ticks,labels=ids[ticks])
        ax.set(title=f'{plane} · {name} · {nused}/{ntotal} events',xlabel='Time from onset (s)',
               ylabel='Retained tile ID',xlim=(relative_s[0],relative_s[-1]))
    fig.colorbar(im,ax=list(axes.flat),label='Trial-mean ΔF/F₀ (%)',extend='both',shrink=.85)
    fig.suptitle('Passive recording — each retained tile averaged across repeated event onsets',fontsize=16)
    figure=out/'gratings_reward_tile_dff.png';fig.savefig(figure,dpi=180);plt.close(fig)
    settings=dict(source_table=str(table_path),source_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        alignment='User-specified row alignment: table row i = Suite2p volume/frame i. FrameIndex is not treated as imaging frame number.',
        dropped_final_table_rows=dropped_rows,fs_hz=float(fs),pre_s=pre_s,post_s=post_s,
        effective_relative_range_s=[float(relative_s[0]),float(relative_s[-1])],grating_codes=grating_codes,
        events=event_counts,event_detection='rising edge into each code; Reward rising edge into 1',
        aggregation='mean across finite trial values per tile/timepoint; no prestimulus subtraction, z scoring, or smoothing',
        trials_axes=['events','retained_tiles','relative_time'],mean_axes=['retained_tiles','relative_time'],
        saved_dff_units='fraction',plotted_units='percent',color_limits_percent=[-limit,limit],color_percentile=color_percentile,colormap='RdBu_r',
        limitations='Frame-level alignment, no within-volume plane timing adjustment. Windows may include effects of other events; event association is not causal isolation.')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    print(json.dumps(event_counts,indent=2));print('Saved:',figure)
    return figure


def subtract_trial_baseline(trials, relative_s, baseline_start_s=-2.0, min_valid_fraction=0.8):
    """Center each trial/tile on its finite prestimulus mean, never including t=0."""
    baseline_mask=(relative_s>=baseline_start_s)&(relative_s<0)
    if not baseline_mask.any() or not 0<min_valid_fraction<=1:
        raise ValueError('Need prestimulus samples and a valid minimum-sample fraction')
    values=trials[...,baseline_mask]
    count=np.isfinite(values).sum(axis=-1)
    minimum=int(np.ceil(min_valid_fraction*baseline_mask.sum()))
    baseline=np.full(count.shape,np.nan,dtype=np.float32)
    np.divide(np.nansum(values,axis=-1),count,out=baseline,where=count>=minimum)
    eligible=np.isfinite(baseline)
    original=np.where(eligible[...,None],trials,np.nan)
    centered=original-baseline[...,None]
    return original,centered,baseline,count,minimum


def plot_prestimulus_comparison(root, baseline_start_s=-2.0, min_valid_fraction=0.8, pupil_table_path=None, color_limit_percent=20, color_percentile=98):
    """Compare original and trial-baseline-subtracted means using matched eligible trials."""
    out=Path(root)/'event_aligned'
    sources=sorted(out.glob('plane[0-9]*_event_aligned_dff.npz'))
    if not sources:raise FileNotFoundError('Run event alignment first')
    events=[('Grating 1','grating_1'),('Grating 2','grating_2'),('Reward','reward')]
    fig,axes=plt.subplots(len(sources),6,figsize=(28,3.6*len(sources)),constrained_layout=True,squeeze=False)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray')
    details={}
    for row,source in enumerate(sources):
        plane=source.name.split('_')[0]
        arrays={};details[plane]={}
        with np.load(source) as saved:
            t=saved['time_from_event_s'];ids=saved['tile_ids'];dt=np.median(np.diff(t))
            if t[0] > baseline_start_s+dt:
                raise ValueError('Saved event window is shorter than the requested baseline')
            order=np.arange(len(ids))
            if pupil_table_path is not None:
                from plot_pupil_lag_correlations import lag_correlations
                table=pd.read_pickle(pupil_table_path)
                with np.load(Path(root)/plane/'tile_fluorescence_dff.npz') as current:
                    if not np.array_equal(ids,current['tile_ids']):raise ValueError('Event tile IDs are stale; rerun event alignment')
                    traces=current['dff'].copy();traces[:,current['badframes'].astype(bool)]=np.nan
                    nt=traces.shape[1]
                if len(table)!=nt:
                    manifests=list((Path(root)/'input').glob('*complete_volumes.json'))
                    if len(manifests)!=1:raise ValueError('Cannot verify pupil alignment')
                    manifest=json.loads(manifests[0].read_text())
                    if len(table)!=nt+int(manifest['omitted_final_pages']>0) or manifest['output_pages']!=nt*len(sources):raise ValueError('Pupil row count mismatch')
                pupil=pd.to_numeric(table.pupil_diameter.iloc[:nt],errors='coerce').to_numpy(dtype=float,copy=True)
                pupil[(pupil<=0)|~np.isfinite(pupil)]=np.nan
                correlations,_=lag_correlations(traces,pupil,[0])
                correlations=correlations[:,0]
                order=np.argsort(-np.nan_to_num(correlations,nan=-np.inf),kind='stable')
                pd.DataFrame(dict(rank=np.arange(1,len(ids)+1),tile_id=ids[order],pearson_r=correlations[order])).to_csv(out/f'{plane}_pupil_sorted_tile_order.csv',index=False)
                arrays.update(pupil_pearson_r=correlations,plot_row_order=order)
            arrays.update(time_from_event_s=t,tile_ids=ids)
            for j,(label,key) in enumerate(events):
                original,centered,baseline,nbase,minimum=subtract_trial_baseline(
                    saved[key+'_trials'],t,baseline_start_s,min_valid_fraction)
                count=np.isfinite(original).sum(axis=0)
                means=[]
                for values in [original,centered]:
                    mean=np.full(count.shape,np.nan,dtype=np.float32)
                    np.divide(np.nansum(values,axis=0),count,out=mean,where=count>0)
                    means.append(mean)
                arrays.update({key+'_prestimulus_mean':baseline,key+'_prestimulus_sample_count':nbase,
                    key+'_centered_trials':centered,key+'_original_matched_mean':means[0],
                    key+'_centered_mean':means[1],key+'_valid_trial_count':count,
                    key+'_onset_rows':saved[key+'_onset_rows']})
                details[plane][key]=dict(trials=len(original),excluded_trial_tile_pairs=int((~np.isfinite(baseline)).sum()),
                    total_trial_tile_pairs=int(baseline.size),minimum_prestimulus_samples=minimum)
                for k,(name,mean) in enumerate(zip(['Original ΔF/F₀','Prestimulus-subtracted'],means)):
                    ax=axes[row,j*2+k]
                    im=ax.imshow(np.ma.masked_invalid(mean[order]*100),origin='upper',aspect='auto',
                        interpolation='nearest',cmap=cmap,vmin=-20,vmax=20,
                        extent=[t[0]-dt/2,t[-1]+dt/2,len(ids)-.5,-.5])
                    ax.axvline(0,color='.3',ls='--',lw=.8)
                    ticks=np.unique(np.linspace(0,len(ids)-1,min(5,len(ids)),dtype=int))
                    ax.set_yticks(ticks,labels=ids[order][ticks])
                    ax.set(title=f'{plane} · {label}\n{name}',xlabel='Time from onset (s)',
                           ylabel='Tile ID (pupil r: high → low)' if pupil_table_path is not None else 'Retained tile ID',xlim=(t[0],t[-1]))
        np.savez_compressed(out/f'{plane}_prestimulus_comparison{"_pupil_sorted" if pupil_table_path is not None else ""}.npz',**arrays)
    images=[ax.images[0] for ax in axes.flat]
    limit=event_color_limit([image.get_array() for image in images], color_limit_percent, color_percentile)
    for image in images: image.set_clim(-limit, limit)
    fig.colorbar(im,ax=list(axes.flat),label='ΔF/F₀ (%) / change from prestimulus level (percentage points)',
                 extend='both',shrink=.85)
    fig.suptitle(f'Original versus per-trial prestimulus subtraction ({baseline_start_s:g} to 0 s); matched eligible trials'+('\nTiles sorted by zero-lag pupil correlation: most positive at top' if pupil_table_path is not None else ''),fontsize=17)
    path=out/('gratings_reward_prestimulus_comparison_pupil_sorted.png' if pupil_table_path is not None else 'gratings_reward_prestimulus_comparison.png')
    fig.savefig(path,dpi=160);plt.close(fig)
    (out/('prestimulus_comparison_pupil_sorted_settings.json' if pupil_table_path is not None else 'prestimulus_comparison_settings.json')).write_text(json.dumps(dict(
        pupil_sort_table=str(pupil_table_path),sort_order="Descending zero-lag Pearson r, missing last" if pupil_table_path is not None else "Original tile ID",baseline_start_s=baseline_start_s,baseline_stop_s_exclusive=0,
        min_valid_fraction=min_valid_fraction,counts=details,
        method='For each trial and tile subtract its own finite prestimulus mean before trial averaging; original comparison uses identical eligible trials.',
        units='Arrays: fractions. Plot original: percent dff; centered: percentage-point change in dff.',
        color_limits=[-limit,limit],color_percentile=color_percentile,colormap='RdBu_r',z_scoring=False,
        limitations='Pre-event activity may contain effects of preceding events. Negative centered values mean lower than prestimulus fluorescence, not necessarily inhibition.'),indent=2))
    print('Saved:',path);print(json.dumps(details,indent=2))
    return path
