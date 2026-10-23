"""Compare rolling-p20 and installed Suite2p maximin, without changing source traces."""
from pathlib import Path
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_baseline_check')
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

import torch
from scipy.ndimage import gaussian_filter1d
from suite2p.extraction import baseline_maximin
import suite2p

def compare_baselines(root, windows=(30, 60, 120)):
    root = Path(root)
    out = root / 'baseline_comparison'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    rows, selections, checks = [], {}, {}
    for plane_dir in sorted(root.glob('plane[0-9]*')):
        path = plane_dir / 'tile_fluorescence_dff.npz'
        if not path.exists():
            continue
        with np.load(path) as z:
            F, ref, old_F0 = z['F'].astype(np.float32), z['dff'], z['F0']
            bad, times = z['badframes'].astype(bool), z['time_s']
            original_ids = z['tile_ids'] if 'tile_ids' in z else np.arange(len(F))
        meta = json.loads(path.with_suffix('.json').read_text())
        fs = float(meta['fs_hz'])
        ntiles, nt = F.shape
        masked = F.copy()
        masked[:, bad] = np.nan
        masked[~np.isfinite(masked)] = np.nan
        # Maximin cannot process NaNs: interpolate only its baseline-estimation input.
        # The final dff still excludes original bad/missing frames for both methods.
        filled = np.empty_like(F)
        for i in range(ntiles):
            valid = np.isfinite(masked[i])
            if valid.sum() < 2:
                raise ValueError(f'{plane_dir.name} tile {i}: fewer than 2 valid samples')
            filled[i] = np.interp(np.arange(nt), np.flatnonzero(valid), masked[i, valid])
        brightness = np.nanmedian(masked, axis=1)
        selected = []
        for label, percentile in [('dim', 10), ('typical', 50), ('bright', 90)]:
            order = np.argsort(np.abs(brightness - np.percentile(brightness, percentile)))
            tile = next(int(i) for i in order if int(i) not in [j for _, j in selected])
            selected.append((label, tile))
        smoothed = gaussian_filter1d(filled, sigma=fs, axis=1)
        excursion = (np.percentile(smoothed, 99, axis=1)-np.percentile(smoothed, 50, axis=1))/np.maximum(brightness, 1e-6)
        tile = next(int(i) for i in np.argsort(excursion)[::-1] if int(i) not in [j for _, j in selected])
        selected.append(('largest smoothed excursion (not validated activity)', tile))
        selections[plane_dir.name] = [(label, int(original_ids[i])) for label, i in selected]
        result = {}
        for method in ['rolling_p20', 'suite2p_maximin']:
            for seconds in windows:
                if method == 'rolling_p20':
                    win = max(3, int(round(seconds*fs)))
                    win += win % 2 == 0
                    min_samples = max(1, int(np.ceil(.25*min(win, nt))))
                    baseline = np.array([pd.Series(trace).rolling(win, center=True, min_periods=min_samples)
                                         .quantile(.2).to_numpy(dtype=np.float32) for trace in masked])
                else:
                    # Installed API sig_baseline is in FRAMES, not seconds.
                    residual = baseline_maximin(filled.copy(), win_baseline=seconds,
                        sig_baseline=10.0, fs=fs, batch_size=100, device=torch.device('cpu'))
                    baseline = filled-residual
                    win = int(seconds*fs)
                    win += win % 2 == 0
                dff = np.full_like(F, np.nan)
                np.divide(F-baseline, baseline, out=dff,
                          where=np.isfinite(masked) & np.isfinite(baseline) & (baseline > 0))
                key = f'{method}_{seconds}s'
                if method == 'rolling_p20' and seconds == 60:
                    np.testing.assert_allclose(baseline, old_F0, equal_nan=True, rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(dff, ref, equal_nan=True, rtol=1e-5, atol=1e-5)
                    checks[plane_dir.name] = '60s rolling-p20 reproduces saved F0 and dff'
                for i in range(ntiles):
                    for segment, use in [('full', np.ones(nt, bool)),
                                         ('interior', (times >= 60) & (times <= times[-1]-60))]:
                        good = use & np.isfinite(dff[i]) & np.isfinite(ref[i])
                        a, b = dff[i, good]*100, ref[i, good]*100
                        corr = np.corrcoef(a, b)[0, 1] if len(a)>2 and np.std(a)>0 and np.std(b)>0 else np.nan
                        rows.append(dict(plane=plane_dir.name, tile=int(original_ids[i]), brightness=float(brightness[i]),
                            method=method, window_s=seconds, actual_window_frames=int(win), segment=segment,
                            mean_pct=float(np.mean(a)), p99_pct=float(np.percentile(a,99)),
                            peak_pct=float(np.max(a)), std_pct=float(np.std(a)),
                            median_abs_difference_pp=float(np.median(np.abs(a-b))),
                            correlation=float(corr), baseline_median=float(np.nanmedian(baseline[i])),
                            invalid_baseline_fraction=float(np.mean(~np.isfinite(baseline[i]) | (baseline[i]<=0)))))
                ids = [i for _,i in selected]
                result[key+'_F0'] = baseline[ids]
                result[key+'_dff'] = dff[ids]
                print(plane_dir.name, key, flush=True)
        np.savez_compressed(out / f'{plane_dir.name}_representative_traces.npz',
            time_s=times, tile_ids=original_ids[[i for _,i in selected]], F=F[[i for _,i in selected]], **result)
        colors = {30:'tab:blue',60:'tab:orange',120:'tab:green'}
        fig, axes = plt.subplots(4,2,figsize=(17,13), constrained_layout=True)
        for row,(label,i) in enumerate(selected):
            axes[row,0].plot(times/60,F[i],color='.75',lw=.4,label='Raw F')
            for method in ['rolling_p20','suite2p_maximin']:
                for sec in windows:
                    key=f'{method}_{sec}s'
                    style='-' if method=='rolling_p20' else '--'
                    axes[row,0].plot(times/60,result[key+'_F0'][row],color=colors[sec],ls=style,lw=1,label=f'{method} {sec}s')
                    axes[row,1].plot(times/60,result[key+'_dff'][row]*100,color=colors[sec],ls=style,lw=.65)
            axes[row,0].set(title=f'Tile {original_ids[i]}: {label}',ylabel='Fluorescence',xlabel='Minutes')
            axes[row,1].set(title='ΔF/F₀ comparison: minutes 4–5',ylabel='ΔF/F₀ (%)',xlabel='Minutes',xlim=(4,5))
        axes[0,0].legend(fontsize=7,ncol=2)
        fig.suptitle(f'{plane_dir.name}: baseline-method and window sensitivity')
        fig.savefig(out/f'{plane_dir.name}_baseline_comparison.png',dpi=150)
        plt.close(fig)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'tile_metrics.csv',index=False)
    summary=metrics.groupby(['segment','method','window_s']).agg(
        n_tiles=('tile','size'), median_mean_pct=('mean_pct','median'),
        median_p99_pct=('p99_pct','median'),median_std_pct=('std_pct','median'),
        median_abs_difference_pp=('median_abs_difference_pp','median'),
        p90_abs_difference_pp=('median_abs_difference_pp',lambda x:x.quantile(.9)),
        median_correlation=('correlation','median')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    interior=metrics[metrics.segment=='interior']
    for ax,(field,label) in zip(axes.flat,[('mean_pct','Median tile mean ΔF/F₀ (%)'),('p99_pct','Median tile 99th-percentile ΔF/F₀ (%)'),
                                          ('median_abs_difference_pp','Median absolute difference from current method (pp)'),('correlation','Median correlation with current method')]):
        for method,style in [('rolling_p20','-'),('suite2p_maximin','--')]:
            g=interior[interior.method==method].groupby('window_s')[field]
            med=g.median();lo=g.quantile(.1);hi=g.quantile(.9)
            ax.plot(med.index,med.values,style,marker='o',label=method)
            ax.fill_between(med.index,lo.values,hi.values,alpha=.15)
        ax.set(xlabel='Baseline window (s)',ylabel=label,xticks=list(windows))
    axes[0,0].legend()
    fig.suptitle('All tiles: median and 10th–90th percentile across tiles; first/last 60 s excluded')
    fig.savefig(out/'baseline_sensitivity_summary.png',dpi=160);plt.close(fig)
    (out/'settings.json').write_text(json.dumps(dict(windows_s=windows,percentile=20,suite2p_sigma_frames=10,
        suite2p_version=getattr(suite2p,'__version__','unknown'),selections=selections,checks=checks,
        maximin_nan_policy='linear interpolation for baseline input only; original missing/bad samples excluded from dff',
        comparison='same F; no neuropil subtraction, deconvolution, or numerator smoothing; separate division by maximin F0',
        notes='No stimulus/behavior reference; highest smoothed excursion is not established biological activity. Plot and metrics are descriptive, not animal-level inference.'),indent=2))
    print(summary.to_string(index=False),flush=True)
    return out

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();compare_baselines(args.root)
