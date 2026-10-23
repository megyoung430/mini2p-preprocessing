"""Display baseline sensitivity using saved, unnormalized whole-FOV fluorescence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_whole_fov(root, fixed_percentile=10, windows_s=(180, 300), plane=None):
    """Read only: return figures and amplitude metrics; never overwrite analysis files."""
    if not 0 <= fixed_percentile <= 100:
        raise ValueError('fixed_percentile must be between 0 and 100')
    if any(not np.isfinite(w) or w <= 0 for w in windows_s):
        raise ValueError('Baseline windows must be positive and finite')
    if plane is not None:
        if isinstance(plane, bool) or not isinstance(plane, (int, np.integer)) or plane < 0:
            raise ValueError('plane must be a nonnegative integer, or None for all planes')
        source = Path(root) / f'plane{plane}' / 'whole_image_roi_dff.npz'
        if not source.is_file():
            raise FileNotFoundError(f'Run whole-image ROI extraction for plane{plane}: {source}')
        sources = [source]
    else:
        sources = sorted(Path(root).glob('plane*/whole_image_roi_dff.npz'))
    if not sources:
        raise FileNotFoundError('Run the whole-image ROI extraction cell first')
    figures, rows = [], []
    for source in sources:
        meta = json.loads(source.with_suffix('.json').read_text())
        with np.load(source) as z:
            f = z['F'].astype(float)
            t = z['time_s'].copy()
            bad = z['badframes'].astype(bool)
            current_f0, current_dff = z['F0'].copy(), z['dff'].copy()
        if len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
            raise ValueError('Invalid imaging time axis')
        fs = 1 / np.median(np.diff(t))
        valid = ~bad & np.isfinite(f)
        masked = np.where(valid, f, np.nan)
        if not valid.any():
            raise ValueError('No valid fluorescence samples')
        current_label = (f"Current: {meta['baseline_window_frames']/fs:.0f}s "
                         f"rolling p{meta['baseline_percentile']:g}")
        baselines = {current_label: current_f0,
                     f'Fixed full-recording p{fixed_percentile:g}':
                         np.full(len(f), np.nanpercentile(masked, fixed_percentile))}
        for seconds in windows_s:
            if seconds > t[-1] - t[0]:
                print(f'{source.parent.name}: skipping {seconds:g}s window, longer than recording')
                continue
            win = max(3, int(round(seconds * fs)))
            win += win % 2 == 0
            minimum = max(1, int(np.ceil(.25 * min(win, len(f)))))
            baselines[f'{seconds:g}s rolling p{meta["baseline_percentile"]:g}'] = (
                pd.Series(masked).rolling(win, center=meta.get('baseline_centered', True),
                                          min_periods=minimum)
                .quantile(meta['baseline_percentile']/100).to_numpy())
        traces = {}
        for label, baseline in baselines.items():
            dff = np.full(len(f), np.nan)
            np.divide(f-baseline, baseline, out=dff,
                      where=valid & np.isfinite(baseline) & (baseline > 0))
            traces[label] = dff
        np.testing.assert_allclose(traces[current_label], current_dff, equal_nan=True,
                                   rtol=1e-5, atol=1e-7)
        # Identical samples for every amplitude comparison.
        common = np.logical_and.reduce([np.isfinite(v) for v in traces.values()])
        if common.sum() < 3:
            raise ValueError('Too few shared valid samples')
        fig, axes = plt.subplots(2+len(traces), 1, figsize=(15, 3*(2+len(traces))),
                                 sharex=True, constrained_layout=True)
        minutes = t / 60
        axes[0].plot(minutes, masked, color='black', lw=.6)
        axes[0].set(title='Raw whole-FOV fluorescence (motion corrected; before ΔF/F)',
                    ylabel='Fluorescence (a.u.)')
        axes[1].plot(minutes, masked, color='.8', lw=.4, label='Raw F')
        colors = plt.get_cmap('tab10').colors
        for i, (label, baseline) in enumerate(baselines.items()):
            axes[1].plot(minutes, baseline, color=colors[i % 10], lw=1, label=label)
        axes[1].set(ylabel='F₀ (a.u.)', title='Baseline definitions')
        axes[1].legend(fontsize=10, ncol=2)
        pooled = np.concatenate([v[np.isfinite(v)]*100 for v in traces.values()])
        lo, hi = min(0., pooled.min()), max(0., pooled.max())
        pad = max((hi-lo)*.05, .01)
        ref = traces[current_label][common]*100
        for i, (label, dff) in enumerate(traces.items()):
            ax = axes[i+2]
            ax.plot(minutes, dff*100, color=colors[i % 10], lw=.6)
            ax.axhline(0, color='.5', lw=.5)
            ax.set(title=label, ylabel='ΔF/F₀ (%)', ylim=(lo-pad, hi+pad))
            values = dff[common]*100
            rows.append(dict(plane=source.parent.name, method=label,
                             median_percent=float(np.median(values)),
                             p95_minus_p5_pp=float(np.percentile(values,95)-np.percentile(values,5)),
                             std_pp=float(np.std(values)),
                             correlation_with_current=float(np.corrcoef(values,ref)[0,1])
                             if np.std(values)>0 and np.std(ref)>0 else np.nan,
                             shared_samples=int(common.sum())))
        axes[-1].set_xlabel('Time (minutes)')
        fig.suptitle(f'{Path(root).parent.name} · {source.parent.name} · whole-FOV baseline sensitivity\n'
                     'No temporal smoothing; identical ΔF/F axes across methods')
        figures.append(fig)
    return figures, pd.DataFrame(rows)
