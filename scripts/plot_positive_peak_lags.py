"""Descriptive positive peak-lag distribution from the corrected lag search."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Shared nLightG2 plot typography.
import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14



def plot_distribution(root):
    out=Path(root)/'pupil_lag_search'
    df=pd.read_csv(out/'tile_peak_lag_statistics.csv')
    # Restrict to tiles whose largest absolute correlation has positive sign.
    selected=df[df.peak_r>0].copy()
    groups=[('All planes',selected)]+[(name,selected[selected.plane==name]) for name in sorted(df.plane.unique())]
    fig,axes=plt.subplots(1,len(groups),figsize=(18,4),sharex=True,sharey=True,constrained_layout=True)
    rows=[]
    for ax,(name,g) in zip(axes,groups):
        x=g.peak_lag_s.dropna().to_numpy()
        ax.hist(x,bins=np.arange(17),weights=np.full(len(x),100/len(x)) if len(x) else None,color='#437c9e',edgecolor='white')
        if len(x):
            q=np.percentile(x,[25,50,75]);ax.axvline(q[1],color='black',ls='--',lw=1.2)
            rows.append(dict(plane=name,n_tiles=len(x),q25_s=q[0],median_s=q[1],q75_s=q[2],mean_s=x.mean(),n_6_to_16_s=int(((x>=6)&(x<=16)).sum()),percent_6_to_16_s=100*((x>=6)&(x<=16)).mean()))
        ax.set(title=f'{name} · n={len(x)}\nMedian {np.median(x):.2f} s' if len(x) else f'{name}: no tiles',xlabel='Peak fluorescence lag (s)',xticks=[0,4,8,12,16])
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Tiles (%)')
    fig.suptitle('Positive dominant correlation peaks: descriptive distribution\nLargest |r| over 0–16 s is positive; exploratory peaks, not significance-filtered',fontsize=12)
    path=out/'positive_peak_lag_distribution.png';fig.savefig(path,dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'positive_peak_lag_summary.csv',index=False)
    return path
