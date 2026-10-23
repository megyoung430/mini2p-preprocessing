"""Pupil correlation maps with outlines for both signs of significant association."""
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

from matplotlib.patches import Rectangle, Patch


def plot_maps(root,alpha=.05):
    root=Path(root)
    if not 0<alpha<1:raise ValueError('alpha must be between zero and one')
    significance=pd.read_csv(root/'pupil_correlation/tile_pupil_significance.csv')
    correlations=pd.read_csv(root/'pupil_correlation/tile_pupil_correlations.csv')
    keys=['plane','tile_id']
    merged=correlations.merge(significance[keys+['pearson_r','q_bh']],on=keys,suffixes=('_current','_tested'),validate='one_to_one',how='outer',indicator=True)
    if not (merged['_merge']=='both').all() or not np.allclose(merged.pearson_r_current,merged.pearson_r_tested,atol=1e-7,equal_nan=True):
        raise ValueError('Correlation and significance files differ; rerun both analyses')
    planes=sorted(merged.plane.unique())
    fig,axes=plt.subplots(1,len(planes),figsize=(4.5*len(planes),5),squeeze=False,constrained_layout=True)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('darkgrey')
    summaries=[]
    for ax,plane in zip(axes.flat,planes):
        folder=root/plane
        with np.load(folder/'tile_fluorescence_dff.npz') as z:ids=z['tile_ids']
        with np.load(folder/'tile_selection.npz') as z:
            if not np.array_equal(ids,np.flatnonzero(z['keep_mask'])):raise ValueError('Tile selection has changed')
        rows=merged[merged.plane==plane].set_index('tile_id')
        if set(rows.index)!=set(ids):raise ValueError('Significance tile IDs do not match retained tiles')
        rows=rows.loc[ids]
        shape=json.loads((folder/'tile_fluorescence_dff.json').read_text())['grid_shape']
        spatial=np.full(shape,np.nan);spatial.flat[ids]=rows.pearson_r_current
        im=ax.imshow(np.ma.masked_invalid(spatial),cmap=cmap,vmin=-1,vmax=1,interpolation='nearest',origin='upper')
        selected=rows.q_bh.le(alpha)&rows.pearson_r_current.notna()
        for tile in rows.index[selected]:
            y,x=divmod(int(tile),shape[1])
            # ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,edgecolor='black',linewidth=.65))
        npos=int((selected&(rows.pearson_r_current>0)).sum());nneg=int((selected&(rows.pearson_r_current<0)).sum())
        ax.set(title=f'{plane}: {npos} positive, {nneg} negative',xlabel='Tile column',ylabel='Tile row')
        summaries.append(dict(plane=plane,retained_tiles=len(ids),significant_positive=npos,significant_negative=nneg))
    fig.colorbar(im,ax=list(axes.flat),label='Pearson r: ΔF/F vs pupil diameter',shrink=.8)
    fig.suptitle('Zero-lag pupil correlation',fontsize=13)
    # fig.suptitle(f'Zero-lag pupil correlation · black outlines: BH q ≤ {alpha}\nTwo-sided circular-shift test; gray tiles excluded',fontsize=13)
    out=root/'pupil_correlation';path=out/'pupil_correlations_significant_outlines.png'
    fig.savefig(path,dpi=220);plt.close(fig)
    pd.DataFrame(summaries).to_csv(out/'outlined_significance_summary.csv',index=False)
    (out/'outlined_heatmap_settings.json').write_text(json.dumps(dict(alpha=alpha,correlation='Raw pupil diameter, zero lag; saved Pearson correlations',test='Existing two-sided circular-shift test with pooled BH across retained tiles in all detected planes',color_limits=[-1,1],outline='Black, both positive and negative q <= alpha',excluded='Gray, no outline',limitation='Exploratory within-recording significance; conditional on alignment and shift-null assumptions'),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    return path
