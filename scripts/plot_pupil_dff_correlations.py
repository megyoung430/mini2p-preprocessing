"""Zero-lag Pearson correlations using the supplied row-aligned pupil table."""
from pathlib import Path
import hashlib
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


def pairwise_pearson(x, y, minimum=30):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < minimum:
        return np.nan, n
    a = x[mask] - x[mask].mean()
    b = y[mask] - y[mask].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 * np.sqrt(n) or nb <= 1e-12 * np.sqrt(n):
        return np.nan, n
    return float(np.clip(a @ b / (na * nb), -1, 1)), n

def plot_pupil_correlations(root,table_path,pupil_column='pupil_diameter',minimum=30):
    root,table_path=Path(root),Path(table_path)
    if minimum<3:raise ValueError('minimum must be at least 3 paired frames')
    table=pd.read_pickle(table_path)
    if pupil_column not in table:raise ValueError(f'Missing pupil column: {pupil_column}')
    planes=sorted(p for p in root.glob('plane[0-9]*') if (p/'tile_fluorescence_dff.npz').exists())
    if not planes:raise ValueError('No saved tile fluorescence files')
    with np.load(planes[0]/'tile_fluorescence_dff.npz') as saved:
        times=saved['time_s'];nt=len(times)
    if len(table)!=nt:
        manifests=list((root/'input').glob('*complete_volumes.json'))
        if len(manifests)!=1:raise ValueError('Cannot verify table length against recording')
        meta=json.loads(manifests[0].read_text())
        if len(table)!=nt+int(meta['omitted_final_pages']>0) or meta['output_pages']!=nt*len(planes):
            raise ValueError('Table rows do not match complete imaging volumes')
    n_extra=len(table)-nt
    pupil=pd.to_numeric(table[pupil_column].iloc[:nt],errors='coerce').to_numpy(dtype=float, copy=True)
    pupil[~np.isfinite(pupil)|(pupil<=0)]=np.nan
    if np.isfinite(pupil).sum()<minimum:raise ValueError('Too few positive finite pupil measurements')
    out=root/'pupil_correlation';out.mkdir(exist_ok=True)
    rows=[];whole_rows=[]
    fig,axes=plt.subplots(len(planes),2,figsize=(15,3.7*len(planes)),constrained_layout=True,squeeze=False)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray')
    for row,plane in enumerate(planes):
        with np.load(plane/'tile_fluorescence_dff.npz') as z:
            dff=z['dff'];ids=z['tile_ids'];bounds=z['tile_bounds_yxyx'];bad=z['badframes'].astype(bool)
            if dff.shape!=(len(ids),nt) or not np.allclose(z['time_s'],times):raise ValueError('Inconsistent imaging frame axes')
        with np.load(plane/'tile_selection.npz') as z:
            if not np.array_equal(ids,np.flatnonzero(z['keep_mask'])):raise ValueError('Retained tile IDs do not match selection')
        selection=json.loads((plane/'tile_fluorescence_dff.json').read_text())
        nr,nc=selection['grid_shape']
        spatial=np.full((nr,nc),np.nan)
        paired_pupil=pupil.copy();paired_pupil[bad]=np.nan
        for tile,trace,bound in zip(ids,dff,bounds):
            r,n=pairwise_pearson(trace,paired_pupil,minimum)
            spatial[int(tile)//nc,int(tile)%nc]=r
            rows.append(dict(plane=plane.name,tile_id=int(tile),pearson_r=r,n_paired_frames=n,
                             y_start=int(bound[0]),y_stop=int(bound[1]),x_start=int(bound[2]),x_stop=int(bound[3])))
        ax=axes[row,0]
        im=ax.imshow(np.ma.masked_invalid(spatial),cmap=cmap,vmin=-1,vmax=1,interpolation='nearest',origin='upper')
        ax.set(title=f'{plane.name}: tile ΔF/F₀ versus pupil',xlabel='Tile-grid column',ylabel='Tile-grid row')
        with np.load(plane/'whole_image_roi_dff.npz') as z:
            if not np.allclose(z['time_s'],times):raise ValueError('Whole-image ROI time axis differs')
            whole=z['dff']*100
            paired_whole=paired_pupil.copy();paired_whole[z['badframes'].astype(bool)]=np.nan
        r,n=pairwise_pearson(whole,paired_whole,minimum)
        good=np.isfinite(whole)&np.isfinite(paired_whole)
        axes[row,1].scatter(paired_whole[good],whole[good],s=5,alpha=.25,color='teal',edgecolors='none')
        axes[row,1].set(title=f'{plane.name}: whole-image ROI · r={r:.3f} · n={n}',
                       xlabel=f'{pupil_column} (recorded units)',ylabel='Whole-image ΔF/F₀ (%)')
        whole_rows.append(dict(plane=plane.name,pearson_r=r,n_paired_frames=n))
    fig.colorbar(im,ax=list(axes[:,0]),label='Pearson r (zero lag)',shrink=.85)
    fig.suptitle('Passive recording: pupil–fluorescence correlation\nGray tiles: excluded or insufficient/constant data',fontsize=15)
    path=out/'pupil_diameter_dff_correlations.png';fig.savefig(path,dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'tile_pupil_correlations.csv',index=False)
    pd.DataFrame(whole_rows).to_csv(out/'whole_image_pupil_correlations.csv',index=False)
    (out/'settings.json').write_text(json.dumps(dict(table_path=str(table_path),pupil_column=pupil_column,
        source_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),dropped_final_table_rows=n_extra,
        alignment='User-specified table row i = Suite2p frame/volume i; FrameIndex not used as an imaging index',
        method='Pearson correlation at zero lag across the full recording; pairwise finite samples; exclude imaging bad frames and nonpositive pupil values',
        min_paired_frames=minimum,pupil_finite_frames=int(np.isfinite(pupil).sum()),imaging_frames=nt,
        input='Saved continuous dff, not trial-averaged or prestimulus-subtracted',
        limitations='Descriptive associations; no independence-based p-values. Shared stimulus, reward, movement, or drift may contribute. No lag search, interpolation, or smoothing. Whole-image ROI includes the full valid image.'),indent=2))
    print('Whole-image correlations:',whole_rows)
    print('Finite pupil frames:',np.isfinite(pupil).sum(),'/',nt)
    print('Saved:',path)
    return path
