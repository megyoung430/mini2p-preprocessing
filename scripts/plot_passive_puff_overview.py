"""Combined-plane full-recording FOV trace and spatial-tile heatmap."""
from pathlib import Path
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Helvetica','Arial'],'xtick.labelsize':12,'ytick.labelsize':12,'axes.labelsize':12,'axes.titlesize':14})


def combined(root, tile_size=32, baseline_percentile=50, baseline_window_s=60, exclusion=15, heatmap_plane=None):
    import pandas as pd
    root=Path(root);whole=[];tiles=[];grids=[]
    for p in sorted(root.glob('plane[0-9]*')):
        ops=np.load(p/'ops.npy',allow_pickle=True).item()
        with np.load(p/'whole_image_roi_dff.npz') as z:
            t=z['time_s'];bad=z['badframes'].astype(bool);wf=z['F'].astype(float)
        if tiles and not np.allclose(times,t):raise ValueError('Plane timing differs')
        times=t;nt=len(t);fs=1/np.median(np.diff(t))
        ya,yb=map(int,ops['yrange']);xa,xb=map(int,ops['xrange'])
        nr,nc=(yb-ya)//tile_size,(xb-xa)//tile_size;grids.append((nr,nc))
        binary=p/'data.bin'
        if not binary.exists():binary=Path(ops['reg_file'])
        movie=np.memmap(binary,dtype=np.int16,mode='r',shape=(nt,int(ops['Ly']),int(ops['Lx'])))
        f=np.empty((nr*nc,nt),np.float32)
        for a in range(0,nt,100):
            b=min(a+100,nt)
            block=np.asarray(movie[a:b,ya:ya+nr*tile_size,xa:xa+nc*tile_size],dtype=np.float32)
            f[:,a:b]=block.reshape(b-a,nr,tile_size,nc,tile_size).mean(axis=(2,4)).reshape(b-a,-1).T
        del movie
        bright=np.mean(f[:,~bad],axis=1);keep=bright>=np.percentile(bright,exclusion)
        window=max(3,round(baseline_window_s*fs));window+=window%2==0
        minimum=max(1,int(np.ceil(.25*min(window,nt))))
        def dff(a):
            a=a.copy();a[bad]=np.nan
            baseline=pd.Series(a).rolling(window,center=True,min_periods=minimum).quantile(baseline_percentile/100).to_numpy()
            out=np.full(nt,np.nan);np.divide(a-baseline,baseline,out=out,where=np.isfinite(baseline)&(baseline>0));return out
        heat=np.full(f.shape,np.nan)
        for i in np.flatnonzero(keep):heat[i]=dff(f[i])
        if heatmap_plane is None or p.name==f"plane{heatmap_plane}":tiles.append(heat)
        whole.append(dff(wf))
    if len(set(grids))!=1:raise ValueError('Plane grids differ')
    def avg(a):
        a=np.stack(a);n=np.isfinite(a).sum(0);out=np.full(n.shape,np.nan);np.divide(np.nansum(a,0),n,out=out,where=n>0);return out
    if not tiles:raise ValueError('Selected heatmap plane was not found')
    heat=avg(tiles);ids=np.flatnonzero(np.isfinite(heat).any(1))
    return times,avg(whole)*100,heat[ids]*100,ids,grids[0]


def plot_comparison(results,output=None,time_bin_s=1.0,recordings=("passive","raw_data_2x_puff"),labels=("Passive","Passive + airpuff"),heatmap_plane=None):
    if time_bin_s<=0:raise ValueError('time_bin_s must be positive')
    if len(recordings)!=2 or len(labels)!=2:raise ValueError('Provide two recordings and labels')
    results=Path(results);data=[combined(results/name/'suite2p',heatmap_plane=heatmap_plane) for name in recordings]
    if data[0][4]!=data[1][4]:raise ValueError('Condition grid shapes differ')
    ids=np.intersect1d(data[0][3],data[1][3])
    if not len(ids):raise ValueError('No common retained tile positions')
    n_common=min(len(d[0]) for d in data)
    reference_time=data[0][0][:n_common]
    if not np.allclose(reference_time,data[1][0][:n_common],atol=1e-6):
        raise ValueError('Sample timing differs; cannot directly match timepoint counts')
    data=[(t[:n_common],mean[:n_common],heat[:,:n_common],tile_ids,grid)
          for t,mean,heat,tile_ids,grid in data]
    end=reference_time[-1]+np.median(np.diff(reference_time))
    edges=np.arange(0,np.ceil(end/time_bin_s)+1)*time_bin_s;centers=(edges[:-1]+edges[1:])/2
    def binmean(a,t):
        index=np.floor(t/time_bin_s).astype(int);out=np.full((a.shape[0],len(centers)),np.nan)
        for b in np.unique(index):
            x=a[:,index==b];n=np.isfinite(x).sum(1)
            np.divide(np.nansum(x,1),n,out=out[:,b],where=n>0)
        return out
    binned=[]
    for t,mean,heat,tile_ids,grid in data:
        selected=np.searchsorted(tile_ids,ids)
        binned.append(binmean(np.vstack([mean,heat[selected]]),t))
    fig,axes=plt.subplots(2,2,figsize=(16,9),sharex=True,sharey='row',constrained_layout=True,gridspec_kw={'height_ratios':[1,3]})
    vmax=max(1,float(np.nanpercentile(np.abs(np.concatenate([d[1:].ravel() for d in binned])),99)))
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgray')
    for col,(label,a) in enumerate(zip(labels,binned)):
        axes[0,col].plot(centers/60,a[0],color='black',lw=1)
        axes[0,col].set(title=label,ylabel='Whole-FOV ΔF/F₀ (%)')
        im=axes[1,col].imshow(np.ma.masked_invalid(a[1:]),origin='upper',aspect='auto',interpolation='nearest',cmap=cmap,vmin=-vmax,vmax=vmax,extent=[edges[0]/60,edges[-1]/60,len(ids)-.5,-.5])
        ticks=np.unique(np.linspace(0,len(ids)-1,6,dtype=int));axes[1,col].set_yticks(ticks,labels=ids[ticks])
        axes[1,col].set(xlabel='Time (min)',ylabel='Spatial tile ID (32 px)',xlim=(0,edges[-1]/60))
        axes[0,col].spines[['top','right']].set_visible(False)
    fig.colorbar(im,ax=list(axes[1]),label='Tile ΔF/F₀ (%)',extend='both',shrink=.9)
    heatmap_label='mean across planes' if heatmap_plane is None else f'plane {heatmap_plane}'
    fig.suptitle(f'Whole-FOV: mean across planes · heatmaps: {heatmap_label} · {time_bin_s:g}-s bins · {len(ids)} common 32-pixel tile positions\nMatched 60-s rolling median baseline; cropped to shorter recording',fontsize=14)
    if output is not None:fig.savefig(output,dpi=160)
    print(f'{len(ids)} shared tile positions; {n_common} matched source timepoints; {len(centers)} identical time bins; saved analysis arrays unchanged.')
    return fig
