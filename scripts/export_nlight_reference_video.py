"""Cropped registered-intensity video for passive nLightG2."""
from pathlib import Path
import json
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw


def export_reference_style(root,start_min=0,stop_min=None,speed=10,window=40,contrast=(10,98),zoom=1,recording_label="passive",contrast_limits=None,output_dir=None,sensor_label="nLightG2"):
    rows=columns=None;tile_size=32
    root=Path(root);plane=root/'plane0'
    ops=np.load(plane/'ops.npy',allow_pickle=True).item()
    if (plane/'tile_fluorescence_dff.npz').exists():
        with np.load(plane/'tile_fluorescence_dff.npz') as z:times=z['time_s'];bad=z['badframes'].astype(bool)
    else:
        fs_ops=float(ops['fs'])
        if not np.isfinite(fs_ops) or fs_ops<=0:raise ValueError('Invalid Suite2p sampling rate')
        times=np.arange(int(ops['nframes']))/fs_ops
        bad=np.asarray(ops.get('badframes',np.zeros(len(times),bool)),dtype=bool)
        if bad.shape!=times.shape:raise ValueError('Bad-frame mask length mismatch')
    fs=1/np.median(np.diff(times));nt,h,w=int(ops['nframes']),int(ops['Ly']),int(ops['Lx'])
    y0,y1=map(int,ops['yrange']);x0,x1=map(int,ops['xrange'])
    if not np.isfinite(zoom) or zoom<1:raise ValueError('zoom must be finite and >=1')
    ch,cw=int((y1-y0)/zoom),int((x1-x0)/zoom)
    if min(ch,cw)<2:raise ValueError('Zoom crop is too small')
    ya=y0+((y1-y0)-ch)//2;yb=ya+ch
    xa=x0+((x1-x0)-cw)//2;xb=xa+cw
    if not speed>0 or window<1 or int(window)!=window:raise ValueError('Invalid speed/window')
    binary=plane/'data.bin'
    if not binary.exists():binary=Path(ops['reg_file'])
    movie=np.memmap(binary,dtype=np.int16,mode='r',shape=(nt,h,w))
    left=(window-1)//2;right=window-1-left
    stop_min=(times[-1]+1/fs)/60 if stop_min is None else stop_min
    indices=np.flatnonzero((times>=60*start_min)&(times<60*stop_min))
    if not len(indices):raise ValueError('No frames in video interval')
    def averaged(i):return movie[max(0,i-left):min(nt,i+right+1),ya:yb,xa:xb].mean(axis=0,dtype=np.float32)
    sample=indices[np.linspace(0,len(indices)-1,min(100,len(indices)),dtype=int)]
    low,high=np.percentile(np.stack([averaged(i) for i in sample]),contrast) if contrast_limits is None else contrast_limits
    if not np.isfinite([low,high]).all() or high<=low:raise ValueError('Invalid contrast limits')
    out=Path(output_dir) if output_dir is not None else root/'videos';out.mkdir(parents=True,exist_ok=True)
    name=f'{recording_label}_plane0_reference_zoom{zoom:g}x_avg{window}_{speed:g}x'
    path=out/(name+'.mp4')
    if path.exists():
        import datetime
        path=out/(name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.mp4')
    # Keep output <=30 fps by evenly sampling source frames when necessary.
    stride=max(1,int(np.ceil(fs*speed/30)));shown=indices[::stride];fps=fs*speed/stride
    size=(864,560);display_h=500;display_w=round((xb-xa)/(yb-ya)*display_h)
    xpad=(size[0]-display_w)//2
    from PIL import ImageFont
    font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",20)
    writer=imageio_ffmpeg.write_frames(str(path),size,fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-movflags','+faststart'])
    writer.send(None)
    try:
        for j,i in enumerate(shown):
            gray=np.round(np.clip((averaged(i)-low)/(high-low),0,1)*255).astype(np.uint8)
            im=Image.fromarray(gray).convert('RGB').resize((display_w,display_h),Image.Resampling.BILINEAR)
            canvas=Image.new('RGB',size);canvas.paste(im,(xpad,48));draw=ImageDraw.Draw(canvas)
            draw.text((xpad,15),f'{sensor_label} | plane 0 | {speed:g}x speed | {zoom:g}x zoom',fill='white',font=font)
            stamp=f'{int(times[i]//60):02d}:{times[i]%60:05.2f}'
            draw.text((xpad+display_w-100,15),stamp,fill='white',font=font)
            writer.send(np.asarray(canvas))
            if j==0:canvas.save(path.with_suffix('.png'))
    finally:writer.close()
    path.with_suffix('.json').write_text(json.dumps(dict(source=str(binary),plane=0,rows_inclusive=rows,columns_inclusive=columns,indexing='Zero-based tile coordinates relative to Suite2p valid crop origin',tile_size=tile_size,crop_yxyx=[ya,yb,xa,xb],time_minutes=[start_min,stop_min],speed=speed,fps=fps,stride=stride,output_frames=len(shown),duration_s=len(shown)/fps,rolling_frames=window,contrast_percentiles=contrast,contrast_limits=[float(low),float(high)],signal='Registered raw intensity, not dff; fixed contrast across time',bad_frames='Included in temporal average',zoom=zoom,presentation='Centered crop; grayscale; no grid; no scale bar; reference-style timestamp',rolling_edge_policy='Shorter averaging windows at recording boundaries'),indent=2))
    return path
