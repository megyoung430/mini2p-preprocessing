"""Cropped registered-intensity video for passive nLightG2."""
from pathlib import Path
import json
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw


def export_crop(root,start_min=1,stop_min=2,rows=(7,11),columns=(10,14),tile_size=32,speed=10,window=20,contrast=(10,98)):
    root=Path(root);plane=root/'plane0'
    ops=np.load(plane/'ops.npy',allow_pickle=True).item()
    with np.load(plane/'tile_fluorescence_dff.npz') as z:times=z['time_s'];bad=z['badframes'].astype(bool)
    fs=1/np.median(np.diff(times));nt,h,w=int(ops['nframes']),int(ops['Ly']),int(ops['Lx'])
    y0,y1=map(int,ops['yrange']);x0,x1=map(int,ops['xrange'])
    ya,yb=y0+rows[0]*tile_size,y0+(rows[1]+1)*tile_size
    xa,xb=x0+columns[0]*tile_size,x0+(columns[1]+1)*tile_size
    if not (y0<=ya<yb<=y1 and x0<=xa<xb<=x1):raise ValueError('Requested tiles fall outside valid crop')
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
    low,high=np.percentile(np.stack([averaged(i) for i in sample]),contrast)
    if high<=low:raise ValueError('Invalid contrast limits')
    out=root/'videos';out.mkdir(exist_ok=True)
    name=f'passive_plane0_rows{rows[0]}-{rows[1]}_cols{columns[0]}-{columns[1]}_tile{tile_size}_{start_min:g}-{stop_min:g}min_{speed:g}x'
    path=out/(name+'.mp4')
    if path.exists():
        import datetime
        path=out/(name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.mp4')
    # Keep output <=30 fps by evenly sampling source frames when necessary.
    stride=max(1,int(np.ceil(fs*speed/30)));shown=indices[::stride];fps=fs*speed/stride
    scale=4;size=((xb-xa)*scale,(yb-ya)*scale+48)
    writer=imageio_ffmpeg.write_frames(str(path),size,fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-movflags','+faststart'])
    writer.send(None)
    try:
        for j,i in enumerate(shown):
            gray=np.round(np.clip((averaged(i)-low)/(high-low),0,1)*255).astype(np.uint8)
            im=Image.fromarray(gray).convert('RGB').resize((size[0],size[1]-48),Image.Resampling.NEAREST)
            canvas=Image.new('RGB',size);canvas.paste(im,(0,48));draw=ImageDraw.Draw(canvas)
            for x in range(0,size[0],tile_size*scale):draw.line((x,48,x,size[1]),fill=(0,100,100),width=1)
            for y in range(48,size[1],tile_size*scale):draw.line((0,y,size[0],y),fill=(0,100,100),width=1)
            flag=' | flagged window' if bad[max(0,i-left):min(nt,i+right+1)].any() else ''
            draw.text((10,8),f'Plane 0 | {times[i]:.2f} s | {speed:g}x | rolling {window} frames{flag}',fill='white')
            draw.text((10,27),f'Rows {rows}, columns {columns} | 32 px tiles | fixed contrast',fill='white')
            writer.send(np.asarray(canvas))
            if j==0:canvas.save(path.with_suffix('.png'))
    finally:writer.close()
    path.with_suffix('.json').write_text(json.dumps(dict(source=str(binary),plane=0,rows_inclusive=rows,columns_inclusive=columns,indexing='Zero-based tile coordinates relative to Suite2p valid crop origin',tile_size=tile_size,crop_yxyx=[ya,yb,xa,xb],time_minutes=[start_min,stop_min],speed=speed,fps=fps,stride=stride,output_frames=len(shown),duration_s=len(shown)/fps,rolling_frames=window,contrast_percentiles=contrast,contrast_limits=[float(low),float(high)],signal='Registered raw intensity, not dff; fixed contrast across time',bad_frames='Included and flagged'),indent=2))
    return path
