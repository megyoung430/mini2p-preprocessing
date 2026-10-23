"""Read explicit event windows from nLight all_data.pkl without guessing time zero."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_event_onsets(path,n_frames,onset_position,grating_groups=('gr_2','gr_4')):
    """onset_position: integer or dict with grating_1, grating_2, reward positions.

    Index arrays contain pre/post-event samples, not just onsets. The time-zero
    position must come from the acquisition/alignment code, not the first index.
    """
    if onset_position is None:
        raise ValueError('Specify the verified time-zero position within all_data.pkl event index arrays')
    data=pd.read_pickle(Path(path))
    if not isinstance(data,dict):raise ValueError('Expected all_data.pkl dictionary')
    if 'dff_Zscore' in data and np.shape(data['dff_Zscore'])[-1]!=n_frames:
        raise ValueError('all_data.pkl imaging time dimension does not match current recording')
    windows={'grating_1':data['grating_indices'][grating_groups[0]],
             'grating_2':data['grating_indices'][grating_groups[1]],
             'reward':data['reward_indices']}
    result={}
    for event,trials in windows.items():
        position=onset_position[event] if isinstance(onset_position,dict) else onset_position
        if int(position)!=position or position<0:raise ValueError('Onset position must be a nonnegative integer')
        starts=[]
        for trial in trials:
            indices=np.asarray(trial).ravel()
            if len(indices)<=position:raise ValueError(f'{event}: onset position outside saved window')
            value=indices[int(position)]
            if not np.isfinite(value) or value!=int(value) or not 0<=value<n_frames:
                raise ValueError(f'{event}: invalid imaging onset index')
            starts.append(int(value))
        result[event]=np.unique(starts)
    return result
