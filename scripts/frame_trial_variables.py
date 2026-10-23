"""Attach canonical magnitude-bandit trial fields to their imaging-frame intervals."""
import sys
import json
import numpy as np
import pandas as pd
from align_scanimage_pycontrol import load_bandit_session


def attach_trial_variables(table, raw, tsv, project):
    # The project currently uses both src.behavior_import and behavior_import imports.
    for path in (project, project / 'src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    canonical = load_bandit_session(tsv, project)
    info = canonical.get('trial_info', [])
    records = [r for segment in info for r in segment] if info and isinstance(info[0], list) else info
    n = len(records)
    printed = raw[raw['type'].eq('variable') & raw['subtype'].eq('print')].sort_values('time', kind='stable')
    # Match canonical trial records back to exact raw print rows; port-finding
    # trials are excluded by extract_trials and must not shift the correspondence.
    parsed = []
    for row in printed.itertuples():
        try:
            value = json.loads(row.content)
        except (TypeError, ValueError):
            continue
        parsed.append((float(row.time), value))
    times, cursor = [], 0
    for record in records:
        while cursor < len(parsed) and parsed[cursor][1] != record:
            cursor += 1
        if cursor == len(parsed):
            raise ValueError('Cannot map canonical trial_info back to raw print timestamps')
        times.append(parsed[cursor][0]); cursor += 1
    wait = np.sort(raw[raw['type'].eq('state') & raw['content'].eq('wait_init')]['time'].to_numpy(float))
    frame_times = table['behavior_time_s'].to_numpy()
    trial_index = np.full(len(table), -1, dtype=int)
    starts, ends = [], []
    recording_end = float(pd.to_numeric(raw['time'], errors='coerce').max())
    used = set()
    for i, logged in enumerate(times):
        j = np.searchsorted(wait, logged, side='right') - 1
        if j < 0 or j in used:
            raise ValueError('Trial does not have a unique preceding wait_init; refusing ambiguous assignment')
        used.add(j)
        start = wait[j]
        end = wait[j+1] if j+1 < len(wait) else recording_end
        starts.append(start); ends.append(end)
        trial_index[(frame_times >= start) & (frame_times < end)] = i
    valid = (trial_index >= 0) & table['alignment_valid'].to_numpy()
    columns = {}

    def add(name, values):
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != n:
            return
        mapped = pd.Series([None] * len(table), dtype=object)
        # Object assignment preserves list-valued variables without broadcasting.
        for f in np.flatnonzero(valid):
            mapped.iat[f] = values[trial_index[f]]
        columns[name] = mapped.infer_objects()

    excluded = {'data','df','trial_info','trial_variables','reversal_events'}
    for key, values in canonical.items():
        if key in excluded:
            continue
        if isinstance(values, dict):
            for subkey, seq in values.items():
                add(f'{key}.{subkey}', seq)
        else:
            add(key, values)
    # Keep core columns available even in recordings with no completed bandit trials.
    for key in ('trial', 'blocks', 'trials_in_block', 'choice', 'chosen_rank',
                'reward_pulses', 'chosen_magnitude'):
        if key not in columns:
            add(key, [None] * n)
    # Preserve additional logged task variables under an explicit namespace.
    if n:
        flat = pd.json_normalize(records, sep='.')
        for name in flat:
            add(f'task.{name}', flat[name].tolist())
    add('trial_start_s', starts)
    add('trial_end_s', ends)
    add('trial_logged_s', times)
    columns['trial_index'] = pd.Series(np.where(valid,trial_index,np.nan),dtype='Int64')
    logged_flag = pd.Series([pd.NA]*len(table),dtype='boolean')
    if n:
        logged_flag.loc[valid] = frame_times[valid] >= np.asarray(times)[trial_index[valid]]
    columns['trial_outcome_logged'] = logged_flag
    result = pd.concat([table,pd.DataFrame(columns)],axis=1)
    if result.columns.duplicated().any():
        raise ValueError('Duplicate trial/frame columns')
    return result, dict(n_trials=n, trial_columns=list(columns),
        trial_assignment='Retrospective trial annotation from preceding wait_init to next wait_init; incomplete/uncovered trials unknown.',
        outcome_timing='choice/reward fields describe the completed trial even before choice; trial_outcome_logged indicates whether its print record had occurred.')
