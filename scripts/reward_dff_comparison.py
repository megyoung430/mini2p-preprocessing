"""Frame-safe peri-event sampling for normalized fluorescence comparisons."""
import numpy as np

def frame_indices(selector, n_original):
    if selector is None:
        return np.arange(n_original)
    selector = np.asarray(selector)
    if selector.ndim != 1:
        raise ValueError('Frame selector must be one-dimensional')
    if selector.dtype == bool:
        if len(selector) != n_original:
            raise ValueError('Boolean selector must cover all original frames')
        return np.flatnonzero(selector)
    if not np.issubdtype(selector.dtype, np.integer):
        raise ValueError('Frame indices must be integers')
    if np.any(selector < 0) or np.any(selector >= n_original) or np.any(np.diff(selector) <= 0):
        raise ValueError('Frame indices must be in bounds, unique, and increasing')
    return selector

def peri_event(traces, times, original_ids, valid, onsets, relative_time, max_gap):
    """Interpolate only between adjacent valid original frames, never across gaps."""
    traces, times = np.asarray(traces), np.asarray(times)
    valid = np.asarray(valid, bool) & np.isfinite(traces)
    if traces.ndim != 2 or valid.shape != traces.shape or traces.shape[1] != len(times):
        raise ValueError('Trace/time/valid shapes differ')
    original_ids = np.asarray(original_ids)
    if len(original_ids) != len(times) or np.any(np.diff(original_ids) <= 0):
        raise ValueError('Original frame IDs must increase and match times')
    clock = np.flatnonzero(np.isfinite(times))
    t = times[clock]
    if len(t) < 2 or np.any(np.diff(t) <= 0) or max_gap <= 0:
        raise ValueError('Need an increasing clock and positive maximum gap')
    output = np.full((len(onsets), traces.shape[0], len(relative_time)), np.nan, np.float32)
    for trial, onset in enumerate(onsets):
        target = onset + relative_time
        right = np.searchsorted(t, target)
        r, l = np.clip(right, 0, len(t)-1), np.clip(right-1, 0, len(t)-1)
        exact = (right < len(t)) & (t[r] == target)
        li, ri = clock[l], clock[r]
        gap = t[r] - t[l]
        bracket = ((right > 0) & (right < len(t)) & (gap > 0) & (gap <= max_gap)
                   & (original_ids[ri] == original_ids[li] + 1))
        weight = np.divide(target-t[l], gap, out=np.zeros_like(target), where=gap > 0)
        values = traces[:, li]*(1-weight) + traces[:, ri]*weight
        keep = bracket[None, :] & valid[:, li] & valid[:, ri]
        values[:, exact] = traces[:, ri[exact]]
        keep[:, exact] = valid[:, ri[exact]]
        output[trial] = np.where(keep, values, np.nan)
    return output

def mean_sem(values, axis=0):
    """Finite-sample SEM; fewer than two observations has unknown SEM."""
    values = np.asarray(values, float)
    good = np.isfinite(values)
    counts = good.sum(axis=axis)
    mean = np.divide(np.where(good, values, 0).sum(axis=axis), counts,
                     out=np.full(counts.shape, np.nan), where=counts > 0)
    residual = values - np.expand_dims(mean, axis)
    ss = np.where(good, residual**2, 0).sum(axis=axis)
    variance = np.divide(ss, counts-1, out=np.full(counts.shape, np.nan), where=counts > 1)
    sem = np.sqrt(np.divide(variance, counts, out=np.full(counts.shape, np.nan), where=counts > 1))
    return mean, sem, counts
