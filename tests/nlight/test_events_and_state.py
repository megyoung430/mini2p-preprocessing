import json

import numpy as np
import pandas as pd
import pytest

from conftest import write_plane
from nlight import analyses, behavior, events, recording_comparison, regression, session, state, traces


# --- behaviour alignment -------------------------------------------------

def test_event_onsets_are_rising_edges():
    np.testing.assert_array_equal(behavior.event_onsets([0, 2, 2, 0, 2, 4, 4], 2), [1, 4])


def test_row_count_must_match_unless_extra_row_is_verified():
    table = pd.DataFrame({"Stim": np.zeros(11)})
    with pytest.raises(ValueError, match="11 rows for 10"):
        behavior.align_rows(table, 10)
    assert len(behavior.align_rows(table, 10, extra_rows_allowed=1)) == 10


def test_verified_extra_row_reads_the_consolidation_record(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "x_complete_volumes.json").write_text(
        json.dumps(dict(output_pages=40, omitted_final_pages=2)))
    assert behavior.verified_extra_rows(tmp_path, 10, 4) == 1
    with pytest.raises(ValueError, match="pages"):
        behavior.verified_extra_rows(tmp_path, 11, 4)


def test_behavior_from_table_cleans_pupil_and_tolerates_missing_columns():
    table = pd.DataFrame({"pupil_diameter": [1.0, -1.0, np.nan, 2.0]})
    b = behavior.behavior_from_table(table)
    assert b.onsets == {} and b.speed is None
    np.testing.assert_array_equal(b.pupil, [1.0, np.nan, np.nan, 2.0])


# --- events ---------------------------------------------------------------

def test_incomplete_windows_are_dropped():
    offsets, _ = events.window_offsets(1.0, 2, 2)
    trials, used = events.aligned_trials(np.arange(10.0), np.array([1, 5, 9]), offsets)
    np.testing.assert_array_equal(used, [5])
    np.testing.assert_array_equal(trials[0, 0], [3, 4, 5, 6, 7])


def test_prestimulus_subtraction_matches_trials_and_excludes_sparse_baselines():
    rel = np.array([-2, -1, 0, 1.0])
    trials = np.array([[[1, 1, 5, 5.0]], [[np.nan, np.nan, 5, 5.0]]])
    original, centred = events.subtract_prestimulus(trials, rel, -2, 0.8)
    np.testing.assert_array_equal(centred[0, 0], [0, 0, 4, 4])
    assert np.isnan(original[1]).all() and np.isnan(centred[1]).all()


def test_mean_sem_counts_finite_trials():
    mean, sem, n = events.mean_sem(np.array([[1.0, np.nan], [3.0, 2.0]]))
    np.testing.assert_allclose(mean, [2, 2])
    np.testing.assert_allclose(sem, [1, np.nan])
    np.testing.assert_array_equal(n, [2, 1])


# --- state ----------------------------------------------------------------

def test_lag_correlation_peaks_at_the_true_delay():
    rng = np.random.default_rng(0)
    p = np.convolve(rng.normal(size=2000), np.ones(10) / 10, "same")
    f = np.r_[np.zeros(5), p[:-5]]  # fluorescence follows pupil by 5 frames
    r = state.lag_correlations(f, p, np.arange(-10, 11))[0]
    assert np.arange(-10, 11)[np.nanargmax(r)] == 5


def test_smoothing_never_bridges_gaps():
    x = np.r_[np.ones(20), np.nan, np.ones(20)]
    smooth, deriv = state.smooth_with_derivative(x, fs=10, window_s=0.5)
    assert np.isnan(smooth[20])
    np.testing.assert_allclose(deriv[np.isfinite(deriv)], 0, atol=1e-12)


def test_running_mask_ignores_short_bouts():
    speed = np.r_[np.zeros(10), np.full(3, 5.0), np.zeros(10), np.full(20, 5.0), np.zeros(5)]
    running = state.running_mask(speed, fs=10, threshold=1, min_bout_s=1.0)
    assert not running[10:13].any() and running[23:43].all()
    on, off = state.state_onsets(running)
    np.testing.assert_array_equal(on, [23])
    np.testing.assert_array_equal(off, [43])


def test_tuning_curve_masks_sparse_bins():
    y = np.r_[np.zeros(50), np.ones(50), 9.0]
    x = np.r_[np.zeros(50), np.ones(50), 0.5]
    df = state.tuning_curve(y, x, np.array([0, 0.4, 0.6, 1.0]), min_samples=30)
    np.testing.assert_allclose(df["mean"], [0, np.nan, 1])


# --- per-baseline analyses on a synthetic session ---------------------------

def test_event_responses_recover_transient_under_every_baseline():
    fs, n = 10.0, 6000
    onsets = np.arange(300, n - 300, 400)
    x = np.zeros(n)
    for o in onsets:
        x[o:o + 20] += 0.2
    F = 1000 * (1 + x)
    spec = recording_comparison.BaselineSpec(30, 20, (10,), (120,))
    variants = recording_comparison.plane_mean_dff(
        [recording_comparison.PlaneTrace(F, np.zeros(n, bool), fs)], spec)
    responses = analyses.event_responses(variants, {"stim": onsets}, fs, 2, 5)
    summary = analyses.event_summary(responses, (0, 1.5))
    np.testing.assert_allclose(summary.centred_response_pp, 20, rtol=0.05)


def test_open_session_loads_exports_behavior_and_tile_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "CACHE", session.Memory(None))
    movie = 1000 + np.random.default_rng(0).normal(0, 10, (600, 16, 16))
    plane = write_plane(tmp_path / "s2p" / "plane0", movie, crop=(0, 16, 0, 16), fs=10.0)
    settings = traces.TraceSettings(tile_size_px=8, baseline_window_s=10)
    traces.extract_tiles(plane, settings, out_root=tmp_path / "exp")
    traces.extract_whole_fov(plane, settings, out_root=tmp_path / "exp")
    stim = np.zeros(600)
    stim[100:110] = 2
    pd.to_pickle({"aligned_data": pd.DataFrame({"Stim": stim, "pupil_diameter": np.ones(600)})},
                 tmp_path / "all_data.pkl")
    rec = session.Recording("r", tmp_path / "exp", tmp_path / "all_data.pkl")
    s = session.open_session(rec, recording_comparison.BaselineSpec(10, 20, (10,), (30,)))
    assert s["n_planes"] == 1 and len(s["tiles"]) == 1
    assert set(s["wf"]) == set(s["tiles"][0]["dff"])
    assert s["tiles"][0]["dff"][next(iter(s["wf"]))].shape == (4, 600)
    np.testing.assert_array_equal(s["behavior"].onsets["Grating 1"], [100])


def test_recordings_config_substitutes_share_root(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"meg1_root": "/share", "recordings": {
        "a": {"export_root": "../out", "suite2p_root": "{meg1}/a/suite2p", "behavior_table": None}}}))
    rec = session.load_recordings(tmp_path / "c.json")["a"]
    assert str(rec.suite2p_root) == "/share/a/suite2p"
    assert rec.export_root == (tmp_path / ".." / "out").resolve()
    assert rec.behavior_table is None


def test_bout_durations_include_edge_bouts():
    mask = np.r_[np.ones(5, bool), np.zeros(5, bool), np.ones(10, bool)]
    np.testing.assert_allclose(state.bout_durations(mask, fs=10), [0.5, 1.0])


# --- regressing out running ------------------------------------------------

def synthetic_running(fs=10.0, n=12000, seed=0):
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    speed = np.clip(gaussian_filter1d(rng.normal(0, 1, n), 30) * 40, 0, None)
    running = state.running_mask(speed, fs, 1.0, 1.0)
    pupil = gaussian_filter1d(rng.normal(0, 1, n), 20) * 10 + 0.5 * speed
    lag = int(2 * fs)  # dF/F follows speed by 2 s
    y = 0.01 * np.r_[np.zeros(lag), speed[:-lag]] + 0.002 * pupil + rng.normal(0, 0.002, n)
    return fs, speed, running, pupil, y


def test_lagged_columns_shift_in_the_stated_direction():
    X = regression.lagged(np.arange(5.0), 1.0, np.array([1, -1]))
    np.testing.assert_array_equal(X[:, 0], [np.nan, 0, 1, 2, 3])
    np.testing.assert_array_equal(X[:, 1], [1, 2, 3, 4, np.nan])


def test_regression_recovers_kernel_peak_and_keeps_mean():
    fs, speed, running, pupil, y = synthetic_running()
    X, lags = regression.running_design(speed, running, fs)
    r = regression.regress_out(y, X)
    assert lags[np.argmax(r["beta"][:-1])] == pytest.approx(2.0)
    assert np.nanmean(r["residual"]) == pytest.approx(np.mean(y[r["rows"]]), rel=1e-6)
    assert regression.blocked_cv_r2(y, X, r["rows"], fs) > 0.5


def test_regressing_out_running_removes_running_driven_pupil_correlation():
    fs, speed, running, pupil, _ = synthetic_running()
    noise = np.random.default_rng(1).normal(0, 0.005, len(speed))
    y = 0.01 * speed + noise  # fluorescence driven only by running; pupil shares running
    before = state.lag_correlations(y, pupil, np.array([0]))[0, 0]
    out = analyses.regress_out_running({"m": y}, speed, running, fs)
    after = state.lag_correlations(out["residual"]["m"], pupil, np.array([0]))[0, 0]
    assert before > 0.3 and abs(after) < 0.05
    assert out["table"].loc["m", "r2_held_out"] > 0.5


def test_tile_regression_matches_per_trace_regression():
    fs, speed, running, pupil, y = synthetic_running(n=3000)
    tiles = np.vstack([y, 2 * y + 0.01])
    out = analyses.regress_out_running({"m": tiles}, speed, running, fs)
    single = analyses.regress_out_running({"m": 2 * y + 0.01}, speed, running, fs)
    np.testing.assert_allclose(out["residual"]["m"][1], single["residual"]["m"], atol=1e-10, equal_nan=True)


# --- regress_running flag ------------------------------------------------------

def running_session(tmp_path, monkeypatch, with_speed=True):
    from scipy.ndimage import gaussian_filter1d
    monkeypatch.setattr(session, "CACHE", session.Memory(None))
    rng = np.random.default_rng(0)
    n, fs = 3000, 10.0
    speed = np.clip(gaussian_filter1d(rng.normal(0, 1, n), 30) * 40, 0, None)
    lag = int(2 * fs)
    drive = 1 + 0.01 * np.r_[np.zeros(lag), speed[:-lag]]
    movie = 1000 * drive[:, None, None] + rng.normal(0, 5, (n, 16, 16))
    plane = write_plane(tmp_path / "s2p" / "plane0", movie, crop=(0, 16, 0, 16), fs=fs)
    settings = traces.TraceSettings(tile_size_px=8, baseline_window_s=30)
    traces.extract_tiles(plane, settings, out_root=tmp_path / "exp")
    traces.extract_whole_fov(plane, settings, out_root=tmp_path / "exp")
    columns = {"pupil_diameter": np.ones(n)}
    if with_speed:
        columns["Speed_Absolute"] = speed
    pd.to_pickle({"aligned_data": pd.DataFrame(columns)}, tmp_path / "all_data.pkl")
    return session.Recording("r", tmp_path / "exp", tmp_path / "all_data.pkl",
                             tmp_path / "s2p", tmp_path / "derived")


SPEC_SMALL = recording_comparison.BaselineSpec(30, 20, (10,), (120,))


def test_regress_running_flag_replaces_dff_and_keeps_originals(tmp_path, monkeypatch):
    rec = running_session(tmp_path, monkeypatch)
    plain = session.open_session(rec, SPEC_SMALL)
    reg = session.open_session(rec, SPEC_SMALL, regress_running=True)
    assert not plain["regress_running"] and reg["regress_running"]
    m = next(iter(reg["wf"]))
    np.testing.assert_array_equal(reg["wf_original"][m], plain["wf"][m])
    assert reg["tiles"][0]["dff"][m].shape == plain["tiles"][0]["dff"][m].shape
    use = np.isfinite(reg["wf"][m]) & np.isfinite(plain["wf"][m])
    speed = reg["behavior"].speed
    before = state.lag_correlations(plain["wf"][m], speed, np.array([20]))[0, 0]
    after = state.lag_correlations(reg["wf"][m], speed, np.array([20]))[0, 0]
    assert before > 0.5 and abs(after) < 0.1
    assert reg["regression"]["table"].loc[m, "r2_held_out"] > 0.3


def test_regress_running_without_speed_fails_loudly(tmp_path, monkeypatch):
    rec = running_session(tmp_path, monkeypatch, with_speed=False)
    with pytest.raises(ValueError, match="no running speed"):
        session.open_session(rec, SPEC_SMALL, regress_running=True)


def test_saving_running_regressed_dff_goes_to_derived_root_not_suite2p(tmp_path, monkeypatch):
    rec = running_session(tmp_path, monkeypatch)
    s = session.open_session(rec, SPEC_SMALL, regress_running=True)
    out = session.save_running_regressed(s)
    assert out == tmp_path / "derived" / "running_regressed"
    with np.load(out / "plane0_tiles_running_regressed_dff.npz") as z:
        assert "current_30_s_rolling_p20" in z.files
    with pytest.raises(ValueError, match="inside the Suite2p folder"):
        session.save_running_regressed(s, tmp_path / "s2p" / "x")
    with pytest.raises(ValueError, match="regress_running=True"):
        session.save_running_regressed(session.open_session(rec, SPEC_SMALL))
