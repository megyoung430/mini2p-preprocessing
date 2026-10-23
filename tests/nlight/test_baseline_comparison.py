"""The comparison must detect what it claims to: lost slow fluctuations and
time-varying rescaling of transients."""
import numpy as np
import pytest

from nlight import baseline_comparison as bc
from nlight import traces, workflows

FS = 5.0
T = np.arange(int(1800 * FS)) / FS  # 30 min


def transients() -> np.ndarray:
    """Identical 20% transients every 20 s."""
    x = np.zeros_like(T)
    for onset in np.arange(10, T[-1] - 10, 20):
        x += 0.2 * np.exp(-np.clip(T - onset, 0, None) / 1.5) * (T >= onset)
    return x


def run(F: np.ndarray, **kwargs) -> dict:
    bad = np.zeros(len(F), bool)
    baselines = bc.candidate_baselines(F, bad, FS, 60, 20, fixed_percentiles=(10,),
                                       long_windows_s=(300, 600))
    return bc.compare_baselines(F, bad, FS, baselines, slow_timescale_s=10, **kwargs)


def test_reference_row_is_identity():
    result = run(1000 * (1 + transients()))
    ref = result["metrics"].iloc[0]
    assert ref.correlation_with_current == pytest.approx(1)
    assert ref.fast_amplitude_ratio == pytest.approx(1)
    assert ref.median_block_gain == pytest.approx(1)
    assert ref.block_gain_cv == pytest.approx(0, abs=1e-12)


def test_slow_fluctuation_much_longer_than_window_is_kept_by_fixed_baseline_only():
    slow = 0.15 * np.sin(2 * np.pi * T / 600)  # 10-min oscillation, 60 s window
    result = run(1000 * (1 + slow + transients()))
    m = result["metrics"]
    fixed = m.loc["Fixed full-recording p10"]
    # the 60 s rolling baseline follows (and so removes) the oscillation
    assert fixed.slow_amplitude_ratio > 3
    assert m.loc["600 s rolling p20"].slow_amplitude_ratio > m.loc["300 s rolling p20"].slow_amplitude_ratio


def test_bleaching_makes_fixed_baseline_shrink_late_transients():
    bleach = np.exp(-T / 900)  # F falls to ~14% over the recording
    result = run(1000 * bleach * (1 + transients()))
    m = result["metrics"]
    assert m.loc["Fixed full-recording p10", "last_over_first_block_gain"] < 0.3
    # a rolling baseline follows the bleach, so relative amplitudes stay put
    assert m.loc["300 s rolling p20", "last_over_first_block_gain"] == pytest.approx(1, abs=0.15)
    assert result["drift"]["percent_change"] < -75


def test_window_longer_than_recording_is_skipped_with_a_warning():
    F = np.full(int(100 * FS), 1000.0)
    with pytest.warns(UserWarning, match="skipping 600"):
        baselines = bc.candidate_baselines(F, np.zeros(len(F), bool), FS, 60, 20,
                                           long_windows_s=(600,))
    assert list(baselines) == [bc.current_label(60, 20), "Fixed full-recording p10"]


def test_block_gain_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    gains = bc.block_gains(np.concatenate([2 * ref[:500], 3 * ref[500:]]), ref, 2)
    np.testing.assert_allclose(gains, [2, 3])


def test_workflow_reproduces_saved_dff_and_saves_outputs(synthetic_plane, tmp_path):
    settings = traces.TraceSettings(tile_size_px=8, baseline_window_s=20, baseline_percentile=20)
    traces.extract_whole_fov(synthetic_plane, settings, out_root=tmp_path / "exports")
    out = workflows.save_baseline_check(tmp_path / "exports" / "plane0", long_windows_s=(30,), n_blocks=3)
    assert (out / "whole_fov_baseline_check.png").is_file()
    assert (out / "metrics.csv").is_file()
