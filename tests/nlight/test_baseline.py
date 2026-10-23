import numpy as np
import pytest

from nlight import baseline


def test_window_is_odd_and_at_least_three():
    assert baseline.window_frames(60, 10) == 601
    assert baseline.window_frames(60, 7.5) == 451
    assert baseline.window_frames(0.01, 10) == 3


@pytest.mark.parametrize("window_s, fs", [(0, 10), (-1, 10), (60, 0), (np.nan, 10)])
def test_window_rejects_invalid(window_s, fs):
    with pytest.raises(ValueError):
        baseline.window_frames(window_s, fs)


def test_rolling_percentile_of_constant_is_constant():
    F = np.full(100, 5.0)
    F0 = baseline.rolling_percentile(F, np.zeros(100, bool), fs=10, window_s=2, percentile=20)
    np.testing.assert_allclose(F0, 5.0)


def test_bad_frames_do_not_enter_the_baseline():
    F = np.full(100, 5.0)
    F[40:60] = 1e6
    bad = np.zeros(100, bool)
    bad[40:60] = True
    F0 = baseline.rolling_percentile(F, bad, fs=10, window_s=3, percentile=100)
    np.testing.assert_allclose(F0[np.isfinite(F0)], 5.0)


def test_rolling_percentile_is_per_trace_for_2d_input():
    F = np.vstack([np.full(50, 1.0), np.full(50, 3.0)])
    F0 = baseline.rolling_percentile(F, np.zeros(50, bool), fs=10, window_s=1, percentile=50)
    assert F0.shape == F.shape
    np.testing.assert_allclose(F0[:, 25], [1.0, 3.0])


def test_fixed_percentile_ignores_bad_frames_and_is_flat():
    F = np.arange(10.0)
    bad = np.zeros(10, bool)
    bad[0] = True  # a bad frame at the minimum must not set the baseline
    F0 = baseline.fixed_percentile(F, bad, 0)
    np.testing.assert_allclose(F0, 1.0)


def test_dff_is_nan_on_bad_frames_and_nonpositive_baseline():
    F = np.array([2.0, 2.0, 2.0, 2.0])
    F0 = np.array([1.0, 0.0, -1.0, 1.0])
    bad = np.array([False, False, False, True])
    np.testing.assert_allclose(baseline.delta_f_over_f(F, F0, bad),
                               [1.0, np.nan, np.nan, np.nan])


def test_all_bad_frames_fail_loudly():
    with pytest.raises(ValueError, match="every frame"):
        baseline.mask_bad_frames(np.ones(5), np.ones(5, bool))


def test_bad_mask_length_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="entries"):
        baseline.delta_f_over_f(np.ones(5), np.ones(5), np.zeros(4, bool))
