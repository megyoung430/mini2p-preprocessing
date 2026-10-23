import numpy as np
import pytest

from nlight import recording_comparison as rc
from nlight import traces, workflows
from conftest import write_plane

SPEC = rc.BaselineSpec(current_window_s=30, current_percentile=20, fixed_percentiles=(10,),
                       long_windows_s=(120,))


def trace(fs: float, minutes: float, amplitude: float, brightness: float = 1000.0,
          seed: int = 0) -> rc.PlaneTrace:
    """Constant baseline plus identical transients of ``amplitude`` every 15 s."""
    t = np.arange(int(minutes * 60 * fs)) / fs
    x = np.zeros_like(t)
    for onset in np.arange(5, t[-1] - 5, 15):
        x += amplitude * np.exp(-np.clip(t - onset, 0, None) / 1.0) * (t >= onset)
    noise = np.random.default_rng(seed).normal(0, 0.002, len(t))
    return rc.PlaneTrace(brightness * (1 + x + noise), np.zeros(len(t), bool), fs)


def test_plane_mean_weights_planes_equally_regardless_of_brightness():
    dim, bright = trace(7.5, 6, 0.1, 100.0), trace(7.5, 6, 0.3, 10000.0)
    mean = rc.plane_mean_dff([dim, bright], SPEC)
    alone = {k: (rc.plane_mean_dff([dim], SPEC)[k] + rc.plane_mean_dff([bright], SPEC)[k]) / 2
             for k in mean}
    for k in mean:
        np.testing.assert_allclose(mean[k], alone[k], equal_nan=True)


def test_frame_invalid_in_any_plane_is_invalid_in_the_mean():
    a, b = trace(7.5, 6, 0.1), trace(7.5, 6, 0.1, seed=1)
    bad = b.bad.copy()
    bad[100] = True
    mean = rc.plane_mean_dff([a, rc.PlaneTrace(b.F, bad, b.fs)], SPEC)
    assert all(np.isnan(v[100]) for v in mean.values())


def test_mismatched_planes_fail_loudly():
    with pytest.raises(ValueError, match="frame counts"):
        rc.plane_mean_dff([trace(7.5, 6, 0.1), trace(7.5, 5, 0.1)], SPEC)


def test_larger_transients_rank_higher_under_every_method_across_frame_rates():
    summaries = {"big, 30 Hz, 1 plane": rc.summarize_recording([trace(30, 6, 0.3)], SPEC),
                 "small, 7.5 Hz, 4 planes": rc.summarize_recording(
                     [trace(7.5, 6, 0.1, seed=s) for s in range(4)], SPEC)}
    table = rc.comparison_table(summaries)
    diff = rc.paired_difference(table, *summaries)
    assert (diff["big, 30 Hz, 1 plane − small, 7.5 Hz, 4 planes", "fast_std_pp"] > 0).all()
    ratio = diff["big, 30 Hz, 1 plane / small, 7.5 Hz, 4 planes", "fast_std_pp"]
    np.testing.assert_allclose(ratio, 3, rtol=0.15)


def test_bin_trace_averages_and_masks_sparse_bins():
    t = np.arange(8) / 4.0  # 4 Hz, two 1-s bins
    x = np.array([1, 1, 3, 3, np.nan, np.nan, np.nan, 5.0])
    centers, means = rc.bin_trace(t, x, 1.0)
    np.testing.assert_allclose(centers, [0.5, 1.5])
    np.testing.assert_allclose(means, [2.0, np.nan])


def test_comparison_workflow_reads_exports_and_saves(tmp_path):
    rng = np.random.default_rng(0)
    roots = {}
    for name, n_planes in [("one", 1), ("four", 4)]:
        for plane in range(n_planes):
            movie = 1000 + rng.normal(0, 10, (3000, 16, 16))
            plane_dir = write_plane(tmp_path / name / "suite2p" / f"plane{plane}", movie,
                                    crop=(0, 16, 0, 16), fs=10.0)
            traces.extract_whole_fov(plane_dir, traces.TraceSettings(),
                                     out_root=tmp_path / name / "exports")
        roots[name] = tmp_path / name / "exports"
    out = workflows.save_recording_comparison(roots, tmp_path / "cmp", spec=SPEC)
    for f in ("recording_comparison.png", "recording_comparison.csv", "paired_difference.csv"):
        assert (out / f).is_file()
