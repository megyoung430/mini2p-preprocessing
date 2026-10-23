"""The refactored extraction must reproduce the original script's outputs, and
write files the existing downstream scripts can still read."""
import runpy
from pathlib import Path

import numpy as np
import pytest

from nlight import suite2p_io, traces

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
SETTINGS = traces.TraceSettings(tile_size_px=8, batch_frames=128, exclude_below_percentile=10,
                                baseline_window_s=20, baseline_percentile=20)


def original_tile_export(plane_dir: Path) -> dict:
    extract = runpy.run_path(str(SCRIPTS / "extract_big2p_tiles.py"))["extract_plane_tiles"]
    extract(plane_dir, 10.0, SETTINGS.tile_size_px, SETTINGS.batch_frames,
            SETTINGS.exclude_below_percentile, SETTINGS.baseline_window_s,
            SETTINGS.baseline_percentile)
    arrays, meta = suite2p_io.load_tile_export(plane_dir)
    return dict(arrays, meta=meta)


def test_tile_export_reproduces_original_script(synthetic_plane, tmp_path):
    expected = original_tile_export(synthetic_plane)
    traces.extract_tiles(synthetic_plane, SETTINGS)
    actual, meta = suite2p_io.load_tile_export(synthetic_plane)
    assert set(expected) - {"meta"} == set(actual)
    np.testing.assert_array_equal(actual["tile_ids"], expected["tile_ids"])
    np.testing.assert_array_equal(actual["tile_bounds_yxyx"], expected["tile_bounds_yxyx"])
    np.testing.assert_array_equal(actual["badframes"], expected["badframes"])
    for key in ("F", "F0", "dff", "time_s"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=1e-6,
                                   equal_nan=True, err_msg=key)
    for key, value in expected["meta"].items():
        assert meta[key] == value, key


def test_whole_fov_export_is_readable_by_existing_comparison_script(synthetic_plane):
    traces.extract_whole_fov(synthetic_plane, SETTINGS)
    compare = runpy.run_path(str(SCRIPTS / "compare_whole_fov_baselines.py"))["compare_whole_fov"]
    # The original script asserts that its own rolling baseline reproduces our saved dff.
    figures, metrics = compare(synthetic_plane.parent, windows_s=(30,), plane=0)
    assert len(metrics) == 3


def test_whole_fov_raw_trace_is_the_pixel_mean_before_normalisation(synthetic_plane):
    traces.extract_whole_fov(synthetic_plane, SETTINGS)
    arrays, _ = suite2p_io.load_whole_fov_export(synthetic_plane)
    plane = suite2p_io.load_plane(synthetic_plane)
    y0, y1, x0, x1 = plane.crop
    np.testing.assert_allclose(arrays["F"], np.asarray(plane.movie())[:, y0:y1, x0:x1].mean(axis=(1, 2)))
    assert np.isnan(arrays["dff"][plane.bad]).all()


def test_frame_rate_disagreeing_with_ops_fails(synthetic_plane):
    with pytest.raises(ValueError, match="disagrees"):
        traces.extract_tiles(synthetic_plane, SETTINGS, fs=30.0)


def test_invalid_settings_fail_at_construction():
    with pytest.raises(ValueError):
        traces.TraceSettings(baseline_percentile=150)


def test_exports_go_to_out_root_and_leave_suite2p_folder_untouched(synthetic_plane, tmp_path):
    before = sorted(p.name for p in synthetic_plane.iterdir())
    out_root = tmp_path / "exports"
    traces.extract_tiles(synthetic_plane, SETTINGS, out_root=out_root)
    traces.extract_whole_fov(synthetic_plane, SETTINGS, out_root=out_root)
    assert sorted(p.name for p in synthetic_plane.iterdir()) == before
    assert (out_root / "plane0" / "whole_image_roi_dff.npz").is_file()
    assert (out_root / "plane0" / "F.npy").is_file()


def test_refuses_to_overwrite_suite2p_roi_traces(synthetic_plane):
    np.save(synthetic_plane / "stat.npy", np.array([]))
    np.save(synthetic_plane / "F.npy", np.ones((3, 600)))
    with pytest.raises(FileExistsError, match="ROI outputs"):
        traces.extract_tiles(synthetic_plane, SETTINGS)
    np.testing.assert_array_equal(np.load(synthetic_plane / "F.npy"), np.ones((3, 600)))
