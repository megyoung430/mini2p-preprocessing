import numpy as np
import pytest

from nlight import tiles


def test_grid_drops_partial_edge_tiles_and_is_row_major():
    grid = tiles.tile_grid((2, 30, 3, 37), 8)  # 28 x 34 px -> 3 x 4 complete tiles
    assert grid.shape == (3, 4)
    np.testing.assert_array_equal(grid.bounds[0], [2, 10, 3, 11])
    np.testing.assert_array_equal(grid.bounds[1], [2, 10, 11, 19])  # next column first
    np.testing.assert_array_equal(grid.bounds[4], [10, 18, 3, 11])


def test_grid_without_a_complete_tile_fails():
    with pytest.raises(ValueError, match="no complete"):
        tiles.tile_grid((0, 5, 0, 5), 8)


def test_tile_means_match_explicit_pixel_averages_across_batches():
    rng = np.random.default_rng(1)
    movie = rng.integers(0, 1000, (7, 20, 20)).astype(np.int16)
    grid = tiles.tile_grid((1, 19, 2, 18), 4)
    expected = np.array([[movie[f, y0:y1, x0:x1].mean() for f in range(7)]
                         for y0, y1, x0, x1 in grid.bounds])
    np.testing.assert_allclose(tiles.tile_means(movie, grid, batch_frames=3), expected, rtol=1e-6)


def test_whole_fov_mean_averages_raw_pixels_in_the_crop_only():
    movie = np.zeros((3, 10, 10))
    movie[:, 2:5, 2:6] = np.array([1.0, 2.0, 3.0])[:, None, None]
    movie[:, 0, 0] = 1e6  # outside the crop
    np.testing.assert_allclose(tiles.whole_fov_mean(movie, (2, 5, 2, 6), batch_frames=2),
                               [1.0, 2.0, 3.0])


def test_brightness_selection_keeps_ties_and_ignores_bad_frames():
    F = np.array([[1.0, 1.0, 100.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]])
    bad = np.array([False, False, True])
    keep, brightness, cutoff = tiles.brightness_selection(F, bad, 40)
    np.testing.assert_allclose(brightness, [1, 2, 2, 3])
    assert cutoff == 2.0
    np.testing.assert_array_equal(keep, [False, True, True, True])


def test_zero_percentile_keeps_every_tile():
    keep, _, _ = tiles.brightness_selection(np.array([[1.0], [5.0]]), np.zeros(1, bool), 0)
    assert keep.all()
