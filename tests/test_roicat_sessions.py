"""Small tests for ID preservation and sparse input conversion; no model downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_roicat_sessions import load_footprints
from run_roicat_sessions import export_matches


class TestROICaTInputs(unittest.TestCase):
    def test_noncontiguous_cellpose_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'masks.npy'
            np.save(path, {'masks': np.array([[0, 4, 4], [19, 0, 0]])})
            sf, ids = load_footprints(path, (2, 3))
            np.testing.assert_array_equal(ids, [4, 19])
            np.testing.assert_allclose(sf.sum(axis=1), [[1], [1]])
            np.testing.assert_allclose(sf.toarray()[0], [0, .5, .5, 0, 0, 0])

    def test_suite2p_filter_keeps_original_index_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'stat.npy'
            stat = [dict(ypix=np.array([0]), xpix=np.array([0]), lam=np.array([1.])) for _ in range(3)]
            np.save(path, stat); np.save(path.with_name('iscell.npy'), [[1, 1], [0, 0], [1, 1]])
            sf, ids = load_footprints(path, (2, 3))
            np.testing.assert_array_equal(ids, [0, 2])
            self.assertEqual(sf[0, 0], 1)
            self.assertEqual(sf[1, 0], 1)

    def test_unassigned_are_not_a_cluster(self):
        with tempfile.TemporaryDirectory() as d:
            sessions = [dict(index=i, session=str(i), run='run-01', n_rois=2,
                             roi_ids=[4, 19], roi_id_kind='mask_label') for i in range(2)]
            summary = export_matches([[3, -1], [3, -1]], sessions, Path(d))
            self.assertEqual(summary['n_clusters'], 1)
            self.assertEqual(summary['n_clusters_in_multiple_sessions'], 1)
            self.assertEqual(summary['n_unassigned'], 2)


if __name__ == '__main__':
    unittest.main()
