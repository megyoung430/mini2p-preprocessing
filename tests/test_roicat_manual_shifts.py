"""Checks for manual translations and compatibility with prepared input tracking."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from manual_shift_roicat_inputs_meg import shift_record, apply_shifts, load_inputs, resolve_record
from prepare_roicat_sessions import prepare
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class TestShifts(unittest.TestCase):
    def test_translation_and_clipping(self):
        image = np.arange(20, dtype=np.float32).reshape(4, 5)
        sf = sparse.csr_matrix(([.5, .5], ([0, 0], [8, 9])), shape=(1, 20))
        im, shifted, clipped = shift_record(image, sf, 1, 1)
        self.assertEqual(clipped, 1)
        self.assertEqual(shifted[0, 14], 1)
        self.assertEqual(im[2, 4], image[1, 3])
        with self.assertRaisesRegex(ValueError, 'entire ROI'):
            shift_record(image, sf, 5, 0)
        _, negative, _ = shift_record(image, sf, -1, -1)
        np.testing.assert_array_equal(negative.indices, [2, 3])

    def test_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for run in ['run-01', 'run-02']:
                plane = root / 'ses-test' / 'funcimg' / run / 'suite2p' / 'plane0'
                plane.mkdir(parents=True)
                Image.fromarray(np.arange(36, dtype=np.float32).reshape(6, 6)).save(plane / 'mean.tif')
                masks = np.zeros((6, 6), int)
                masks[2, 2] = 4
                masks[3, 3] = 19
                np.save(plane / 'seg.npy', {'masks': masks})
                entries.append(dict(session='ses-test', run=run))
            config = dict(root=str(root), sessions=entries, image_file='mean.tif', roi_file='seg.npy')
            prepared, shifted = root / 'prepared', root / 'shifted'
            prepare(config, prepared)
            manifest, images, footprints = load_inputs(prepared)
            with self.assertRaisesRegex(ValueError, 'expected one'):
                resolve_record(manifest, 'ses-test')
            table = root / 'shifts.csv'
            table.write_text('session,run,dx,dy\nses-test,run-02,1,-1\n')
            apply_shifts(prepared, shifted, table)
            result, new_images, new_sf = load_inputs(shifted)
            self.assertEqual(result['sessions'][1]['roi_ids'], [4, 19])
            np.testing.assert_array_equal(new_images[0], images[0])
            np.testing.assert_array_equal(new_sf[0].toarray(), footprints[0].toarray())
            np.testing.assert_array_equal(load_inputs(prepared)[2][1].toarray(), footprints[1].toarray())
            self.assertEqual(new_sf[1][0, 9], 1)
            with self.assertRaises(ValueError):
                apply_shifts(prepared, shifted, table)
            table.write_text('session,run,dx,dy\nses-test,run-02,1,-1\nses-test,run-02,0,0\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                apply_shifts(prepared, root / 'invalid', table)
            self.assertFalse((root / 'invalid').exists())
            subprocess.run([sys.executable, str(SCRIPTS / 'preview_roicat_mask_shift.py'),
                '--input-root', str(prepared), '--reference-session', 'ses-test', '--reference-run', 'run-01',
                '--session', 'ses-test', '--run', 'run-02', '--dx', '1', '--dy', '-1',
                '--out', str(root / 'preview.png')], check=True)
            self.assertGreater((root / 'preview.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
