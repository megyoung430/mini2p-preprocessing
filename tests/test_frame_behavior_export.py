import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from calculate_run_dff import calculate_dff
from align_run_dff_behavior import frame_behavior
from align_run_dff_behavior import confirmed_csv_row
from frame_trial_variables import attach_trial_variables


class FrameExportTests(unittest.TestCase):
    def test_csv_only_confirms_reviewed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'matches.csv'
            pd.DataFrame([dict(session=216,run=1,status='TIMESTAMP_ONLY'),
                          dict(session=216,run=2,status='CONFIRMED',behavior_path='a.tsv',h5='a.h5',imaging_tiff='a.tif')]).to_csv(p,index=False)
            self.assertIsNone(confirmed_csv_row(p,216,1))
            self.assertIsNone(confirmed_csv_row(p,216,3))
            self.assertEqual(confirmed_csv_row(p,216,2)['behavior_path'],'a.tsv')

    def test_trial_intervals_and_outcome_timing(self):
        records=[{'num_t':1,'chosen_tow':'A1'}]
        canonical=dict(trial_info=[records],trial=[1],choice=['A1'],reward_magnitude_by_arm={'A1':[3]})
        raw=pd.DataFrame(dict(time=[0,2,3,5],type=['state','variable','state','info'],
                              subtype=['','print','',''],content=['wait_init',json.dumps(records[0]),'wait_init','end']))
        table=frame_behavior(raw,np.array([np.nan,1.,2.5,3.5]))
        with patch('frame_trial_variables.load_bandit_session',return_value=canonical):
            result,meta=attach_trial_variables(table,raw,Path('a.tsv'),ROOT)
        self.assertEqual(result.loc[1,'choice'],'A1')
        self.assertFalse(result.loc[1,'trial_outcome_logged'])
        self.assertTrue(result.loc[2,'trial_outcome_logged'])
        self.assertEqual(result.loc[2,'reward_magnitude_by_arm.A1'],3)
        self.assertTrue(pd.isna(result.loc[3,'trial']))
        self.assertTrue(pd.isna(result.loc[0,'trial']))

    def test_matches_notebook_including_invalid_baselines(self):
        nb = json.loads((ROOT / 'scripts/check_dff_ses216_run01.ipynb').read_text())
        source = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('def calculate_dff'))
        scope = {'np': np}
        exec(source.split('channels = {}')[0], scope)
        F = np.array([[10,12,20,15], [0,0,0,0], [-1,-2,-3,-4], [1,np.nan,2,3]])
        Fneu = np.ones_like(F)
        actual = calculate_dff(F,Fneu,.7)
        expected = scope['calculate_dff'](F,Fneu,.7)
        for key in ['dff','F0','valid']:
            np.testing.assert_equal(actual[key],expected[key])

    def test_events_count_repeats_and_uncovered_frames_unknown(self):
        raw = pd.DataFrame({'time':[0.0,1.04,1.05,1.8,3.0],
                            'type':['state','event','event','state','event'],
                            'content':['wait','lick','lick','choice','lick']})
        table = frame_behavior(raw,np.array([np.nan,1.,2.,np.nan]))
        self.assertEqual(table.loc[1,'event:lick'],2)
        self.assertEqual(table.loc[2,'event:lick'],0)
        self.assertTrue(np.isnan(table.loc[0,'event:lick']))
        self.assertEqual(table.loc[1,'behavior_state'],'wait')
        self.assertEqual(table.loc[2,'behavior_state'],'choice')
        self.assertEqual(len(table.loc[1,'events']),2)
        self.assertAlmostEqual(table.loc[1,'events'][0]['offset_s'],.04)

if __name__ == '__main__':
    unittest.main()
