import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reward_dff_comparison import frame_indices, peri_event, mean_sem

class RewardComparisonTests(unittest.TestCase):
    def test_excluded_frame_is_not_interpolated_across(self):
        a=peri_event(np.array([[0.,1.,2.,3.]]),np.arange(4.),np.arange(4),
                     np.array([[True,False,True,True]]),np.array([0.]),
                     np.array([.5,1.,1.5,2.,2.5]),1.5)
        np.testing.assert_allclose(a[0,0],[np.nan,np.nan,np.nan,2.,2.5],equal_nan=True)
    def test_removed_frames_stay_missing(self):
        a=peri_event(np.array([[0.,2.,3.]]),np.array([0.,2.,3.]),np.array([0,2,3]),
                     np.ones((1,3),bool),np.array([0.]),np.array([-1.,1.,2.5,4.]),3.)
        np.testing.assert_allclose(a[0,0],[np.nan,np.nan,2.5,np.nan],equal_nan=True)
    def test_selector_validation(self):
        np.testing.assert_equal(frame_indices([True,False,True],3),[0,2])
        for selector in [[1,0],[-1,0],[0,0],[0.,1.],[True]]:
            with self.assertRaises(ValueError): frame_indices(selector,3)
    def test_sem_uses_finite_counts(self):
        mean,sem,count=mean_sem([[1.,np.nan],[3.,2.]])
        np.testing.assert_allclose(mean,[2.,2.])
        np.testing.assert_allclose(sem,[1.,np.nan],equal_nan=True)
        np.testing.assert_equal(count,[2,1])
if __name__=='__main__': unittest.main()
