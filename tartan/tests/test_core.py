import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


from tartan.data.pose_utils import to_local_se2
from tartan.data.pose_utils import discover_trajectories


class CoreTests(unittest.TestCase):

    def test_discover_trajectories_accepts_dataset_or_environment_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "Env" / "Data_anymal" / "P1" / "pose_lcam_front.txt"
            pose.parent.mkdir(parents=True)
            pose.write_text("", encoding="utf-8")
            from_dataset = discover_trajectories(root)
            from_environment = discover_trajectories(root / "Env")
            self.assertEqual([record.key for record in from_dataset], ["Env_anymal_P1"])
            self.assertEqual([record.key for record in from_environment], ["Env_anymal_P1"])


    def test_local_transform(self):
        anchor = np.array([10.0, 20.0, np.pi / 2])
        poses = np.array([[10.0, 21.0, np.pi / 2], [9.0, 20.0, np.pi]])
        local = to_local_se2(poses, anchor)
        np.testing.assert_allclose(local[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(local[1], [0.0, 1.0, np.pi / 2], atol=1e-6)





if __name__ == "__main__":
    unittest.main()


def test_proxy_voxel_conversion_uses_z_and_body_origin():
    from scipy.spatial.transform import Rotation
    from tartan.data.pose_utils import proxy_sparse_to_body
    # Pure geometry, never a training fixture: pitch maps camera Z to world X.
    camera = np.r_[10.,20.,0.,Rotation.from_euler('y',90,degrees=True).as_quat()]
    geometry = {'source_frame':'camera_quaternion_local','bounds_m':[-.5,10,-.5,10,-.5,10],
                'resolution_m':1.,'camera_pose_world':camera,'world_frame_policy':'ned_to_nwu'}
    actual = proxy_sparse_to_body(np.array([[0,2,4,3]]),geometry,[11.,-20.,0.])
    np.testing.assert_array_equal(actual,[[56,46,0,3]])


def test_proxy_empty_body_grid_retains_explicit_extent():
    from tartan.research_score.evaluation.shortest_path import sparse_grid
    assert sparse_grid(np.empty((0,4),int),101).shape == (101,101)


def test_proxy_body_se3_rotates_translation_before_ned_conversion(monkeypatch):
    from scipy.spatial.transform import Rotation
    from tartan.data import pose_utils
    q = Rotation.from_euler('z',90,degrees=True).as_quat()
    monkeypatch.setattr(pose_utils,'load_poses',lambda _:np.array([np.r_[10.,20.,0.,q]]))
    row={'pose_path':'unused_math_only','world_frame_policy':'ned_to_nwu',
         'camera_to_body_se3':[1.,0.,0.,0.,0.,0.,1.]}
    np.testing.assert_allclose(pose_utils.load_proxy_se2(row),[[10.,-21.,-np.pi/2]],atol=1e-6)


def test_observed_reference_uses_real_pose_and_rejects_body_extrinsic(monkeypatch):
    from tartan.data import pose_utils
    import pytest
    monkeypatch.setattr(pose_utils,'load_poses',lambda _:np.array([[10.,20.,3.,0.,0.,0.,1.]]))
    row={'pose_path':'math_only','world_frame_policy':'ned_to_nwu','reference_pose_policy':'observed_lcam_front_reference'}
    np.testing.assert_allclose(pose_utils.load_proxy_se2(row),[[10.,-20.,0.]])
    row['camera_to_body_se3']=[0.,0.,0.,0.,0.,0.,1.]
    with pytest.raises(ValueError,match='must not apply'):pose_utils.load_proxy_se2(row)
