import unittest

import numpy as np

from tartan.planning.adaptation import adapt_trajectory

from diffusion_planner.utils.config import Config
from tartan.data.features import build_model_features
from tartan.evaluation.metrics import trajectory_metrics
from tartan.data.pose_utils import to_local_se2
from tartan.data.terrain import ElevationMap, terrain_feasibility_metrics


class CoreTests(unittest.TestCase):
    def test_adapter_enforces_continuous_robot_speed(self):
        history = np.zeros((21, 3), dtype=np.float32)
        history[:, 0] = np.arange(-20, 1) * 0.1
        raw = np.zeros((80, 3), dtype=np.float32)
        raw[:, 0] = np.linspace(1.0, 80.0, 80)
        adapted = adapt_trajectory(raw, history, "anymal", 0.1)
        speed = np.linalg.norm(np.diff(np.vstack(([0, 0], adapted[:, :2])), axis=0), axis=1) / 0.1
        self.assertLessEqual(float(speed.max()), 1.001)
        self.assertAlmostEqual(float(adapted[0, 0]), 0.1, places=4)

    def test_local_transform(self):
        anchor = np.array([10.0, 20.0, np.pi / 2])
        poses = np.array([[10.0, 21.0, np.pi / 2], [9.0, 20.0, np.pi]])
        local = to_local_se2(poses, anchor)
        np.testing.assert_allclose(local[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(local[1], [0.0, 1.0, np.pi / 2], atol=1e-6)

    def test_feature_shapes_and_finiteness(self):
        config = Config("checkpoints/args.json", guidance_fn=None)
        gt = np.zeros((80, 3), dtype=np.float32)
        gt[:, 0] = np.linspace(0.1, 8.0, 80)
        features = build_model_features(config, gt, "diff")
        self.assertEqual(tuple(features["neighbor_agents_past"].shape), (32, 21, 11))
        self.assertEqual(tuple(features["lanes"].shape), (70, 20, 12))
        self.assertEqual(tuple(features["route_lanes"].shape), (25, 20, 12))
        for tensor in features.values():
            self.assertTrue(tensor.isfinite().all())

    def test_perfect_prediction_metrics(self):
        gt = np.zeros((80, 3), dtype=np.float32)
        gt[:, 0] = np.linspace(0.1, 8.0, 80)
        metrics = trajectory_metrics(gt, gt, "diff", 0.1)
        self.assertAlmostEqual(metrics["ade_m"], 0.0)
        self.assertAlmostEqual(metrics["fde_m"], 0.0)
        self.assertAlmostEqual(metrics["route_deviation_mean_m"], 0.0, places=5)
        self.assertAlmostEqual(metrics["goal_progress_ratio"], 1.0, places=5)

    def test_flat_terrain_is_feasible(self):
        terrain = ElevationMap(
            elevation=np.zeros((100, 100), dtype=np.float32),
            origin_xy=np.array([-25.0, -25.0]),
            resolution=0.5,
            environment="synthetic",
        )
        trajectory = np.zeros((80, 3), dtype=np.float32)
        trajectory[:, 0] = np.linspace(0.1, 8.0, 80)
        metrics, elevation = terrain_feasibility_metrics(
            trajectory, np.array([0.0, 0.0, 0.0]), terrain, "anymal"
        )
        self.assertTrue(np.isfinite(elevation).all())
        self.assertAlmostEqual(metrics["terrain_coverage_rate"], 1.0)
        self.assertAlmostEqual(metrics["support_failure_rate"], 0.0)
        self.assertAlmostEqual(metrics["geometric_contact_feasibility_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
