import unittest

import numpy as np


from utils.config import Config
from datasets.tartanground.features import build_model_features


class CoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
