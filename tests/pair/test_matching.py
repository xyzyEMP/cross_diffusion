import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.pair.matching import MiningConfig, compare_paths, mine_candidate_pairs
from datasets.pair.trajectory import LocalSegment


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _segment(segment_id, embodiment, points, center):
    points = np.asarray(points, dtype=float)
    return LocalSegment(
        segment_id=segment_id,
        trajectory_id=segment_id,
        map_id="ModularNeighborhood",
        embodiment=embodiment,
        start_frame=0,
        end_frame=len(points) - 1,
        points=points,
        entry_position=points[0],
        exit_position=points[-1],
        center_position=np.asarray(center, dtype=float),
        path_length=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
    )


class MatchingTest(unittest.TestCase):
    def setUp(self):
        self.config = MiningConfig.from_mapping(
            {
                "segment_length": 4.0,
                "segment_stride": 2.0,
                "region_search_radius": 1.0,
                "entry_threshold": 0.5,
                "exit_threshold": 0.5,
                "resample_points": 16,
                "mean_path_distance_threshold": 0.1,
                "max_path_distance_threshold": 0.5,
                "path_relation": "similar",
                "visualization_count": 1,
                "random_seed": 0,
            }
        )

    def test_compare_identical_paths(self):
        segment = _segment(
            "a", "anymal", [[0, 0, 0], [0, 0, 0], [2, 0, 0]], [0, 0, 0]
        )
        metrics = compare_paths(segment, segment, self.config)
        self.assertEqual(metrics["mean_correspondence_distance"], 0.0)
        self.assertEqual(metrics["max_correspondence_distance"], 0.0)
        self.assertEqual(metrics["chamfer_distance"], 0.0)

    def test_grid_search_across_cell_boundary(self):
        segment_a = _segment(
            "a", "anymal", [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0.99, 0, 0]
        )
        segment_b = _segment(
            "b", "diff", [[0.05, 0, 0], [1.05, 0, 0], [2.05, 0, 0]], [1.01, 0, 0]
        )

        candidates = mine_candidate_pairs([segment_a], [segment_b], self.config)

        self.assertEqual(len(candidates), 1)
        record = candidates[0].to_record()
        self.assertEqual(record["embodiment_a"], "anymal")
        self.assertEqual(record["embodiment_b"], "diff")

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
    def test_visualization(self):
        from utils.visualization.pair import visualize_pair

        segment_a = _segment(
            "a", "anymal", [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0.99, 0, 0]
        )
        segment_b = _segment(
            "b", "diff", [[0.05, 0, 0], [1.05, 0, 0], [2.05, 0, 0]], [1.01, 0, 0]
        )
        candidate = mine_candidate_pairs([segment_a], [segment_b], self.config)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "pair.png"
            visualize_pair(candidate, output_path)
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
