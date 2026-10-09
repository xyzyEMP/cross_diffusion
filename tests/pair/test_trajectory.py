import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.pair.trajectory import Trajectory, extract_local_segments, load_trajectory


class TrajectoryTest(unittest.TestCase):
    def test_load_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sequence_dir = Path(temp_dir) / "ModularNeighborhood" / "Data_diff" / "P1000"
            sequence_dir.mkdir(parents=True)
            poses = np.column_stack(
                (
                    np.arange(5),
                    np.zeros((5, 2)),
                    np.zeros((5, 3)),
                    np.ones(5),
                )
            )
            np.savetxt(sequence_dir / "pose_lcam_front.txt", poses)
            (sequence_dir / "P1000_metadata.json").write_text(
                json.dumps({"time_step": 0.1})
            )

            trajectory = load_trajectory(sequence_dir, "diff")

            self.assertEqual(trajectory.trajectory_id, "P1000")
            self.assertEqual(trajectory.map_id, "ModularNeighborhood")
            self.assertEqual(trajectory.embodiment, "diff")
            np.testing.assert_array_equal(trajectory.positions, poses[:, :3])
            np.testing.assert_array_equal(trajectory.quaternions, poses[:, 3:7])
            np.testing.assert_allclose(trajectory.timestamps, np.arange(5) * 0.1)

            (sequence_dir / "P1000_metadata.json").write_text("{}")
            self.assertIsNone(load_trajectory(sequence_dir, "diff").timestamps)

    def test_extract_local_segments_uses_half_length_stride(self) -> None:
        positions = np.column_stack((np.arange(11), np.zeros((11, 2)))).astype(float)
        trajectory = Trajectory(
            trajectory_id="P2000",
            map_id="ModularNeighborhood",
            embodiment="anymal",
            timestamps=None,
            positions=positions,
            quaternions=np.zeros((11, 4)),
        )

        segments = extract_local_segments(trajectory, segment_length=4.0)

        self.assertEqual(
            [(segment.start_frame, segment.end_frame) for segment in segments],
            [(0, 4), (2, 6), (4, 8), (6, 10)],
        )
        np.testing.assert_allclose(
            [segment.path_length for segment in segments], [4.0] * 4
        )
        np.testing.assert_allclose(
            [segment.center_position[0] for segment in segments], [2.0, 4.0, 6.0, 8.0]
        )
        self.assertEqual(
            segments[0].segment_id, "ModularNeighborhood/anymal/P2000:000000"
        )

    def test_extract_local_segments_interpolates_sparse_frames(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
        trajectory = Trajectory(
            trajectory_id="P0000",
            map_id="ModularNeighborhood",
            embodiment="omni",
            timestamps=None,
            positions=positions,
            quaternions=np.zeros((3, 4)),
        )

        segments = extract_local_segments(trajectory, segment_length=4.0)

        self.assertEqual(len(segments), 4)
        np.testing.assert_allclose(segments[0].points, [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        np.testing.assert_allclose(segments[-1].points, [[6.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        np.testing.assert_allclose([segment.path_length for segment in segments], [4.0] * 4)


if __name__ == "__main__":
    unittest.main()
