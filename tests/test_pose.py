import unittest

import numpy as np

from pallet_nbv.pose_estimation import PoseEstimationResult, compute_pose_estimation_metrics


class PoseTests(unittest.TestCase):
    def test_only_vertical_half_turn_counts_as_pallet_symmetry(self):
        truth = np.eye(4)
        for rotation, expected in (([-1, 1, -1], 0.0), ([1, -1, -1], 180.0)):
            with self.subTest(rotation=rotation):
                pose = np.diag([*rotation, 1]).astype(float)
                pose[0, 3] = 0.25
                metrics = compute_pose_estimation_metrics(pose, truth)
                self.assertAlmostEqual(metrics["position_error_m"], 0.25)
                self.assertAlmostEqual(metrics["rotation_error_deg"], expected)

    def test_floor_constraint_preserves_heading_and_down_axis(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        pose[:3, 3] = [0.2, 0.3, 3.0]
        result = PoseEstimationResult(True, pose[:3, 3], pose[:3, :3], np.zeros(3), pose, 0.0, 8, 1.0)
        np.testing.assert_allclose(result.constrain_to_floor().transformation_matrix, pose, atol=1e-6)