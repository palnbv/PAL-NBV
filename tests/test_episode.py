from dataclasses import asdict
import unittest

from pallet_nbv.episode import run_episode


class FakeController:
    def __init__(self, *, collision_at=None, stop_at=None, stop_reason=None, fail_at=None):
        self.collision_at = collision_at
        self.stop_at = stop_at
        self.stop_reason = stop_reason
        self.fail_at = fail_at

    def reset(self, scenario):
        self.frame = 0
        return self.frame

    def estimate(self, frame):
        return {"frame": frame, "success": frame != self.fail_at}

    def termination(self):
        return "collision" if self.frame == self.collision_at else None

    def next_action(self, obs, step):
        assert obs == step == self.frame
        return "move", self.stop_reason if step == self.stop_at else None

    def step(self, action):
        assert action == "move"
        self.frame += 1
        return self.frame


class EpisodeTests(unittest.TestCase):
    def test_result_has_exactly_four_fields(self):
        result = run_episode(FakeController(), {}, 2)
        self.assertEqual(asdict(result), {
            "initial_pose_estimation": {"frame": 0, "success": True},
            "final_pose_estimation": {"frame": 2, "success": True},
            "steps": 2,
            "termination_reason": "step_limit",
        })

    def test_collision_at_step_limit_keeps_real_terminal_frame(self):
        result = run_episode(FakeController(collision_at=2, fail_at=2), {}, 2)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.termination_reason, "collision")
        self.assertEqual(result.final_pose_estimation, {"frame": 2, "success": False})

    def test_goal_and_stall_do_not_count_unapplied_actions(self):
        for reason in ("goal_reached", "stalled:no_bbox_detection"):
            with self.subTest(reason=reason):
                result = run_episode(FakeController(stop_at=3, stop_reason=reason), {}, 9)
                self.assertEqual(result.steps, 3)
                self.assertEqual(result.final_pose_estimation["frame"], 3)
                self.assertEqual(result.termination_reason, reason)

    def test_collision_at_reset_without_step(self):
        result = run_episode(FakeController(collision_at=0), {}, 3)
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.initial_pose_estimation, result.final_pose_estimation)
        self.assertEqual(result.termination_reason, "collision")