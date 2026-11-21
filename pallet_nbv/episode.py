"""Episode lifecycle and JSON-compatible output, independent of the simulator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    initial_pose_estimation: dict
    final_pose_estimation: dict
    steps: int
    termination_reason: str


def run_episode(controller, scenario, max_steps: int) -> EpisodeResult:
    """Count applied actions; estimate from the initial and actual terminal observations."""
    obs = controller.reset(scenario)
    initial = controller.estimate(obs)
    steps = 0
    reason = controller.termination()
    while reason is None and steps < max_steps:
        action, reason = controller.next_action(obs, steps)
        if reason is None:
            obs = controller.step(action)
            steps += 1
            reason = controller.termination()
    final = controller.estimate(obs) if steps else initial
    return EpisodeResult(initial, final, steps, reason or "step_limit")