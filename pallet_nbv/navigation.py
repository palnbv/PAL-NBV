"""Geometric viewpoint selection and collision-checked SE(2) navigation."""

from collections import deque
import math

import numpy as np

from .nbv_ompl import SE2OmplPlanner
from .next_best_view_2d_obstacle_map import NextBestView2D


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Navigator:
    def __init__(self, grid_min, grid_max, resolution):
        self.grid_min = grid_min
        self.resolution = resolution
        self.selector = NextBestView2D(
            grid_min=grid_min, grid_max=grid_max, grid_resolution=resolution,
            preferred_angle_min_deg=25, preferred_angle_max_deg=55,
            optimal_distance_min=3.25, optimal_distance_max=4.25, min_obstacle_distance=0.6,
        )
        self.reset()

    def reset(self):
        self.goal = None
        self.commands = deque()
        self.failure = "no_bbox_detection"
        self._previous = None
        self._planned_points = 0
        self._mask = None
        self._ompl = None
        self._last_replan = -40

    def at_goal(self, position, yaw):
        return (self.goal is not None and np.linalg.norm(position[:2] - self.goal[:2]) < 0.1
                and abs(wrap_angle(yaw - self.goal[2])) < math.radians(2))

    def update(self, grid, position, yaw, target_points, had_bbox, step, *, geometric=True):
        obstacles = (grid == 2).cpu().numpy()
        previous_count = np.count_nonzero(self._previous) if self._previous is not None else 0
        changed = previous_count > 0 and np.count_nonzero(obstacles != self._previous) / previous_count > 0.1
        self._previous = obstacles
        if self.commands and not changed and not self.at_goal(position, yaw) and step - self._last_replan <= 40:
            return

        self._last_replan = step
        if geometric and target_points < 24:
            self.failure = "no_target_octree" if had_bbox else "no_bbox_detection"
        elif geometric and (changed or not self._planned_points
                            or abs(target_points - self._planned_points) / self._planned_points >= 0.1):
            self._planned_points = target_points
            view = self.selector.get_best_viewpoint(grid, current_robot_position_world=tuple(position))
            if view is None:
                self.failure = "no_best_viewpoint"
            else:
                self.set_goal((*view["position_world"][:2], view["target_robot_yaw"]))

        if self.goal is not None and not self.at_goal(position, yaw):
            self.route(grid, position, yaw)

    def set_goal(self, goal):
        self.goal = tuple(float(value) for value in goal)
        self.commands.clear()

    def route(self, grid, position, yaw):
        mask = (grid.cpu().numpy() > 1).astype(np.uint8)
        if not np.array_equal(mask, self._mask):
            self._ompl = SE2OmplPlanner(
                occ_grid=mask, resolution=self.resolution, origin=(self.grid_min, self.grid_min),
                robot_radius=0.8, motion_model="reeds_shepp",
            )
            self._mask = mask
        path = self._ompl.plan_teleport_commands(
            start=(float(position[0]), float(position[1]), yaw), goal=self.goal, timeout=0.5, speedup_factor=10,
        )
        self.commands = deque(path[1:] if path else ())
        self.failure = "empty_teleport_queue" if self.commands else "no_path_to_viewpoint"