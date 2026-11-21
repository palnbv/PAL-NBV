"""RGB-D perception, viewpoint navigation, and pose snapshots."""

import math
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch

from isaaclab.utils.math import convert_camera_frame_orientation_convention, matrix_from_quat

from .navigation import Navigator, wrap_angle
from .obstacle_map_2d import GlobalObstacleMap2D
from .pose_estimation import PalletPoseEstimator, compute_pose_estimation_metrics, convert_keypoints_to_pixels
from .scene_processing import build_scene_octree, compute_camera_pose, create_world_pointcloud, filter_target_from_detection
from .target_accumulator import TargetAccumulator

GRID_MIN, GRID_MAX, RESOLUTION = -20.0, 20.0, 0.1


def _transform(position, orientation):
    transform = np.eye(4)
    transform[:3, :3] = matrix_from_quat(orientation)[0].cpu().numpy()
    transform[:3, 3] = position[0].cpu().numpy()
    return transform


class Controller:
    def __init__(self, env, model_path, bbox_confidence, mode, llm_selector=None):
        self.env = env
        self.device = env.device
        self.bbox_confidence = bbox_confidence
        self.mode = mode
        self.llm_selector = llm_selector
        # Keep the original inference backend (CPU) independent of the simulator GPU.
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.estimator = PalletPoseEstimator(env.rgb.cfg)
        self.target = TargetAccumulator(RESOLUTION, 1.575)
        self.occupancy = GlobalObstacleMap2D(GRID_MIN, GRID_MAX, RESOLUTION, device=self.device, confidence_threshold=1)
        self.nav = Navigator(GRID_MIN, GRID_MAX, RESOLUTION)
        self.hold = torch.zeros((1, 4), device=self.device)

    def reset(self, scenario):
        self.target.reset()
        self.occupancy.reset()
        self.nav.reset()
        self.had_bbox = False
        self.llm_used = False
        self.idle_steps = 0
        self.goal_since = None
        self._detections = None
        return self.env.reset_to_scenario(scenario)

    def step(self, action):
        self._detections = None
        return self.env.step(action)[0]

    def termination(self):
        return "collision" if self.env.collided() else None

    def _detect(self, obs):
        if self._detections is None:
            rgb = obs["camera_rgb"].permute(0, 3, 1, 2).cpu().numpy().astype(np.float32) / 255.0
            self._detections = self.session.run(None, {self.input_name: rgb})[0][:, :1, :]
        return self._detections

    def _camera_pose(self, obs):
        return compute_camera_pose(obs["robot_pos"], obs["robot_quat"], obs["camera_offset_pos"],
                                   obs["camera_offset_rot"], 1)

    def estimate(self, obs):
        detections = self._detect(obs)
        bbox = float(detections[0, 0, 4]) if detections.shape[1] else None
        estimate = {"success": False, "bbox_confidence": bbox, "camera_from_pallet": None,
                "pnp_reprojection_error_px": None, "position_error_m": None, "rotation_error_deg": None}
        if bbox is None or not math.isfinite(bbox):
            estimate["bbox_confidence"] = None
            return estimate
        points = convert_keypoints_to_pixels(torch.as_tensor(detections), num_keypoints=8)
        result = self.estimator.estimate_pose(points, kpt_confidence_threshold=0.7).constrain_to_floor()
        if not result.success:
            return estimate

        pallet = self.env.pallet.data
        center = pallet.root_pos_w.clone()
        center[:, 2] += 0.1  # USD origin is at the pallet base; the estimator uses its center.
        convention = convert_camera_frame_orientation_convention(pallet.root_quat_w, origin="world", target="ros")
        camera = _transform(*self._camera_pose(obs))
        truth = np.linalg.solve(camera, _transform(center, convention))
        matrix = result.transformation_matrix
        if not np.isfinite(matrix).all():
            return estimate
        metrics = compute_pose_estimation_metrics(matrix, truth, consider_symmetry=True)
        estimate.update(success=True, camera_from_pallet=matrix.tolist(),
                        pnp_reprojection_error_px=float(result.reprojection_error),
                        position_error_m=float(metrics["position_error_m"]),
                        rotation_error_deg=float(metrics["rotation_error_deg"]))
        return estimate

    def next_action(self, obs, step):
        cam_pos, cam_quat = self._camera_pose(obs)
        depth, intrinsics = obs["camera_depth"], obs["intrinsic_matrix"]
        cloud = create_world_pointcloud(depth, intrinsics, cam_pos, cam_quat, self.device)
        self.occupancy.update_based_on_octree(build_scene_octree(cloud, RESOLUTION, 2))
        detections = self._detect(obs)
        if detections.shape[1] and np.isfinite(detections[0, 0, :5]).all():
            points, had_bbox = filter_target_from_detection(
                detections, depth, intrinsics, cam_pos, cam_quat, self.bbox_confidence, 1.575, self.device,
            )
            self.had_bbox |= had_bbox
            self.target.accumulate(points)
        if self.target.octree is not None:
            self.occupancy.update_target_based_on_octree(self.target.octree)

        position = obs["robot_pos"][0].cpu().numpy()
        w, x, y, z = obs["robot_quat"][0].cpu().tolist()
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        grid = self.occupancy.get_occupancy_grid()
        previous_goal = self.nav.goal
        self.nav.update(grid, position, yaw, self.target.target_points_for_planning, self.had_bbox,
                        step, geometric=self.mode != "chatgpt")
        if self.mode == "chatgpt" and not self.llm_used:
            self._try_llm(obs, grid, position, yaw)

        if self.nav.goal != previous_goal or not self.nav.at_goal(position, yaw):
            self.goal_since = None
        if self.nav.at_goal(position, yaw):
            if self.goal_since is None:
                self.goal_since = step
            return self.hold, "goal_reached" if step - self.goal_since >= 5 else None

        if not self.nav.commands:
            self.idle_steps += 1
            if self.idle_steps >= 5:
                if self.mode == "hybrid" and not self.llm_used:
                    self._try_llm(obs, grid, position, yaw)
                if not self.nav.commands:
                    return self.hold, f"stalled:{self.nav.failure}"
            else:
                return self.hold, None

        self.idle_steps = 0
        tx, ty, target_yaw = self.nav.commands.popleft()
        dx, dy = tx - position[0], ty - position[1]
        c, s = math.cos(yaw), math.sin(yaw)
        return torch.tensor([[c * dx + s * dy, -s * dx + c * dy, 0.0, wrap_angle(target_yaw - yaw)]],
                            dtype=torch.float32, device=self.device), None

    def _try_llm(self, obs, grid, position, yaw):
        self.llm_used = True
        started = perf_counter() if self.mode == "chatgpt" else None
        try:
            goal = self.llm_selector(obs, self.occupancy, position, yaw)
        finally:
            if started is not None:
                print(f"ChatGPT feedback cycle took {perf_counter() - started:.3f} s", flush=True)
        if goal is None:
            self.nav.failure = "no_llm_viewpoint"
            return
        self.nav.set_goal(goal)
        self.nav.route(grid, position, yaw)