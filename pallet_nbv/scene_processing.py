"""
Scene Processing Utilities.

Handles pointcloud processing, octree building, and target object filtering.
"""

from __future__ import annotations

import numpy as np
import torch
import pyoctomap
from typing import Tuple

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils.math import quat_apply, quat_mul

from .target_object_pointcloud_filter import get_target_pointcloud_from_bbox


def compute_camera_pose(
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    camera_offset_pos: torch.Tensor,
    camera_offset_rot: torch.Tensor,
    num_envs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute camera world pose from robot pose and offset.
    
    Args:
        robot_pos: Robot positions (num_envs, 3)
        robot_quat: Robot quaternions (num_envs, 4) in [w,x,y,z]
        camera_offset_pos: Camera offset position (3,)
        camera_offset_rot: Camera offset rotation (4,) in [w,x,y,z]
        num_envs: Number of environments
        
    Returns:
        Tuple of (camera_pos, camera_quat)
    """
    offset_pos = camera_offset_pos.unsqueeze(0).expand(num_envs, -1)
    offset_quat = camera_offset_rot.unsqueeze(0).expand(num_envs, -1)
    
    cam_quat = quat_mul(robot_quat, offset_quat)
    cam_pos = robot_pos + quat_apply(robot_quat, offset_pos)
    
    return cam_pos, cam_quat


def create_world_pointcloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Create world-frame pointcloud from depth image.
    
    Args:
        depth: Depth image (H, W) or (1, H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
        cam_pos: Camera position in world (3,)
        cam_quat: Camera quaternion in world [w,x,y,z] (4,)
        device: Torch device
        
    Returns:
        Pointcloud in world coordinates (N, 3)
    """
    return create_pointcloud_from_depth(
        intrinsic_matrix=intrinsics,
        depth=depth.squeeze(),
        position=cam_pos,
        orientation=cam_quat,
        device=device,
    )


def build_scene_octree(
    pointcloud: torch.Tensor,
    resolution: float,
    subsample_stride: int = 1,
    ground_threshold: float = 0.010,
) -> pyoctomap.OcTree:
    """
    Build octree from pointcloud.
    
    Args:
        pointcloud: World-frame pointcloud (N, 3)
        resolution: Octree resolution
        subsample_stride: Subsample stride for points
        ground_threshold: Z threshold for ground points
        
    Returns:
        OcTree with occupied and free space
    """
    tree = pyoctomap.OcTree(resolution)
    
    points = pointcloud.cpu().numpy()
    if subsample_stride > 1:
        points = points[::subsample_stride]
    
    # Separate ground and occupied points
    occupied = points[points[:, 2] > ground_threshold]
    free = points[points[:, 2] <= ground_threshold]
    
    def _dedup(pts):
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        quantized = np.round(pts / resolution).astype(np.int32)
        unique = np.unique(quantized, axis=0)
        return unique.astype(np.float32) * resolution
    
    for point in _dedup(occupied):
        tree.updateNode(point, True)
    for point in _dedup(free):
        tree.updateNode(point, False)
    
    return tree


def filter_target_from_detection(
    pose_detections: np.ndarray,
    depth_image: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    bbox_confidence_threshold: float,
    depth_consistency_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Return filtered world points and whether the bounding box passed the confidence gate."""
    if pose_detections[0, 0, 4] < bbox_confidence_threshold:
        return torch.empty((0, 3), device=device), False

    points = get_target_pointcloud_from_bbox(
        depth_image=depth_image[0],
        intrinsics=intrinsics[0],
        cam_pos_w=cam_pos[0],
        cam_quat_w=cam_quat[0],
        bbox=torch.as_tensor(pose_detections[0, 0, :4], device=device),
        is_ortho=True,
        z_floor_threshold=0.004,
        depth_consistency_threshold=depth_consistency_threshold,
    )
    
    return points, True
