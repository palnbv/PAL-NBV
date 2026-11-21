"""
Target Object Pointcloud Filter

Filters pointcloud data to extract only points belonging to the target object
based on detected bounding box coordinates and various filtering strategies.
"""

import torch
from isaaclab.utils.math import unproject_depth, transform_points


def get_target_pointcloud_from_bbox(
    depth_image: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_w: torch.Tensor,
    bbox: torch.Tensor,
    is_ortho: bool = True,
    z_floor_threshold: float = 0.008,
    depth_consistency_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Extract target object pointcloud from depth image using 2D bbox.
    
    Args:
        depth_image: (H, W) or (H, W, 1) depth/distance image
        intrinsics: (3, 3) camera intrinsic matrix
        cam_pos_w: (3,) camera position in world frame
        cam_quat_w: (4,) camera orientation in world frame (ROS convention: w,x,y,z)
        bbox: (4,) bounding box [u_min, v_min, u_max, v_max] in pixels
        is_ortho: Whether camera is orthographic
        z_floor_threshold: World z-axis threshold to filter floor points (meters)
        depth_consistency_threshold: Depth variation threshold from median (meters)
    
    Returns:
        points_world: (N, 3) filtered pointcloud in world coordinates
    """
    # 1) Clamp and cast bbox to valid integer pixel indices
    H, W = depth_image.shape[:2]
    u_min, v_min, u_max, v_max = bbox.round().to(torch.int64)
    
    u_min = int(torch.clamp(u_min, 0, W - 1))
    u_max = int(torch.clamp(u_max, 0, W - 1))
    v_min = int(torch.clamp(v_min, 0, H - 1))
    v_max = int(torch.clamp(v_max, 0, H - 1))
    
    if u_max <= u_min or v_max <= v_min:
        # Invalid bbox -> empty cloud
        return depth_image.new_zeros((0, 3))
    
    # 2) Extract depth ROI for that bbox
    depth_roi = depth_image[v_min:v_max, u_min:u_max]  # shape (h, w) or (h, w, 1)

    # Adjust Intrinsics for ROI
    K_full = intrinsics.clone()
    K_crop = K_full.clone()
    K_crop[0, 2] = K_full[0, 2] - float(u_min)  # cx'
    K_crop[1, 2] = K_full[1, 2] - float(v_min)  # cy'
    
    # 3) Unproject ROI depth to camera frame
    points_cam = unproject_depth(depth=depth_roi,intrinsics=K_crop,is_ortho=is_ortho)  # (P, 3) in camera frame, flattened ROI
    
    # Filter out invalid points (depth <= 0)
    valid_mask = (points_cam[:, 2] > 0) & torch.isfinite(points_cam).all(dim=1)
    points_cam = points_cam[valid_mask]
    
    if points_cam.shape[0] == 0:
        return depth_image.new_zeros((0, 3))
    
    # 4) Transform to world frame
    points_world = transform_points(
        points_cam,
        pos=cam_pos_w,
        quat=cam_quat_w,
    )  # (P, 3)
    
    # 5) Filter floor points (world z-axis)
    floor_mask = points_world[:, 2] > z_floor_threshold
    points_world = points_world[floor_mask]
    if points_world.shape[0] == 0:
        return depth_image.new_zeros((0, 3))
    
    # 6) Depth consistency filter (in camera frame is better for this)
    # Use camera frame Z for depth filtering
    points_cam = points_cam[floor_mask]  # Apply same mask to camera frame points
    depths = points_cam[:, 2]
    median_depth = torch.median(depths)
    depth_mask = torch.abs(depths - median_depth) < depth_consistency_threshold
    return points_world[depth_mask]
