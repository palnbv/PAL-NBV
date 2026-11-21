"""
6DOF Pose Estimation using PnP algorithm for pallet detection.

This module implements a robust pose estimation pipeline using:
1. EPnP algorithm for initial pose estimation
2. RANSAC for outlier rejection
3. Levenberg-Marquardt refinement for accuracy
4. Symmetry handling for ambiguous object orientations
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import logging

# Configure logging (disable verbose output by default)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

@dataclass
class PoseEstimationResult:
    """Result of 6DOF pose estimation."""
    success: bool
    position: np.ndarray  # Translation vector (3,)
    rotation_matrix: np.ndarray  # Rotation matrix (3, 3)
    rotation_vector: np.ndarray  # Rodrigues rotation vector (3,)
    transformation_matrix: np.ndarray  # 4x4 transformation matrix
    reprojection_error: float  # Mean reprojection error in pixels
    num_inliers: int  # Number of RANSAC inliers
    pose_estimation_confidence_score: float  # Overall confidence based on error and inliers
    error_message: str = ""  # Error message if failed
    keypoint_confidences: Optional[np.ndarray] = field(default=None)  # (8,) confidence per keypoint
    
    def constrain_to_floor(self) -> 'PoseEstimationResult':
        """Constrain pose assuming pallet lies flat on floor (XZ plane in camera coords).
        
        Both the pallet keypoint model and the level optical camera use +Y down.
        Preserve the fitted heading about Y and remove pitch and roll.
        
        Returns:
            New PoseEstimationResult with constrained rotation
        """
        if not self.success:
            return self
        
        # Project object's forward direction (Z-axis) onto camera's XZ plane
        obj_z_in_cam = self.rotation_matrix[:, 2]
        yaw = np.arctan2(obj_z_in_cam[0], obj_z_in_cam[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_constrained = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        
        # Update rotation vector
        rvec_constrained, _ = cv2.Rodrigues(R_constrained)
        
        # Build new transformation matrix
        T_constrained = np.eye(4, dtype=np.float32)
        T_constrained[:3, :3] = R_constrained
        T_constrained[:3, 3] = self.position
        
        return PoseEstimationResult(
            success=True,
            position=self.position.copy(),
            rotation_matrix=R_constrained,
            rotation_vector=rvec_constrained.flatten(),
            transformation_matrix=T_constrained,
            reprojection_error=self.reprojection_error,
            num_inliers=self.num_inliers,
            pose_estimation_confidence_score=self.pose_estimation_confidence_score,
            keypoint_confidences=self.keypoint_confidences,
            error_message=""
        )


@dataclass
class PalletDimensions:
    """Configurable pallet dimensions in meters."""
    width: float = 1.0   # X-axis extent
    height: float = 0.2  # Y-axis extent (vertical)
    depth: float = 1.2   # Z-axis extent

class PalletPoseEstimator:
    """6DOF pose estimator for pallet objects using keypoint detection.
    
    Uses PnP (Perspective-n-Point) algorithm with RANSAC for robust estimation
    and handles object symmetries to resolve pose ambiguities.
    """
    
    # RANSAC parameters (class-level for easy tuning)
    RANSAC_ITERATIONS: int = 2000
    RANSAC_REPROJ_THRESHOLD: float = 8.0  # pixels - increased from 2.0 for more tolerance
    RANSAC_CONFIDENCE: float = 0.99
    MIN_INLIERS: int = 4
    
    def __init__(self, camera_config, pallet_dims: Optional[PalletDimensions] = None, 
                 verbose: bool = False):
        """Initialize pose estimator with camera parameters.
        
        Args:
            camera_config: TiledCameraCfg object containing camera parameters
            pallet_dims: Optional pallet dimensions (uses default if None)
            verbose: Enable verbose logging output
        """
        self.camera_config = camera_config
        self.pallet_dims = pallet_dims or PalletDimensions()
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self.K, self.dist_coeffs = self._extract_camera_intrinsics()
        self.object_points_3d = self._define_pallet_keypoints_3d()
        # Pre-compute symmetry rotations once
        self._symmetry_rotations_cache: Optional[List[np.ndarray]] = None
        
    def _extract_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract camera intrinsics from camera configuration."""
        # Extract parameters from PinholeCameraCfg
        focal_length_mm = self.camera_config.spawn.focal_length  # 24.0 mm
        horizontal_aperture_mm = self.camera_config.spawn.horizontal_aperture  # 20.955 mm
        width = self.camera_config.width  # 640 pixels
        height = self.camera_config.height  # 640 pixels
        
        # Calculate focal length in pixels
        # fx = fy = (focal_length_mm / horizontal_aperture_mm) * width
        fx = fy = (focal_length_mm / horizontal_aperture_mm) * width
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        # Camera intrinsic matrix
        K = np.array([[fx,  0, cx],
                      [ 0, fy, cy],
                      [ 0,  0,  1]], dtype=np.float32)
        
        # Assume no distortion for pinhole model (can be refined if needed)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        
        logger.debug(f"Camera intrinsics - fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")
        return K, dist_coeffs
    
    def _define_pallet_keypoints_3d(self) -> np.ndarray:
        """Define 3D keypoints of the pallet in object coordinate system.
        
        Coordinate system: Standard camera coordinates +X=right, +Y=down, +Z=forward
        
        8-keypoint order (no center):
        0: LDB - Left-Down-Back
        1: LUB - Left-Up-Back
        2: RDB - Right-Down-Back
        3: RUB - Right-Up-Back
        4: LDF - Left-Down-Front
        5: LUF - Left-Up-Front
        6: RDF - Right-Down-Front
        7: RUF - Right-Up-Front
        
        Returns:
            np.ndarray: Shape (8, 3) containing 3D keypoint coordinates
        """
        w2 = self.pallet_dims.width / 2.0
        h2 = self.pallet_dims.height / 2.0
        d2 = self.pallet_dims.depth / 2.0
        
        # 8 keypoints in the defined order
        # Down = +Y (bottom), Up = -Y (top)
        # Left = -X, Right = +X
        # Back = +Z, Front = -Z
        obj_pts = np.array([
            [-w2, +h2, +d2],  # 0: LDB - Left-Down-Back
            [-w2, -h2, +d2],  # 1: LUB - Left-Up-Back
            [+w2, +h2, +d2],  # 2: RDB - Right-Down-Back
            [+w2, -h2, +d2],  # 3: RUB - Right-Up-Back
            [-w2, +h2, -d2],  # 4: LDF - Left-Down-Front
            [-w2, -h2, -d2],  # 5: LUF - Left-Up-Front
            [+w2, +h2, -d2],  # 6: RDF - Right-Down-Front
            [+w2, -h2, -d2],  # 7: RUF - Right-Up-Front
        ], dtype=np.float32)
        
        logger.debug(f"3D object points shape: {obj_pts.shape}")
        logger.debug(f"Pallet dimensions: W={self.pallet_dims.width}, H={self.pallet_dims.height}, D={self.pallet_dims.depth}")
        
        return obj_pts

    def _symmetry_rotations_z(self) -> List[np.ndarray]:
        """Generate in-plane Z-axis symmetry rotations for a floor-lying cuboid.
        
        For a pallet lying on the ground (XZ plane), we consider rotations around Y-axis.
        - Always include 0° and 180° (works for rectangular pallets W != D).
        - If the footprint is square (|W - D| ~ 0), also include 90° and 270°.
        
        Returns:
            List of 3x3 rotation matrices representing valid symmetry transforms
        """
        # Use cached result if available
        if self._symmetry_rotations_cache is not None:
            return self._symmetry_rotations_cache
        
        # Compare X (width) and Z (depth) extents for floor-lying object
        x_extent = self.pallet_dims.width
        z_extent = self.pallet_dims.depth
        
        # Always include 0° and 180° rotations around Y-axis
        angles = [0.0, np.pi]
        
        # Add 90° and 270° only if footprint is square
        if abs(x_extent - z_extent) < 1e-4:
            angles.extend([np.pi / 2.0, 3.0 * np.pi / 2.0])
        
        rots: List[np.ndarray] = []
        for a in angles:
            c, s = np.cos(a), np.sin(a)
            # Rotation around Y-axis (vertical axis for floor-lying object)
            rots.append(np.array([[ c, 0.0,  s],
                                  [0.0, 1.0, 0.0],
                                  [-s, 0.0,  c]], dtype=np.float32))
        
        self._symmetry_rotations_cache = rots
        return rots

    def estimate_pose(self, keypoints_2d: np.ndarray, 
                     kpt_confidence_threshold: float = 0.5,
                     consider_symmetries: bool = True,
                     diagnose: bool = False) -> PoseEstimationResult:
        """Estimate 6DOF pose from 2D keypoint detections.
        
        Pipeline:
        1. Filter keypoints by confidence threshold
        2. For each symmetry candidate:
           a. Apply symmetry rotation to 3D model points
           b. Run EPnP + RANSAC for initial robust estimate
           c. Refine on inliers using iterative PnP
           d. Final Levenberg-Marquardt refinement on all points
           e. Validate solution (cheirality check)
        3. Select best solution based on confidence score
        
        Args:
            keypoints_2d: Array of shape (8, 3) containing [x, y, confidence] for each keypoint
            kpt_confidence_threshold: Minimum confidence to use a keypoint (0.0-1.0)
            consider_symmetries: If True, try all symmetry rotations and pick best
            diagnose: If True, run diagnostics on keypoints before estimation
            
        Returns:
            PoseEstimationResult containing pose estimation results
        """
        # Filter keypoints by confidence
        valid_mask = keypoints_2d[:, 2] >= kpt_confidence_threshold
        num_valid = np.sum(valid_mask)
        
        if num_valid < self.MIN_INLIERS:
            return self._create_failed_result(
                f"Not enough confident keypoints: {num_valid} < {self.MIN_INLIERS}",
                keypoints_2d=keypoints_2d
            )
        
        # Extract valid 2D points (in pixels) and corresponding 3D points
        img_pts = keypoints_2d[valid_mask, :2].astype(np.float32)
        base_obj_pts = self.object_points_3d[valid_mask]

        # Prepare symmetry candidates
        candidate_rotations = self._symmetry_rotations_z() if consider_symmetries else [np.eye(3, dtype=np.float32)]

        best_result: Optional[PoseEstimationResult] = None
        best_score: float = -1.0
        total_points = img_pts.shape[0]
        last_error: str = ""
        
        # Also try without RANSAC for cases where all points are good
        pnp_methods = [
            ("RANSAC", True),
            ("DIRECT", False),
        ]

        for R_sym in candidate_rotations:
            # Rotate object keypoints around symmetry axis (object frame)
            obj_pts_rot = (base_obj_pts @ R_sym.T).astype(np.float32)
            
            for method_name, use_ransac in pnp_methods:
                try:
                    if use_ransac:
                        # Stage 1: Robust PnP with RANSAC (initial estimate)
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            obj_pts_rot, img_pts, self.K, self.dist_coeffs,
                            iterationsCount=self.RANSAC_ITERATIONS,
                            reprojectionError=self.RANSAC_REPROJ_THRESHOLD,
                            confidence=self.RANSAC_CONFIDENCE,
                            flags=cv2.SOLVEPNP_EPNP
                        )
                        if not success or inliers is None or len(inliers) < self.MIN_INLIERS:
                            last_error = f"{method_name}: RANSAC failed or insufficient inliers ({0 if inliers is None else len(inliers)})"
                            continue
                        num_inliers = len(inliers)
                    else:
                        # Direct solve without RANSAC - useful when keypoints are all good
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts_rot, img_pts, self.K, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_EPNP
                        )
                        if not success:
                            last_error = f"{method_name}: Direct PnP failed"
                            continue
                        inliers = np.arange(len(img_pts)).reshape(-1, 1)
                        num_inliers = len(img_pts)

                    # Stage 2: Refine on inliers for stable initialization
                    inlier_obj_pts = obj_pts_rot[inliers.flatten()]
                    inlier_img_pts = img_pts[inliers.flatten()]
                    
                    success_refine, rvec, tvec = cv2.solvePnP(
                        inlier_obj_pts, inlier_img_pts, self.K, self.dist_coeffs,
                        rvec, tvec, useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if not success_refine:
                        last_error = f"{method_name}: Iterative refinement failed"
                        continue

                    # Stage 3: Final Levenberg-Marquardt refinement on ALL valid keypoints
                    if hasattr(cv2, "solvePnPRefineLM"):
                        try:
                            rvec, tvec = cv2.solvePnPRefineLM(
                                obj_pts_rot, img_pts, self.K, self.dist_coeffs, rvec, tvec
                            )
                        except cv2.error:
                            pass  # LM refinement failed, continue with current estimate

                    # Convert to matrices
                    R, _ = cv2.Rodrigues(rvec)
                    tvec_flat = tvec.flatten()
                    T_cam_obj = np.eye(4, dtype=np.float32)
                    T_cam_obj[:3, :3] = R
                    T_cam_obj[:3, 3] = tvec_flat

                    # Validation: Cheirality check (all points must be in front of camera)
                    points_cam = (R @ obj_pts_rot.T + tvec_flat.reshape(3, 1)).T
                    if not np.all(points_cam[:, 2] > 0):
                        last_error = f"{method_name}: Cheirality check failed (points behind camera)"
                        continue

                    # Compute quality metrics
                    reprojection_error = self._compute_reprojection_error(
                        obj_pts_rot, img_pts, rvec, tvec_flat
                    )
                    
                    # Score and select best
                    score = self._compute_pose_estimation_confidence_score(
                        reprojection_error, num_inliers, total_points
                    )

                    if score > best_score or (abs(score - best_score) < 1e-6 and 
                                              best_result is not None and 
                                              reprojection_error < best_result.reprojection_error):
                        best_score = score
                        best_result = PoseEstimationResult(
                            success=True,
                            position=tvec_flat.copy(),
                            rotation_matrix=R.copy(),
                            rotation_vector=rvec.flatten().copy(),
                            transformation_matrix=T_cam_obj.copy(),
                            reprojection_error=reprojection_error,
                            num_inliers=num_inliers,
                            pose_estimation_confidence_score=score,
                            keypoint_confidences=keypoints_2d[:, 2].copy()
                        )
                except cv2.error as e:
                    last_error = f"{method_name}: OpenCV error: {e}"
                    continue
                except Exception as e:
                    last_error = f"{method_name}: Unexpected error: {e}"
                    logger.warning(f"PnP estimation failed for symmetry candidate: {e}")
                    continue

        if best_result is None:
            return self._create_failed_result(
                f"PnP failed for all symmetry candidates. Last error: {last_error}",
                keypoints_2d=keypoints_2d
            )

        return best_result

    def _compute_reprojection_error(self, obj_pts: np.ndarray, img_pts: np.ndarray,
                                  rvec: np.ndarray, tvec: np.ndarray) -> float:
        """Compute mean reprojection error in pixels.
        
        Args:
            obj_pts: 3D object points (N, 3)
            img_pts: 2D image points (N, 2)
            rvec: Rotation vector (3,) or (3, 1)
            tvec: Translation vector (3,) or (3, 1)
            
        Returns:
            Mean Euclidean distance between projected and detected 2D points
        """
        projected_pts, _ = cv2.projectPoints(
            obj_pts, rvec.reshape(3, 1), tvec.reshape(3, 1), 
            self.K, self.dist_coeffs
        )
        projected_pts = projected_pts.reshape(-1, 2)
        errors = np.linalg.norm(projected_pts - img_pts, axis=1)
        return float(np.mean(errors))
    
    def _compute_pose_estimation_confidence_score(self, reprojection_error: float, 
                                                num_inliers: int, total_points: int) -> float:
        """Compute overall confidence score [0, 1].
        
        Combines reprojection error quality with inlier ratio using weighted sum.
        
        Args:
            reprojection_error: Mean reprojection error in pixels
            num_inliers: Number of RANSAC inliers
            total_points: Total number of valid keypoints used
            
        Returns:
            Confidence score in [0, 1], higher is better
        """
        # Error component: exponential decay with characteristic scale of 5 pixels
        # error=0 -> 1.0, error=5 -> 0.37, error=10 -> 0.14
        error_score = np.exp(-reprojection_error / 5.0)
        
        # Inlier ratio component
        inlier_ratio = num_inliers / max(total_points, 1)
        
        # Combined confidence (weighted average)
        confidence = 0.7 * error_score + 0.3 * inlier_ratio
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _create_failed_result(self, reason: str, keypoints_2d: Optional[np.ndarray] = None) -> PoseEstimationResult:
        """Create a failed pose estimation result with error message.

        Includes any available keypoint confidence information when provided so
        callers don't need to dereference potentially-None fields.
        """
        logger.debug(f"Pose estimation failed: {reason}")

        # Extract keypoint confidences if keypoints were provided
        keypoint_confidences = None
        try:
            if keypoints_2d is not None:
                # Expect shape (N,3) where column 2 is confidence
                keypoint_confidences = keypoints_2d[:, 2].copy()
        except Exception:
            keypoint_confidences = None

        return PoseEstimationResult(
            success=False,
            position=np.zeros(3, dtype=np.float32),
            rotation_matrix=np.eye(3, dtype=np.float32),
            rotation_vector=np.zeros(3, dtype=np.float32),
            transformation_matrix=np.eye(4, dtype=np.float32),
            reprojection_error=float('inf'),
            num_inliers=0,
            pose_estimation_confidence_score=0.0,
            error_message=reason,
            keypoint_confidences=keypoint_confidences
        )

def convert_keypoints_to_pixels(pose_detections: torch.Tensor, 
                                 num_keypoints: int = 8) -> np.ndarray:
    """Convert normalized keypoint detections to pixel coordinates.
    
    Vectorized implementation for efficiency.
    
    Args:
        pose_detections: Tensor of shape (num_envs, num_detections, D) containing normalized detections.
                        D = 6 (bbox) + num_keypoints * 3 (x, y, conf per keypoint)
        num_keypoints: Number of keypoints to extract (default 8)
        
    Returns:
        Array of shape (num_keypoints, 3) containing [x_pixels, y_pixels, confidence]
    """
   
    # Extract first environment's first detection
    detections = pose_detections[0, 0].cpu().numpy()
    
    # Vectorized extraction: reshape keypoint data
    # Keypoints start at index 6, each has (x, y, conf)
    kp_start = 6
    kp_data = detections[kp_start:kp_start + num_keypoints * 3].reshape(num_keypoints, 3)
    
    # Scale x and y to pixel coordinates
    keypoints = np.empty((num_keypoints, 3), dtype=np.float32)
    keypoints[:, 0] = kp_data[:, 0]   # x
    keypoints[:, 1] = kp_data[:, 1]   # y
    keypoints[:, 2] = kp_data[:, 2]   # confidence (unchanged)
    
    return keypoints


def compute_pose_estimation_metrics(estimated_pose: np.ndarray, ground_truth_pose: np.ndarray,
                                    consider_symmetry: bool = True) -> dict:
    """Compare translation and orientation in the same camera optical frame."""
    est_rotation = estimated_pose[:3, :3]
    truth_rotation = ground_truth_pose[:3, :3]
    equivalent_rotations = [truth_rotation]
    if consider_symmetry:
        # A rectangular pallet is invariant to a half turn about its local vertical (Y).
        equivalent_rotations.append(truth_rotation @ np.diag([-1, 1, -1]))
    cos_angle = max((np.trace(est_rotation.T @ truth) - 1) / 2 for truth in equivalent_rotations)
    return {
        "position_error_m": float(np.linalg.norm(estimated_pose[:3, 3] - ground_truth_pose[:3, 3])),
        "rotation_error_deg": float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))),
    }