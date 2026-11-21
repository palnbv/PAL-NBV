"""
Next-Best-View planning using the 2D obstacle map.

Simplified approach:
  1. Find target cells (value 3) in the occupancy grid.
  2. Use their centroid as the target centre.
  3. Fit a minimum-area rectangle (cv2.minAreaRect) to obtain orientation.
  4. Generate candidate viewpoints on rings inside the preferred zone
     (preferred distance + preferred angle).
  5. Hard-filter candidates: obstacle clearance and clear line of sight.
  6. Score survivors with a simple unweighted sum and pick the best.
"""

import math
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


class NextBestView2D:
    """Computes next-best-view positions based on 2D obstacle map and target detection."""

    def __init__(
        self,
        grid_min: float = -20.0,
        grid_max: float = 20.0,
        grid_resolution: float = 0.2,
        preferred_angle_min_deg: float = 20.0,
        preferred_angle_max_deg: float = 60.0,
        optimal_distance_min: float = 3.0,
        optimal_distance_max: float = 4.5,
        min_obstacle_distance: float = 0.6,
    ):
        self.grid_min = grid_min
        self.grid_resolution = grid_resolution
        self.grid_size = int((grid_max - grid_min) / grid_resolution)

        self.preferred_angle_min_deg = preferred_angle_min_deg
        self.preferred_angle_max_deg = preferred_angle_max_deg
        self.optimal_distance_min = optimal_distance_min
        self.optimal_distance_max = optimal_distance_max
        self.min_obstacle_distance = min_obstacle_distance
        self.camera_fov_rad = np.radians(47.0)


    # ------------------------------------------------------------------
    # Target fitting
    # ------------------------------------------------------------------

    def fit_target_to_cells(
        self, occupancy_grid: torch.Tensor,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[float]]:
        """Fit a minimum-area rectangle to detected target cells.

        Returns:
            (target_map, target_center_grid, target_orientation)
            All ``None`` when fewer than 3 target cells are found.
        """
        grid_np = (
            occupancy_grid.cpu().numpy()
            if hasattr(occupancy_grid, "cpu")
            else np.asarray(occupancy_grid)
        )
        target_mask = grid_np == 3
        target_coords = np.argwhere(target_mask)  # (N, 2) — (row/y, col/x)
        if len(target_coords) < 3:
            return None, None, None

        # Centroid in grid coords
        center_y, center_x = target_coords.mean(axis=0)
        target_center_grid = (int(round(center_x)), int(round(center_y)))

        # Minimum-area rectangle via OpenCV (needs (x, y) points)
        pts_xy = target_coords[:, ::-1].astype(np.float32)  # (N, 2) in (x, y)
        (_cx, _cy), (_w, _h), angle_deg = cv2.minAreaRect(pts_xy)
        target_orientation = np.radians(angle_deg)

        # Build target_map: mark fitted-rectangle cells as value 4
        target_map = grid_np.copy()
        box = cv2.boxPoints(((_cx, _cy), (_w, _h), angle_deg))
        box_int = np.round(box).astype(np.int32)
        rect_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        cv2.fillConvexPoly(rect_mask, box_int, 1)
        target_map[rect_mask == 1] = 4

        return target_map, target_center_grid, target_orientation

    # ------------------------------------------------------------------
    # Candidate generation & scoring
    # ------------------------------------------------------------------

    def compute_candidate_viewpoints(
        self,
        target_map: np.ndarray,
        target_center_grid: Tuple[int, int],
        target_orientation: float,
        current_robot_position_grid: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate and score candidate viewpoints around the target.

        Hard constraints (all must pass):
          * inside grid bounds
          * cell is not occupied / target
          * minimum clearance from obstacles
          * clear line-of-sight to target centre

        Soft scores (simple sum, no weights):
          * *in_preferred_angle* — 1 if inside preferred band, else 0
          * *distance*           — 1 inside optimal band, linear fall-off outside
          * *robot_proximity*    — closer to robot ⇒ higher (0 – 1)
        """
        # --- Pre-compute masks for O(1) hard-filter checks ---
        obstacle_mask = (target_map == 2)
        clearance_dist = distance_transform_edt(~obstacle_mask) * self.grid_resolution
        # Valid cell: not occupied/target AND enough clearance
        occupied_or_target = np.isin(target_map, [2, 3, 4])
        valid_mask = (~occupied_or_target) & (clearance_dist >= self.min_obstacle_distance)

        target_cw = self._grid_to_world(target_center_grid)
        candidates: List[Dict[str, Any]] = []

        num_angles = 72  # every 5°
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        # Sample distances across the preferred band
        d_lo = self.optimal_distance_min
        d_hi = self.optimal_distance_max 
        distances_m = np.linspace(d_lo, d_hi, 7)

        # Pre-compute cos/sin tables for candidate generation
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        tcx, tcy = target_center_grid
        gs = self.grid_size

        # ---- Phase 1: hard filters + soft scores (no ray counting) ----
        for dist_m in distances_m:
            dist_cells = int(dist_m / self.grid_resolution)
            for ai, angle in enumerate(angles):
                cx = tcx + int(dist_cells * cos_angles[ai])
                cy = tcy + int(dist_cells * sin_angles[ai])

                # --- hard constraints (O(1) lookups) ---
                if not (0 <= cx < gs and 0 <= cy < gs):
                    continue
                if not valid_mask[cy, cx]:
                    continue

                if not self._has_clear_los(target_map, (cx, cy), target_center_grid):
                    continue

                # --- soft scores ---
                cand_world = self._grid_to_world((cx, cy))
                yaw_to_target = math.atan2(
                    target_cw[1] - cand_world[1],
                    target_cw[0] - cand_world[0],
                )

                angle_score = self._angle_in_preferred(angle, target_orientation)
                dist_score = self._distance_score(dist_m)
                robot_score = self._robot_proximity_score(
                    (cx, cy), current_robot_position_grid,
                )

                soft_total = angle_score + dist_score + robot_score

                candidates.append({
                    "position_grid": (cx, cy),
                    "position_world": cand_world,
                    "target_robot_yaw": yaw_to_target,
                    "total_score": soft_total,
                })

        # ---- Phase 2: ray-count only the top-K candidates by soft score ----
        RAY_COUNT_TOP_K = 50
        candidates.sort(key=lambda c: -c["total_score"])
        top_k = candidates[:RAY_COUNT_TOP_K]

        for cand in top_k:
            n_obstacle, _, _ = self._count_obstacle_rays(
                target_map, cand["position_grid"], target_center_grid,
            )
            cand["obstacle_rays"] = n_obstacle

        # Primary: fewest obstacle rays.  Secondary: highest soft score.
        top_k.sort(key=lambda c: (c["obstacle_rays"], -c["total_score"]))
        return top_k

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _angle_in_preferred(
        self, viewing_angle: float, target_orientation: float,
    ) -> float:
        """1.0 if the relative viewing angle is inside the preferred band, else 0."""
        rel = abs(viewing_angle - (target_orientation + np.pi))
        rel_deg = np.degrees(rel) % 90  # fold by pallet 4-fold symmetry
        if self.preferred_angle_min_deg <= rel_deg <= self.preferred_angle_max_deg:
            return 1.0
        return 0.0

    def _distance_score(self, dist_m: float) -> float:
        """1.0 inside optimal band, linear fall-off to 0 at ±2 m from the band."""
        if self.optimal_distance_min <= dist_m <= self.optimal_distance_max:
            return 1.0
        if dist_m < self.optimal_distance_min:
            gap = self.optimal_distance_min - dist_m
        else:
            gap = dist_m - self.optimal_distance_max
        return max(0.0, 1.0 - gap / 2.0)

    def _robot_proximity_score(
        self,
        cand_grid: Tuple[int, int],
        robot_grid: Optional[Tuple[int, int]],
    ) -> float:
        """0–1: closer to current robot position is better."""
        if robot_grid is None:
            return 0.5
        d_m = math.hypot(
            cand_grid[0] - robot_grid[0], cand_grid[1] - robot_grid[1],
        ) * self.grid_resolution
        return max(0.0, 1.0 - d_m / 20.0)

    # ------------------------------------------------------------------
    # Obstacle-ray counting (lightweight, used during scoring)
    # ------------------------------------------------------------------

    def _count_obstacle_rays(
        self,
        target_map: np.ndarray,
        viewpoint_grid: Tuple[int, int],
        target_center_grid: Tuple[int, int],
    ) -> Tuple[int, int, int]:
        """Cast FOV rays from *viewpoint_grid* toward the target and count hits.

        Returns:
            (n_obstacle, n_target, n_total) — number of rays that hit an
            obstacle, the target, and total rays cast.
        """
        vx, vy = viewpoint_grid
        dx = target_center_grid[0] - vx
        dy = target_center_grid[1] - vy
        center_angle = math.atan2(dy, dx)
        ray_length = int(math.hypot(dx, dy) * 1.5)
        half_fov = int(np.degrees(self.camera_fov_rad) / 2)
        gs = self.grid_size

        n_obstacle = 0
        n_target = 0
        n_total = 2 * half_fov + 1

        for off in range(-half_fov, half_fov + 1):
            a = center_angle + math.radians(off)
            ex = vx + int(ray_length * math.cos(a))
            ey = vy + int(ray_length * math.sin(a))

            first = True
            for x, y in self._bresenham_gen(vx, vy, ex, ey):
                if first:
                    first = False
                    continue  # skip start cell
                if not (0 <= x < gs and 0 <= y < gs):
                    break
                v = target_map[y, x]
                if v == 2:
                    n_obstacle += 1
                    break
                if v == 3 or v == 4:
                    n_target += 1
                    break

        return n_obstacle, n_target, n_total

    # ------------------------------------------------------------------
    # Hard-constraint helpers
    # ------------------------------------------------------------------

    def _has_clear_los(
        self,
        target_map: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> bool:
        """Bresenham ray-cast: True if no obstacle (value 2) blocks the path."""
        gs = self.grid_size
        first = True
        for x, y in self._bresenham_gen(start[0], start[1], end[0], end[1]):
            if first:
                first = False
                continue  # skip start cell
            if x == end[0] and y == end[1]:
                break  # skip end cell
            if 0 <= x < gs and 0 <= y < gs:
                if target_map[y, x] == 2:
                    return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_best_viewpoint(
        self,
        occupancy_grid: torch.Tensor,
        current_robot_position_world: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the best viewpoint dict, or ``None``."""
        target_map, center_grid, orientation = self.fit_target_to_cells(
            occupancy_grid,
        )
        if target_map is None:
            return None

        robot_grid = None
        if current_robot_position_world is not None:
            robot_grid = self._world_to_grid(current_robot_position_world[:2])

        candidates = self.compute_candidate_viewpoints(
            target_map, center_grid, orientation, robot_grid,
        )
        if not candidates:
            return None

        best = candidates[0]

        return {
            "position_world": best["position_world"],
            "target_robot_yaw": best["target_robot_yaw"],
        }

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        return (
            self.grid_min + (grid_pos[0] + 0.5) * self.grid_resolution,
            self.grid_min + (grid_pos[1] + 0.5) * self.grid_resolution,
        )

    def _world_to_grid(self, world_pos) -> Tuple[int, int]:
        return (
            int((world_pos[0] - self.grid_min) / self.grid_resolution),
            int((world_pos[1] - self.grid_min) / self.grid_resolution),
        )

    # ------------------------------------------------------------------
    # Bresenham line
    # ------------------------------------------------------------------

    @staticmethod
    def _bresenham_gen(
        x0: int, y0: int, x1: int, y1: int,
    ) -> Generator[Tuple[int, int], None, None]:
        """Bresenham line as a generator — no list allocation."""
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            yield (x, y)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
