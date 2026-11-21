"""Project RGB-D scene and target octrees to a persistent 2D occupancy grid."""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pyoctomap


class GlobalObstacleMap2D:
    """Maintains a continuously updated global 2D obstacle map from octrees."""
    
    def __init__(
        self,
        grid_min: float = -20.0,
        grid_max: float = 20.0,
        grid_resolution: float = 0.2,
        device: torch.device = torch.device("cpu"),
        confidence_threshold: int = 1
    ):
        """Initialize the global obstacle map.
        
        Args:
            grid_min: Minimum grid boundary in meters (both x and y)
            grid_max: Maximum grid boundary in meters (both x and y)
            grid_resolution: Grid cell size in meters
            device: Torch device for computations
            confidence_threshold: Number of observations required to mark cell as occupied
        """
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_resolution = grid_resolution
        self.grid_size = int((grid_max - grid_min) / grid_resolution)
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize occupancy grid
        # Values: 0=unknown, 1-255 = confidence level (higher = more certain)
        self.occupancy_grid = torch.zeros(
            (self.grid_size, self.grid_size),
            device=device,
            dtype=torch.uint8
        )
        
        # Separate tracking for free space and obstacles
        self.free_space_count = torch.zeros(
            (self.grid_size, self.grid_size),
            device=device,
            dtype=torch.int32
        )
        
        self.occupied_count = torch.zeros(
            (self.grid_size, self.grid_size),
            device=device,
            dtype=torch.int32
        )
        
        self.target_count = torch.zeros(
            (self.grid_size, self.grid_size),
            device=device,
            dtype=torch.int32
        )
        
        # Track total number of updates
        self.num_updates = 0
    
    def _update_occupancy_classification(self):
        """Update the occupancy grid based on accumulated counts."""
        # Reset to unknown
        self.occupancy_grid.fill_(0)
        
        # Mark as target (3) first if any target observations (highest priority)
        target_confident = self.target_count >= self.confidence_threshold
        self.occupancy_grid[target_confident] = 3
        
        # Mark as occupied (2) if sufficient occupied observations, but NOT if already target
        occupied_confident = self.occupied_count >= self.confidence_threshold
        not_target = self.occupancy_grid != 3
        self.occupancy_grid[occupied_confident & not_target] = 2
        
        # Mark as free (1) if sufficient free observations and no conflicts
        free_confident = self.free_space_count >= self.confidence_threshold
        no_occupied = self.occupied_count < self.confidence_threshold
        not_target_or_occupied = (self.occupancy_grid != 3) & (self.occupancy_grid != 2)
        self.occupancy_grid[free_confident & no_occupied & not_target_or_occupied] = 1

    def update_based_on_octree(self, octree: 'pyoctomap.OcTree', z_bounds: Optional[Tuple[float, float]] = None):
        """Update the map from an OcTree by projecting occupied/free nodes once.

        Args:
            octree: The OcTree representing the environment.
            z_bounds: Optional (min_z, max_z) to ignore nodes outside a height band.
        """
        if octree is None:
            return
        self.num_updates += 1
        z_min, z_max = z_bounds if z_bounds is not None else (-np.inf, np.inf)
        self._accumulate_octree(octree, target_only=False, z_min=z_min, z_max=z_max)
        self._update_occupancy_classification()

    def update_target_based_on_octree(self, target_octree: 'pyoctomap.OcTree', z_bounds: Optional[Tuple[float, float]] = None):
        """Update the map with a target-only OcTree.

        Args:
            target_octree: The OcTree representing only the target object.
            z_bounds: Optional (min_z, max_z) to ignore nodes outside a height band.
        """
        if target_octree is None:
            return
        self.num_updates += 1
        z_min, z_max = z_bounds if z_bounds is not None else (-np.inf, np.inf)
        self._accumulate_octree(target_octree, target_only=True, z_min=z_min, z_max=z_max)
        self._update_occupancy_classification()

    def _accumulate_octree(self, octree: 'pyoctomap.OcTree', target_only: bool, z_min: float, z_max: float):
        """Project OcTree leaf nodes into the 2D grid in one pass."""
        grid_min = self.grid_min
        grid_size = self.grid_size
        resolution = self.grid_resolution
        occupied_cells: List[Tuple[int, int]] = []
        free_cells: List[Tuple[int, int]] = []
        target_cells: List[Tuple[int, int]] = []

        # NOTE: pyoctomap's leaf iterator yields nodes with coordinates in meters.
        for leaf in octree.begin_leafs():
            coord = leaf.getCoordinate()
            x, y, z = coord[0], coord[1], coord[2]
            if z < z_min or z > z_max:
                continue
            gx = int((x - grid_min) / resolution)
            gy = int((y - grid_min) / resolution)
            if gx < 0 or gx >= grid_size or gy < 0 or gy >= grid_size:
                continue

            if target_only:
                if octree.isNodeOccupied(leaf):
                    target_cells.append((gy, gx))
            else:
                if octree.isNodeOccupied(leaf):
                    occupied_cells.append((gy, gx))
                else:
                    free_cells.append((gy, gx))

        def _bulk_increment(coords: List[Tuple[int, int]], tensor: torch.Tensor):
            if not coords:
                return
            idx = torch.tensor(coords, device=self.device, dtype=torch.long)
            ones = torch.ones(idx.shape[0], device=self.device, dtype=torch.int32)
            tensor.index_put_((idx[:, 0], idx[:, 1]), ones, accumulate=True)

        if target_only:
            _bulk_increment(target_cells, self.target_count)
        else:
            _bulk_increment(occupied_cells, self.occupied_count)
            _bulk_increment(free_cells, self.free_space_count)

    def get_occupancy_grid(self) -> torch.Tensor:
        """Get the current occupancy grid.
        
        Returns:
            Occupancy grid (grid_size, grid_size) with values:
            0=unknown, 1=free, 2=occupied, 3=target_object
        """
        return self.occupancy_grid.clone()
    
    def reset(self):
        """Reset the obstacle map to initial state."""
        self.occupancy_grid.fill_(0)
        self.free_space_count.fill_(0)
        self.occupied_count.fill_(0)
        self.target_count.fill_(0)
        self.num_updates = 0
    
    def _detect_obstacle_clusters(
        self,
        grid_np: np.ndarray,
        min_cluster_size: int = 5,
        max_clusters: int = 8
    ) -> List[dict]:
        """Detect obstacle clusters in the occupancy grid.
        
        Args:
            grid_np: Occupancy grid as numpy array
            min_cluster_size: Minimum number of cells for a valid cluster
            max_clusters: Maximum number of clusters to return
            
        Returns:
            List of cluster dictionaries with keys:
                - 'center_world': (x, y) center in world coordinates
                - 'center_grid': (gx, gy) center in grid coordinates
                - 'size': number of cells in cluster
                - 'bbox': (min_gx, min_gy, max_gx, max_gy) bounding box in grid coords
        """
        # Create binary obstacle mask
        obstacle_mask = (grid_np == 2).astype(np.uint8)
        
        # Connect nearby obstacle points so scattered points within a small radius
        # are merged into a single cluster before labeling.
        # connection_distance_m controls how far apart cells can be to be merged.
        connection_distance_m = 0.6
        kernel_cells = int(np.ceil(connection_distance_m / self.grid_resolution))
        if kernel_cells % 2 == 0:
            kernel_cells += 1
        kernel_cells = max(1, kernel_cells)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_cells, kernel_cells))
        # Close gaps between nearby obstacle pixels (dilate then erode)
        connected_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        # Remove tiny specks (noise) that could create spurious clusters
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, noise_kernel)
        # Find connected components on the processed mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            connected_mask, connectivity=8
        )
        
        clusters = []
        # Skip label 0 (background)
        for label_id in range(1, num_labels):
            size = stats[label_id, cv2.CC_STAT_AREA]
            if size < min_cluster_size:
                continue
            
            # Get bounding box in grid coordinates
            min_gx = stats[label_id, cv2.CC_STAT_LEFT]
            min_gy = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            max_gx = min_gx + width
            max_gy = min_gy + height
            
            # Centroid in grid coordinates (note: centroids are (x, y) = (col, row))
            center_gx = centroids[label_id][0]
            center_gy = centroids[label_id][1]
            
            # Convert to world coordinates
            center_world_x = self.grid_min + center_gx * self.grid_resolution
            center_world_y = self.grid_min + center_gy * self.grid_resolution
            
            clusters.append({
                'center_world': (center_world_x, center_world_y),
                'center_grid': (center_gx, center_gy),
                'size': size,
                'bbox': (min_gx, min_gy, max_gx, max_gy),
                'label_id': label_id
            })
        
        # Sort by size (largest first) and limit to max_clusters
        clusters.sort(key=lambda c: c['size'], reverse=True)
        return clusters[:max_clusters]

    def _compute_candidate_positions(
        self,
        clusters: List[dict],
        grid_np: np.ndarray,
        inflated_mask: np.ndarray,
        candidate_distance: float = 3.0,
        robot_clearance: float = 1.0,
        robot_position: Optional[np.ndarray] = None
    ) -> List[dict]:
        """Compute candidate robot positions for each obstacle cluster.
        
        Each candidate is placed outside the obstacle safe zone, facing the cluster center.
        For each cluster, all valid positions around the cluster are computed, then the
        closest one to the robot is selected.
        
        Args:
            clusters: List of detected obstacle clusters
            grid_np: Occupancy grid as numpy array
            inflated_mask: Inflated obstacle mask (including safety zone)
            candidate_distance: Distance from cluster center to place candidate (meters)
            robot_clearance: Robot clearance radius in meters
            robot_position: Current robot position [x, y, z] for ranking candidates by distance
            
        Returns:
            List of candidate dictionaries with keys:
                - 'position_world': (x, y) position in world coordinates
                - 'yaw': yaw angle in radians (pointing toward cluster)
                - 'cluster_center': (x, y) cluster center in world coordinates
                - 'label': assigned letter (A, B, C, ...)
        """
        candidates = []
        labels = 'ABCDEFGH'
        
        # Get robot position for distance ranking (default to origin if not provided)
        robot_x = robot_position[0] if robot_position is not None else 0.0
        robot_y = robot_position[1] if robot_position is not None else 0.0
        
        for idx, cluster in enumerate(clusters):
            if idx >= 8:
                break
                
            cx, cy = cluster['center_world']
            cluster_label = labels[idx]
            cluster_id = int(cluster.get('label_id', idx))
            
            # Try 8 directions around the cluster to find ALL valid candidate positions
            valid_candidates_for_cluster = []
            angles_to_try = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
            
            for angle in angles_to_try:
                # Candidate position at candidate_distance from cluster center, this results in a circle around the cluster
                cand_x = cx + candidate_distance * np.cos(angle)
                cand_y = cy + candidate_distance * np.sin(angle)
                
                # Check if within grid bounds
                if (cand_x < self.grid_min + robot_clearance or 
                    cand_x > self.grid_max - robot_clearance or
                    cand_y < self.grid_min + robot_clearance or 
                    cand_y > self.grid_max - robot_clearance):
                    continue
                
                # Convert to grid coordinates
                gx = int((cand_x - self.grid_min) / self.grid_resolution)
                gy = int((cand_y - self.grid_min) / self.grid_resolution)
                
                # Check grid bounds
                if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
                    continue
                
                # Check if position is in safe area (not in inflated obstacle zone)
                if inflated_mask[gy, gx] == 1:
                    continue
                
                # Check a small area around the candidate for safety
                safe = True
                check_radius = int(np.ceil(robot_clearance / self.grid_resolution))
                for dy in range(-check_radius, check_radius + 1):
                    for dx in range(-check_radius, check_radius + 1):
                        check_gx = gx + dx
                        check_gy = gy + dy
                        if (0 <= check_gx < self.grid_size and 
                            0 <= check_gy < self.grid_size):
                            if inflated_mask[check_gy, check_gx] == 1:
                                safe = False
                                break
                    if not safe:
                        break
                
                if not safe:
                    continue
                
                # Valid candidate found - compute distance to robot
                dist_to_robot = np.sqrt((cand_x - robot_x)**2 + (cand_y - robot_y)**2)
                
                # Yaw points toward cluster center
                yaw = np.arctan2(cy - cand_y, cx - cand_x)
                
                valid_candidates_for_cluster.append({
                    'position_world': (cand_x, cand_y),
                    'yaw': yaw,
                    'cluster_center': (cx, cy),
                    'cluster_label': cluster_label,
                    'cluster_id': cluster_id,
                    'cluster_size': int(cluster.get('size', 0)),
                    'label': cluster_label,
                    'distance_to_robot': dist_to_robot
                })
            
            # Select the candidate closest to the robot
            if valid_candidates_for_cluster:
                best_candidate = min(valid_candidates_for_cluster, key=lambda c: c['distance_to_robot'])
                candidates.append(best_candidate)

        # Safety: ensure exactly one candidate per cluster (in case upstream logic changes)
        unique_by_cluster = {}
        for cand in candidates:
            key = cand.get('cluster_id', cand.get('cluster_label', cand.get('label')))
            existing = unique_by_cluster.get(key, None)
            if existing is None or cand.get('distance_to_robot', np.inf) < existing.get('distance_to_robot', np.inf):
                unique_by_cluster[key] = cand

        # Keep stable order by cluster label
        unique_candidates = list(unique_by_cluster.values())
        unique_candidates.sort(key=lambda c: str(c.get('cluster_label', c.get('label', ''))))
        return unique_candidates

    def _format_candidates_text(self, candidates: List[dict], robot_position: Optional[np.ndarray] = None, robot_yaw: Optional[float] = None) -> str:
        """Format candidate positions as a text list for prompt generation.
        
        Args:
            candidates: List of candidate dictionaries from _compute_candidate_positions
            robot_position: Current robot position (3,) [x, y, z] for fallback candidate
            robot_yaw: Current robot yaw in degree for fallback candidate
            
        Returns:
            Formatted string with candidate positions, e.g.:
            "Candidate A: x=3.50, y=2.10, yaw=1.57 rad (90°)
             Candidate B: x=-1.20, y=4.30, yaw=-0.79 rad (-45°)
             ..."
        """
        if not candidates:
            # Provide a fallback candidate at the current robot position if available
            if robot_position is not None and robot_yaw is not None:
                x, y = float(robot_position[0]), float(robot_position[1])
                yaw_deg = np.degrees(robot_yaw)
                return f"Candidate A (FALLBACK - stay in place): x={x:.2f}, y={y:.2f}, yaw={robot_yaw:.2f} rad ({yaw_deg:.0f}°)"
            return "No candidate viewpoints available. Return current robot position with confidence 0.0."
        
        lines = []
        for cand in candidates:
            x, y = cand['position_world']
            yaw = cand['yaw']
            label = cand['label']
            yaw_deg = np.degrees(yaw)
            dist = cand.get('distance_to_robot', None)
            dist_str = f"candidate distance to robot={dist:.2f} m" if dist is not None else ""
            cluster_label = cand.get('cluster_label', label)
            cluster_center = cand.get('cluster_center', None)
            if cluster_center is not None:
                ccx, ccy = float(cluster_center[0]), float(cluster_center[1])
                cluster_str = f" (cluster {cluster_label}, center=({ccx:.2f}, {ccy:.2f}))"
            else:
                cluster_str = f" (cluster {cluster_label})"
            lines.append(
                f"Candidate {label}{cluster_str}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad ({yaw_deg:.0f}°){dist_str}"
            )
        
        return "\n".join(lines)
