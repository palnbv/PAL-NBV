# Below is a self-contained OMPL SE(2) planner that:

# Plans in SE(2) using OMPL (RRT* by default).

# Extracts the continuous path.

# Densifies it into teleport commands: a list of small pose steps (x, y, yaw) where each step respects:

# max_step_trans (max linear movement per teleport)

# max_step_rot (max yaw change per teleport)

# You can feed those poses directly into your sim and just set the robot pose each step.

# Full code: SE(2) OMPL planner → teleport pose sequence
import math
from typing import Callable, List, Tuple, Literal

import numpy as np
from scipy.ndimage import binary_dilation
from ompl import base as ob
from ompl import geometric as og

MotionModel = Literal["se2", "reeds_shepp", "r2"]


class SE2OmplPlanner:
    """
    2D SE(2) path planner using OMPL.

    - State: (x, y, yaw) or (x, y) for R2 model
    - Obstacle map: 2D occupancy grid (0 = free, >0 = obstacle)
    - World frame: [origin_x, origin_x + W*res] x [origin_y, origin_y + H*res]
    
    Motion models:
        - "se2": Unconstrained SE(2), holonomic robot (recommended for omnidirectional)
        - "reeds_shepp": Car-like with reversing (smoother than Dubins)
        - "r2": Position-only planning, rotation interpolated separately (simplest paths)
    """

    def __init__(
        self,
        occ_grid: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        robot_radius: float = 0.0,
        planner_factory: Callable[[ob.SpaceInformation], ob.Planner] = None,
        motion_model: MotionModel = "se2",
    ):
        """
        occ_grid: HxW uint8/float array, 0 = free, >0 = obstacle
        resolution: meters per cell
        origin: (x0, y0) world position of cell (0, 0)
        robot_radius: inflate obstacles by this radius (in meters)
        motion_model: "se2" | "reeds_shepp" | "r2"
        """
        assert occ_grid.ndim == 2, "occ_grid must be 2D"

        self.res = float(resolution)
        self.origin_x, self.origin_y = origin
        self.robot_radius = float(robot_radius)
        self.motion_model = motion_model

        self.height, self.width = occ_grid.shape

        # Pre-inflate obstacles by robot radius so is_state_valid is O(1)
        raw = occ_grid.astype(np.uint8)
        if robot_radius > 0.0:
            rad_cells = int(robot_radius / resolution)
            struct = np.ones((2 * rad_cells + 1, 2 * rad_cells + 1), dtype=bool)
            inflated_mask = binary_dilation(raw > 0, structure=struct)
            raw = np.where(inflated_mask, np.uint8(1), np.uint8(0))
        self.occ_grid = raw

        # --- 1) State space based on motion model ---
        if motion_model == "r2":
            # Position-only planning (2D)
            space = ob.RealVectorStateSpace(2)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(0, self.origin_x)
            bounds.setLow(1, self.origin_y)
            bounds.setHigh(0, self.origin_x + self.width * self.res)
            bounds.setHigh(1, self.origin_y + self.height * self.res)
            space.setBounds(bounds)
        elif motion_model == "reeds_shepp":
            space = ob.ReedsSheppStateSpace(turningRadius=0.3)
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(0, self.origin_x)
            bounds.setLow(1, self.origin_y)
            bounds.setHigh(0, self.origin_x + self.width * self.res)
            bounds.setHigh(1, self.origin_y + self.height * self.res)
            space.setBounds(bounds)
        else:  # "se2" - unconstrained SE(2)
            space = ob.SE2StateSpace()
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(0, self.origin_x)
            bounds.setLow(1, self.origin_y)
            bounds.setHigh(0, self.origin_x + self.width * self.res)
            bounds.setHigh(1, self.origin_y + self.height * self.res)
            space.setBounds(bounds)

        # --- 2) SimpleSetup & validity checker ------------------------------
        self.ss = og.SimpleSetup(space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.is_state_valid)
        )
        si = self.ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.01 / space.getMaximumExtent())
        obj = ob.PathLengthOptimizationObjective(si)
        self.ss.setOptimizationObjective(obj)


        # --- 3) Planner (RRTConnect: finds path fast, no optimisation timeout) --
        if planner_factory is None:
            planner = og.RRTConnect(si)
        else:
            planner = planner_factory(si)
        self.ss.setPlanner(planner)

    # ------------------------------------------------------------------ #
    # State validity: collision check on occupancy grid
    # ------------------------------------------------------------------ #
    def is_state_valid(self, state: ob.State) -> bool:
        # Extract x, y based on motion model
        if self.motion_model == "r2":
            x, y = state[0], state[1]
        else:
            x = state.getX()
            y = state.getY()

        j = int((x - self.origin_x) / self.res)
        i = int((y - self.origin_y) / self.res)

        if i < 0 or j < 0 or i >= self.height or j >= self.width:
            return False  # outside map

        # Single lookup — obstacles already inflated by robot_radius at construction
        return self.occ_grid[i, j] == 0

    # ------------------------------------------------------------------ #
    # Core: plan path as list of (x, y, yaw)
    # ------------------------------------------------------------------ #
    def _plan_path_poses(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        timeout: float,
        interpolate: int,
    ) -> List[Tuple[float, float, float]]:
        space = self.ss.getStateSpace()

        if self.motion_model == "r2":
            # R2: position-only planning
            start_state = ob.State(space)
            start_state()[0] = start[0]
            start_state()[1] = start[1]

            goal_state = ob.State(space)
            goal_state()[0] = goal[0]
            goal_state()[1] = goal[1]
        else:
            # SE2 or ReedsShepp: full pose planning
            start_state = ob.State(space)
            s = start_state()
            s.setX(start[0])
            s.setY(start[1])
            s.setYaw(start[2])

            goal_state = ob.State(space)
            g = goal_state()
            g.setX(goal[0])
            g.setY(goal[1])
            g.setYaw(goal[2])

        # Clear previous planning data so the instance can be reused
        self.ss.clear()
        self.ss.setStartAndGoalStates(start_state, goal_state)

        solved = self.ss.solve(timeout)
        if not solved:
            return []

        self.ss.simplifySolution()
        path: og.PathGeometric = self.ss.getSolutionPath()
        if interpolate is not None and interpolate > 0:
            path.interpolate(interpolate)

        # Extract poses from path
        poses: List[Tuple[float, float, float]] = []
        n_states = path.getStateCount()
        
        if self.motion_model == "r2":
            # R2: interpolate yaw from start to goal along path
            for i in range(n_states):
                st = path.getState(i)
                x, y = st[0], st[1]
                # Linear yaw interpolation from start to goal
                alpha = i / max(1, n_states - 1)
                dtheta = (goal[2] - start[2] + math.pi) % (2.0 * math.pi) - math.pi
                yaw = start[2] + alpha * dtheta
                poses.append((x, y, yaw))
        else:
            for i in range(n_states):
                st = path.getState(i)
                poses.append((st.getX(), st.getY(), st.getYaw()))

        return poses

    # ------------------------------------------------------------------ #
    # Public: plan & directly get teleport poses
    # ------------------------------------------------------------------ #
    def plan_teleport_commands(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        timeout: float = 1.0,
        interpolate: int = 200,
        max_step_trans: float = 0.05,       # max distance per teleport (m)
        max_step_rot: float = math.radians(5.0),  # max yaw change per teleport (rad)
        speedup_factor: int = 1,            # take every Nth teleport (1=all, 5=every 5th)
    ) -> List[Tuple[float, float, float]]:
        """
        Plan from start to goal and return a list of teleport poses (x, y, yaw),
        such that each step between consecutive poses respects:
            |Δpos| <= max_step_trans
            |Δyaw| <= max_step_rot
        
        speedup_factor: Sample every Nth teleport from the full sequence.
                       E.g., speedup_factor=5 means each output pose corresponds
                       to 5 consecutive teleports, making navigation 5x faster.
        """
        path_poses = self._plan_path_poses(start, goal, timeout, interpolate)
        if len(path_poses) == 0:
            return []

        teleports: List[Tuple[float, float, float]] = []
        teleports.append(path_poses[0])

        for k in range(1, len(path_poses)):
            x0, y0, th0 = teleports[-1]  # last appended teleport
            x1, y1, th1 = path_poses[k]

            dx = x1 - x0
            dy = y1 - y0
            d_pos = math.hypot(dx, dy)

            # Wrap yaw difference to [-pi, pi]
            dtheta = (th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi
            abs_dtheta = abs(dtheta)

            # How many substeps needed to respect both constraints?
            n_steps_pos = math.ceil(d_pos / max_step_trans) if max_step_trans > 0 else 1
            n_steps_rot = math.ceil(abs_dtheta / max_step_rot) if max_step_rot > 0 else 1
            n_steps = max(1, int(max(n_steps_pos, n_steps_rot)))

            if n_steps == 1:
                # Directly append target pose
                teleports.append((x1, y1, th1))
                continue

            # Interpolate intermediate teleports
            for s in range(1, n_steps + 1):
                alpha = s / n_steps
                x = x0 + alpha * dx
                y = y0 + alpha * dy
                yaw = th0 + alpha * dtheta
                # wrap yaw to [-pi, pi] for cleanliness
                yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
                teleports.append((x, y, yaw))

        # Apply speedup factor: sample every Nth teleport
        if speedup_factor > 1:
            speedup_teleports = [teleports[0]]  # Always include start
            for i in range(speedup_factor, len(teleports), speedup_factor):
                speedup_teleports.append(teleports[i])
            # Always include final pose if not already included
            if speedup_teleports[-1] != teleports[-1]:
                speedup_teleports.append(teleports[-1])
            return speedup_teleports

        return teleports
