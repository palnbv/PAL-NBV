"""The single-robot RGB-D warehouse used by PALLET NBV."""

from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from .assets import BLOCKERS, PALLET, WAREHOUSE


@configclass
class PalletEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 30.0  # Required by DirectRLEnv; the runner owns the actual step limit.
    sim = sim_utils.SimulationCfg(dt=1 / 120, render_interval=1)
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=20.0, replicate_physics=True)
    action_space = 4
    observation_space = 0  # The runner consumes named sensor tensors directly.
    state_space = 0
    asset_root: str = ""


def _usd_object(root: Path, name: str, asset: str, mass: float, *, static: bool = False) -> RigidObject:
    return RigidObject(RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(root / asset),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=static),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
    ))


def _camera(name: str, data_type: str, far_clip: float) -> TiledCamera:
    return TiledCamera(TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot/{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.51, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, far_clip)
        ),
        width=640,
        height=640,
        update_latest_camera_pose=False,
    ))


class PalletEnv(DirectRLEnv):
    cfg: PalletEnvCfg

    def _setup_scene(self):
        root = Path(self.cfg.asset_root)
        self.robot = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.95),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.5, 2.25)),
        ))
        self.pallet = _usd_object(root, "object", PALLET, 1000.0)
        warehouse = _usd_object(root, "warehouse", WAREHOUSE, 10000000.0, static=True)
        self.blockers = [_usd_object(root, f"distractor_{i}", usd, 10.0) for i, usd in enumerate(BLOCKERS)]
        self.pallet_props = [_usd_object(root, f"pallet_distractor_{i}", usd, 5.0) for i, usd in enumerate(BLOCKERS[:4])]
        self.rgb = _camera("Camera", "rgb", 30.0)
        self.depth = _camera("Depth_Camera", "distance_to_image_plane", 10.0)
        self.contact = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot"))

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects.update(robot=self.robot, pallet=self.pallet, warehouse=warehouse)
        self.scene.rigid_objects.update({f"distractor_{i}": obj for i, obj in enumerate(self.blockers)})
        self.scene.rigid_objects.update({f"pallet_distractor_{i}": obj for i, obj in enumerate(self.pallet_props)})
        self.scene.sensors.update(rgb=self.rgb, depth=self.depth, contact=self.contact)
        self._offset_position = torch.tensor(self.rgb.cfg.offset.pos, device=self.device)
        self._offset_rotation = torch.tensor(self.rgb.cfg.offset.rot, device=self.device)

    def reset_to_scenario(self, scenario: dict):
        super()._reset_idx(self.robot._ALL_INDICES)
        self._place(self.robot, scenario["robot_position"], scenario["robot_yaw"])
        self._place(self.pallet, scenario["object_position"], scenario["object_yaw"])
        hidden = (1000.0, 1000.0, 1000.0)
        positions = scenario.get("blocking_distractor_positions", [])[:scenario.get("num_blocking_distractors", 0)]
        for i, obj in enumerate(self.blockers):
            self._place(obj, positions[i] if i < len(positions) else hidden)
        for i, obj in enumerate(self.pallet_props):
            if i < scenario.get("num_pallet_distractors", 0):
                rng = np.random.RandomState(1000 + i)  # Keep legacy placements without changing the global RNG.
                offsets = (rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), 0.6)
                self._place(obj, np.asarray(scenario["object_position"]) + offsets, rng.uniform(-np.pi, np.pi))
            else:
                self._place(obj, hidden)

        self.scene.write_data_to_sim()
        for _ in range(32):
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)
        self.contact.reset()
        return self._get_observations()

    def _place(self, obj, position, yaw=0.0):
        state = torch.zeros((1, 13), device=self.device)
        state[0, :3] = torch.as_tensor(position, device=self.device)
        state[0, 3] = np.cos(yaw / 2)
        state[0, 6] = np.sin(yaw / 2)
        obj.write_root_state_to_sim(state)

    def _pre_physics_step(self, actions):
        self.actions = actions

    def _apply_action(self):
        rotation = self.robot.data.root_quat_w
        half_yaw = self.actions[:, 3] / 2
        delta_quat = torch.zeros_like(rotation)
        delta_quat[:, 0] = half_yaw.cos()
        delta_quat[:, 3] = half_yaw.sin()
        state = torch.zeros_like(self.robot.data.root_state_w)
        state[:, :3] = self.robot.data.root_pos_w + quat_apply(rotation, self.actions[:, :3])
        state[:, 3:7] = quat_mul(rotation, delta_quat)
        self.robot.write_root_state_to_sim(state)

    def _get_observations(self):
        return {
            "camera_rgb": self.rgb.data.output["rgb"],
            "camera_depth": self.depth.data.output["distance_to_image_plane"],
            "robot_pos": self.robot.data.root_pos_w,
            "robot_quat": self.robot.data.root_quat_w,
            "intrinsic_matrix": self.rgb.data.intrinsic_matrices,
            "camera_offset_pos": self._offset_position,
            "camera_offset_rot": self._offset_rotation,
        }

    def collided(self) -> bool:
        return bool(self.contact.data.net_forces_w.any().item())

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        # Disable DirectRLEnv's automatic reset so the runner can estimate the terminal pose.
        done = torch.zeros(1, dtype=torch.bool, device=self.device)
        return done, done