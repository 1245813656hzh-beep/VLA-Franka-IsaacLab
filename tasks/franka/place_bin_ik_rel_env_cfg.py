# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Place cubes into a bin task — Franka IK Relative control with cameras.

Task: Pick up cubes scattered on the table and place them into a blue sorting bin.
No success condition — use E key to manually save episodes.
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import ManagerBasedEnv
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import (
    GripperRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import (
    Se3RelRetargeterCfg,
)
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import bin_stack_joint_pos_env_cfg
from . import place_bin_observations


##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def randomize_eef_xyz(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    num_envs = len(env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    env.sim.render()

    ee_body_idx = asset.body_names.index("panda_hand")
    jacobi_idx = ee_body_idx - 1 if asset.is_fixed_base else ee_body_idx

    jacobian = asset.root_physx_view.get_jacobians()[env_ids, jacobi_idx, :, :]
    ee_pos_w = asset.data.body_pos_w[env_ids, ee_body_idx]
    ee_quat_w = asset.data.body_quat_w[env_ids, ee_body_idx]
    root_pos_w = asset.data.root_pos_w[env_ids]
    root_quat_w = asset.data.root_quat_w[env_ids]

    from isaaclab.utils.math import subtract_frame_transforms

    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    xyz_offset = torch.zeros(num_envs, 3, device=device)
    xyz_offset[:, 0] = torch.randn(num_envs, device=device) * 0.09
    xyz_offset[:, 1] = torch.randn(num_envs, device=device) * 0.09
    xyz_offset[:, 2] = torch.randn(num_envs, device=device) * 0.04

    target_pos_b = ee_pos_b + xyz_offset
    target_pose_b = torch.cat([target_pos_b, ee_quat_b], dim=-1)

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)
    ik_controller.set_command(target_pose_b)

    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos_des = joint_pos_des.clamp(limits[..., 0], limits[..., 1])
    asset.set_joint_position_target(joint_pos_des, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des), env_ids=env_ids)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values and camera images."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        table_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        table_cam_side = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam_side"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask termination observations for Mimic data generation.

        These signals let Mimic annotate/detect when each subtask completes:
          - grasp_N: cube_N is grasped (EE close + gripper closed)
          - lift_N: cube_N is grasped and lifted above table
          - place_N: cube_N is in the bin (XY/Z check + gripper open)
        """

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        place_1 = ObsTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_1"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        place_2 = ObsTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_2"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )
        grasp_3 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )
        lift_3 = ObsTerm(
            func=place_bin_observations.object_lifted,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
                "min_height": 0.08,
                "diff_threshold": 0.08,
            },
        )
        place_3 = ObsTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_3"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaPlaceBinEnvCfg(bin_stack_joint_pos_env_cfg.FrankaBinStackEnvCfg):
    """Place cubes into bin — IK relative control, dual cameras, no success condition."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent (sets up bin, cubes, events, etc.)
        super().__post_init__()

        # Remove ALL termination conditions (no success, no timeout, no dropping)
        self.terminations.success = None
        self.terminations.time_out = None
        self.terminations.cube_1_dropping = None
        self.terminations.cube_2_dropping = None
        self.terminations.cube_3_dropping = None

        self.events.randomize_franka_joint_state = None
        self.events.randomize_franka_position = EventTerm(
            func=randomize_eef_xyz,
            mode="reset",
            params={
                "std": 0.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.events.reset_blue_bin_pose.params["pose_range"] = {
            "x": (0.38, 0.42),
            "y": (-0.03, 0.03),
            "z": (0.0203, 0.0203),
            "yaw": (-0.1, 0.1, 0),
        }
        self.events.reset_cube_1_pose = None
        self.events.reset_cube_pose = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.62, 0.73),
                    "y": (-0.22, 0.22),
                    "z": (0.0203, 0.0203),
                    "yaw": (-1.0, 1.0, 0),
                },
                "min_separation": 0.1,
                "asset_cfgs": [
                    SceneEntityCfg("cube_1"),
                    SceneEntityCfg("cube_2"),
                    SceneEntityCfg("cube_3"),
                ],
            },
        )

        # Set Franka with stiffer PD controller for better IK tracking
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set IK relative control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.9,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Wrist camera (mounted on panda_hand)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.2, 0.0, -0.15),
                rot=(0.68301, -0.18301, -0.18301, 0.68301),
                convention="ros",
            ),
        )

        # Table camera (overhead view) - original position
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.18, 0.0, 0.7),  # Original position
                rot=(0.28761, -0.64597, -0.64597, 0.28761),
                convention="ros",
            ),
        )

        # Side table camera (side view looking at table)
        self.scene.table_cam_side = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam_side",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.45, -0.8, 0.6),  # Side position
                rot=(-0.5, 0.866, 0, 0),
                convention="ros",
            ),
        )

        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["table_cam", "table_cam_side", "wrist_cam"]

        # Teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.09,
                    rot_sensitivity=0.09,
                    sim_device=self.sim.device,
                ),
            }
        )
