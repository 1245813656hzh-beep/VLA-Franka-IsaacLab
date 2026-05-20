# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic environment class for the place-cubes-into-bin task (Franka, IK Rel).

Extends ManagerBasedRLMimicEnv with:
- EEF pose retrieval (same as stack task)
- IK-relative action ↔ target pose conversions (same as stack task)
- Subtask termination signals using object_grasped() and object_in_bin()
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaPlaceBinIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for Franka place-cubes-into-bin IK Rel env."""

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Get current robot end effector pose as a 4x4 matrix.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices. If None, all envs are considered.

        Returns:
            Tensor of shape (len(env_ids), 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose + gripper action to a normalized delta-pose action.

        Args:
            target_eef_pose_dict: Dict of 4x4 target pose per end-effector.
            gripper_action_dict: Dict of gripper actions per end-effector.
            action_noise_dict: Optional noise per end-effector.
            env_id: Environment index.

        Returns:
            Action tensor compatible with env.step().
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # Disable end-effector rotation augmentation:
        # always keep orientation unchanged during datagen.
        delta_rotation = torch.zeros(3, device=target_pos.device, dtype=target_pos.dtype)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            # Position-only noise (no rotation noise)
            noise = torch.zeros_like(pose_action)
            noise[:3] = action_noise_dict[eef_name] * torch.randn_like(pose_action[:3])
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert env action to target EEF pose (inverse of target_eef_pose_to_action).

        Args:
            action: Environment action of shape (num_envs, action_dim).

        Returns:
            Dict mapping eef_name → target pose tensor.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Keep current orientation fixed (disable rotation augmentation)
        target_rot = curr_rot

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper action from environment actions.

        Args:
            actions: Env actions of shape (num_envs, num_steps, action_dim).

        Returns:
            Dict mapping eef_name → gripper action tensor.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Get termination signal flags for each subtask.

        Cube3 mode with split subtasks:
          - grasp_3: cube_3 grasped
          - lift_3: cube_3 lifted above table
          - place_3: cube_3 placed in bin

        Args:
            env_ids: Environment indices. If None, all envs are considered.

        Returns:
            Dict of termination signal flags for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "grasp_3": subtask_terms["grasp_3"][env_ids],
            "lift_3": subtask_terms["lift_3"][env_ids],
            "place_3": subtask_terms["place_3"][env_ids],
        }

    def get_expected_attached_object(
        self, eef_name: str, subtask_index: int, env_cfg
    ) -> str | None:
        """(SkillGen) Return the expected attached object for the given EEF/subtask.

        For 'place' subtasks, the robot is holding the cube grasped in the preceding 'grasp' subtask.
        For 'grasp' subtasks, nothing is attached at the start.
        """
        if eef_name not in env_cfg.subtask_configs:
            return None

        subtask_configs = env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None

        current_cfg = subtask_configs[subtask_index]
        # If placing into bin, expect we are holding the object grasped in a prior subtask
        if "place" in str(current_cfg.subtask_term_signal or "").lower() or (
            current_cfg.subtask_term_signal is None and subtask_index > 0
        ):
            if subtask_index > 0:
                for i in range(subtask_index - 1, -1, -1):
                    prev_cfg = subtask_configs[i]
                    if "grasp" in str(prev_cfg.subtask_term_signal or "").lower():
                        return prev_cfg.object_ref
        return None
