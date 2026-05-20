# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic environment config for the place-cubes-into-bin task (Franka, IK Rel).

Defines 3 subtasks (for target cube_3):
  grasp_3 → lift_3 → place_3
This split reduces dragging after grasp and improves motion quality in generated data.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from . import place_bin_observations
from isaaclab.utils import configclass

from .place_bin_ik_rel_env_cfg import FrankaPlaceBinEnvCfg


@configclass
class FrankaPlaceBinIKRelMimicEnvCfg(FrankaPlaceBinEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config for Franka place-cubes-into-bin IK Rel env."""

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # ── datagen config ──────────────────────────────────────────────
        self.datagen_config.name = "demo_src_place_bin_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # ── subtask configs (cube_3: grasp -> lift -> place) ───────────
        subtask_configs = []
        # 1) Grasp cube_3
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_3",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=10,
                num_fixed_steps=4,
                apply_noise_during_interpolation=False,
                description="Grasp cube 3",
                next_subtask_description="Lift cube 3 above the table",
            )
        )

        # 2) Lift cube_3 (prevents dragging while moving)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="lift_3",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=12,
                num_fixed_steps=3,
                apply_noise_during_interpolation=False,
                description="Lift cube 3",
                next_subtask_description="Place cube 3 into the bin",
            )
        )

        # 3) Place cube_3 into bin
        subtask_configs.append(
            SubTaskConfig(
                object_ref="blue_sorting_bin",
                subtask_term_signal="place_3",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.00,
                num_interpolation_steps=12,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube 3 into the bin",
            )
        )

        self.subtask_configs["franka"] = subtask_configs

        # ── success termination (required for Mimic datagen) ───────────
        self.terminations.success = DoneTerm(
            func=place_bin_observations.object_in_bin,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube_3"),
                "bin_cfg": SceneEntityCfg("blue_sorting_bin"),
            },
        )
