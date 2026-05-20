#!/usr/bin/env python3
"""Quick env check: print initial observation values."""
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.franka import place_bin_ik_rel_env_cfg  # noqa: F401

env_cfg = parse_env_cfg("Isaac-Place-Bin-Franka-IK-Rel-v0", device="cuda:0")
env_cfg.observations.policy.concatenate_terms = False
env = gym.make("Isaac-Place-Bin-Franka-IK-Rel-v0", cfg=env_cfg)
obs, info = env.reset(seed=42)
obs_dict = obs["policy"]

print("=" * 60)
print("ENV OBSERVATION (after reset)")
print("=" * 60)
for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
    val = obs_dict[key][0].cpu().numpy()
    print(f"  {key}: {val}")

for cam in ["table_cam", "table_cam_side", "wrist_cam"]:
    img = obs_dict[cam][0].cpu().numpy()
    print(f"  {cam}: shape={img.shape}, dtype={img.dtype}, min={img.min():.4f}, max={img.max():.4f}")

env.close()
simulation_app.close()
