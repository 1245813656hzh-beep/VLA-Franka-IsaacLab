#!/usr/bin/env python
"""Debug single step of ACT inference."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

# Optionally add custom LeRobot path
_lerobot_env = os.environ.get("LEROBOT_PATH", "")
if _lerobot_env:
    sys.path.insert(0, _lerobot_env)

# Monkey-patch missing transformers class
import transformers.cache_utils

if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
    transformers.cache_utils.SlidingWindowCache = object  # type: ignore[attr-defined]

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "./pretrained_models/act_place_bin")
DATASET_PATH = os.environ.get("DATASET_PATH", "./datasets/lerobot/auto_collected")


def main():
    model_path = Path(MODEL_PATH)

    # Load config
    with open(model_path / "config.json") as f:
        config_dict = json.load(f)
    for key in {"use_peft", "use_rabc", "rabc_progress_path", "rabc_kappa", "rabc_epsilon", "rabc_head_mode"}:
        config_dict.pop(key, None)

    temp_dir = Path("/tmp/act_model_debug")
    temp_dir.mkdir(exist_ok=True)
    with open(temp_dir / "config.json", "w") as f:
        json.dump(config_dict, f)
    shutil.copy(model_path / "model.safetensors", temp_dir / "model.safetensors")

    policy = ACTPolicy.from_pretrained(temp_dir)
    policy.to("cuda")
    policy.eval()

    preprocessor = None
    postprocessor = None
    if (model_path / "policy_preprocessor.json").exists():
        preprocessor = DataProcessorPipeline.from_pretrained(
            model_path, config_filename="policy_preprocessor.json"
        )
    if (model_path / "policy_postprocessor.json").exists():
        postprocessor = DataProcessorPipeline.from_pretrained(
            model_path, config_filename="policy_postprocessor.json"
        )

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from tasks.franka import place_bin_ik_rel_env_cfg  # noqa: F401

    env_cfg = parse_env_cfg(
        "Isaac-Place-Bin-Franka-IK-Rel-v0",
        device="cuda:0",
    )
    env = gym.make("Isaac-Place-Bin-Franka-IK-Rel-v0", cfg=env_cfg)
    obs, info = env.reset(seed=42)

    obs_dict = obs["policy"]

    # Print raw observation values
    print("=" * 60)
    print("RAW OBSERVATION FROM ISAAC SIM (after reset)")
    print("=" * 60)
    for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
        val = obs_dict[key][0].cpu().numpy()
        print(f"  {key}: {val}")

    # Construct batch exactly like inference script
    batch = {}
    use_preprocessor = preprocessor is not None

    for cam in ["table_cam", "table_cam_side", "wrist_cam"]:
        img = obs_dict[cam][0].cpu().numpy()
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
        if not use_preprocessor:
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        batch[f"observation.images.{cam}"] = torch.from_numpy(img).unsqueeze(0).to("cuda")
        print(f"  Image {cam}: shape={batch[f'observation.images.{cam}'].shape}, "
              f"min={batch[f'observation.images.{cam}'].min():.4f}, max={batch[f'observation.images.{cam}'].max():.4f}")

    eef_pos = obs_dict["eef_pos"][0].cpu().numpy().flatten()[:3]
    eef_quat = obs_dict["eef_quat"][0].cpu().numpy().flatten()[:4]
    gripper_pos = obs_dict["gripper_pos"][0].cpu().numpy().flatten()[:2]
    joint_pos = obs_dict["joint_pos"][0].cpu().numpy().flatten()[:9]
    state = np.concatenate([eef_pos, eef_quat, gripper_pos, joint_pos]).astype(np.float32)

    if not use_preprocessor:
        with open(f"{DATASET_PATH}/meta/stats.json") as f:
            stats = json.load(f)
        state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
        state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
        state_std = np.where(state_std == 0, 1.0, state_std)
        state = (state - state_mean) / state_std

    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to("cuda")
    print(f"  observation.state (after prep): shape={batch['observation.state'].shape}, "
          f"min={batch['observation.state'].min():.4f}, max={batch['observation.state'].max():.4f}")

    # Apply preprocessor
    if preprocessor is not None:
        batch = preprocessor(batch)
        print(f"  observation.state (after preprocessor): "
              f"min={batch['observation.state'].min():.4f}, max={batch['observation.state'].max():.4f}")

    # Inference
    with torch.no_grad():
        raw_action = policy.select_action(batch).squeeze().cpu().numpy()
    print(f"  RAW action from model: {raw_action}")

    # Postprocess
    if postprocessor is not None:
        action_dict = {"action": torch.from_numpy(raw_action).unsqueeze(0)}
        action_dict = postprocessor(action_dict)
        action = action_dict["action"].squeeze().cpu().numpy()
    else:
        with open(f"{DATASET_PATH}/meta/stats.json") as f:
            stats = json.load(f)
        action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
        action_std = np.array(stats["action"]["std"], dtype=np.float32)
        action = raw_action * action_std + action_mean

    print(f"  FINAL action (after postproc): {action}")
    print(f"  dz = {action[2]:.6f} (negative=down, positive=up)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
