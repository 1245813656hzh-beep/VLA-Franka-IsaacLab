#!/usr/bin/env python3
"""
Debug: Compare model output on env observation vs nearest training frame.
Run in Isaac env with GPU.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Optionally add custom LeRobot path
_lerobot_env = os.environ.get("LEROBOT_PATH", "")
if _lerobot_env:
    sys.path.insert(0, _lerobot_env)

import transformers.cache_utils
if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
    transformers.cache_utils.SlidingWindowCache = object

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "./pretrained_models/act_place_bin")
DATASET_PATH = os.environ.get("DATASET_PATH", "./datasets/lerobot/auto_collected")


def load_model(device="cuda"):
    model_path = Path(MODEL_PATH)
    with open(model_path / "config.json") as f:
        config_dict = json.load(f)
    for key in {"use_peft", "use_rabc", "rabc_progress_path", "rabc_kappa", "rabc_epsilon", "rabc_head_mode"}:
        config_dict.pop(key, None)

    temp_dir = Path("/tmp/act_debug_env")
    temp_dir.mkdir(exist_ok=True)
    with open(temp_dir / "config.json", "w") as f:
        json.dump(config_dict, f)
    shutil.copy(model_path / "model.safetensors", temp_dir / "model.safetensors")
    for fname in ["policy_preprocessor.json", "policy_preprocessor_step_3_normalizer_processor.safetensors",
                  "policy_postprocessor.json", "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]:
        if (model_path / fname).exists():
            shutil.copy(model_path / fname, temp_dir / fname)

    policy = ACTPolicy.from_pretrained(temp_dir)
    policy.to(device)
    policy.eval()

    preprocessor = None
    postprocessor = None
    if (model_path / "policy_preprocessor.json").exists():
        preprocessor = DataProcessorPipeline.from_pretrained(
            model_path, config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
        )
    if (model_path / "policy_postprocessor.json").exists():
        postprocessor = DataProcessorPipeline.from_pretrained(
            model_path, config_filename="policy_postprocessor.json",
            overrides={"device_processor": {"device": device}},
        )

    return policy, preprocessor, postprocessor


def run_inference(batch, policy, preprocessor, postprocessor):
    if preprocessor is not None:
        batch = preprocessor(batch)
    with torch.no_grad():
        policy.reset()
        raw = policy.select_action(batch).squeeze().cpu().numpy()
    if postprocessor is not None:
        action_dict = {"action": torch.from_numpy(raw).unsqueeze(0)}
        action_dict = postprocessor(action_dict)
        pred = action_dict["action"].squeeze().cpu().numpy()
    else:
        with open(f"{DATASET_PATH}/meta/stats.json") as f:
            stats = json.load(f)
        action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
        action_std = np.array(stats["action"]["std"], dtype=np.float32)
        pred = raw * action_std + action_mean
    return raw, pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_model(device)

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.franka import place_bin_ik_rel_env_cfg  # noqa: F401

    env_cfg = parse_env_cfg("Isaac-Place-Bin-Franka-IK-Rel-v0", device="cuda:0")
    env = gym.make("Isaac-Place-Bin-Franka-IK-Rel-v0", cfg=env_cfg)
    obs, info = env.reset(seed=42)
    obs_dict = obs["policy"]

    # Print raw env observation
    print("=" * 70)
    print("ENV OBSERVATION (after reset)")
    print("=" * 70)
    for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
        val = obs_dict[key][0].cpu().numpy()
        print(f"  {key}: {val}")

    # Build env batch exactly like inference script
    env_batch = {}
    for cam in ["table_cam", "table_cam_side", "wrist_cam"]:
        img = obs_dict[cam][0].cpu().numpy()
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        env_batch[f"observation.images.{cam}"] = torch.from_numpy(img).unsqueeze(0).to(device)
        print(f"  {cam}: shape={env_batch[f'observation.images.{cam}'].shape}, "
              f"dtype={env_batch[f'observation.images.{cam}'].dtype}, "
              f"min={env_batch[f'observation.images.{cam}'].min():.4f}, max={env_batch[f'observation.images.{cam}'].max():.4f}")

    eef_pos = obs_dict["eef_pos"][0].cpu().numpy().flatten()[:3]
    eef_quat = obs_dict["eef_quat"][0].cpu().numpy().flatten()[:4]
    gripper_pos = obs_dict["gripper_pos"][0].cpu().numpy().flatten()[:2]
    joint_pos = obs_dict["joint_pos"][0].cpu().numpy().flatten()[:9]
    env_state = np.concatenate([eef_pos, eef_quat, gripper_pos, joint_pos]).astype(np.float32)
    env_batch["observation.state"] = torch.from_numpy(env_state).unsqueeze(0).to(device)
    print(f"  observation.state: shape={env_state.shape}, Z={env_state[2]:.4f}")

    env_raw, env_pred = run_inference(env_batch, policy, preprocessor, postprocessor)
    print(f"\n  ENV  raw action:  {env_raw}")
    print(f"  ENV  pred action: {env_pred}")
    print(f"  ENV  dz = {env_pred[2]:.6f}")

    # Find closest training frame by Z height
    print("\n" + "=" * 70)
    print("FINDING CLOSEST TRAINING FRAME BY Z HEIGHT")
    print("=" * 70)
    pf = pq.ParquetFile(f"{DATASET_PATH}/data/chunk-000/file-000.parquet")
    table = pf.read()
    states = [np.array(s, dtype=np.float32) for s in table.column("observation.state").to_pylist()]
    actions = [np.array(a, dtype=np.float32) for a in table.column("action").to_pylist()]
    ep_indices = np.array(table.column("episode_index").to_pylist())
    frame_indices = np.array(table.column("frame_index").to_pylist())

    # Find frame with closest Z
    env_z = env_state[2]
    closest_idx = min(range(len(states)), key=lambda i: abs(states[i][2] - env_z))
    train_state = states[closest_idx]
    train_action = actions[closest_idx]
    print(f"  Env Z: {env_z:.4f}")
    print(f"  Closest train frame: row={closest_idx}, ep={ep_indices[closest_idx]}, frame={frame_indices[closest_idx]}")
    print(f"  Train Z: {train_state[2]:.4f}")
    print(f"  Train state: {train_state}")
    print(f"  Train action (GT): {train_action}")

    # Load corresponding video frames
    import av
    train_batch = {}
    for cam in ["table_cam", "table_cam_side", "wrist_cam"]:
        path = f"{DATASET_PATH}/videos/{cam}/chunk-000/file-000.mp4"
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.base_rate)
        target_frame = frame_indices[closest_idx]
        container.seek(int(max(0, target_frame / fps - 0.5) / stream.time_base), stream=stream)
        for packet in container.demux(stream):
            for frame in packet.decode():
                idx = int(frame.pts * stream.time_base * fps)
                if idx == target_frame:
                    img = frame.to_ndarray(format="rgb24")
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    train_batch[f"observation.images.{cam}"] = img.unsqueeze(0).to(device)
                    break
            if f"observation.images.{cam}" in train_batch:
                break
        container.close()

    train_batch["observation.state"] = torch.from_numpy(train_state).unsqueeze(0).to(device)
    train_raw, train_pred = run_inference(train_batch, policy, preprocessor, postprocessor)
    print(f"\n  TRAIN raw action:  {train_raw}")
    print(f"  TRAIN pred action: {train_pred}")
    print(f"  TRAIN dz = {train_pred[2]:.6f}")

    # Also test with EXACT same state but env image
    print("\n" + "=" * 70)
    print("HYBRID TEST: env images + train state")
    print("=" * 70)
    hybrid_batch = dict(env_batch)
    hybrid_batch["observation.state"] = torch.from_numpy(train_state).unsqueeze(0).to(device)
    hybrid_raw, hybrid_pred = run_inference(hybrid_batch, policy, preprocessor, postprocessor)
    print(f"  HYBRID pred action: {hybrid_pred}")
    print(f"  HYBRID dz = {hybrid_pred[2]:.6f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
