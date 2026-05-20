#!/usr/bin/env python3
"""
GR00T-N1.5 inference for Franka place-bin task on Isaac Sim.

Usage:
python scripts/inference/inference_gr00t_isaaclab.py \
    --model_path ./pretrained_models/gr00t_place_bin \
    --num_episodes 3 \
    --headless \
    --save_video
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add GR00T-N1.5 to path (set GR00T_PATH env var or place it under third_party/)
GR00T_PATH = os.environ.get("GR00T_PATH", str(PROJECT_ROOT / "third_party" / "GR00T-N1.5"))
if os.path.exists(GR00T_PATH):
    sys.path.insert(0, str(GR00T_PATH))
else:
    print(f"[WARNING] GR00T path not found: {GR00T_PATH}. Ensure GR00T is installed or set GR00T_PATH.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Task description used for language conditioning
DEFAULT_TASK_DESCRIPTION = "pick up the green cubes and place them into the blue bin"


def parse_args():
    parser = argparse.ArgumentParser(
        description="GR00T-N1.5 Inference on Isaac Sim for Franka Place-Bin"
    )
    parser.add_argument("--task", type=str, default="Isaac-Place-Bin-Franka-IK-Rel-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--episode_length", type=int, default=250)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained_models/gr00t_place_bin",
        help="Path to GR00T pretrained model directory",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr00t.config.franka_config:FrankaDataConfig",
        help="Data config module path for GR00T",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag for GR00T model",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=4,
        help="Number of denoising steps for GR00T action head",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=DEFAULT_TASK_DESCRIPTION,
        help="Language task description for policy conditioning",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--save_video", action="store_true", help="Save episode videos")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for videos",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


class GR00TPolicyWrapper:
    """Wrapper for GR00T-N1.5 policy to interface with Isaac Sim environment."""

    def __init__(
        self,
        model_path: str,
        data_config: str,
        embodiment_tag: str = "new_embodiment",
        denoising_steps: int = 4,
        device: str = "cuda",
        task_description: Optional[str] = None,
        verbose: bool = False,
    ):
        self.device = device
        self.verbose = verbose
        self.task_description = task_description or DEFAULT_TASK_DESCRIPTION

        print(f"Loading GR00T model from: {model_path}")

        # Import GR00T modules (need sys.path set up before)
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy

        # Load data config to get modality config and transform
        self.data_config = load_data_config(data_config)
        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()

        print(f"  Data config: {data_config}")
        print(f"  Embodiment tag: {embodiment_tag}")
        print(f"  Denoising steps: {denoising_steps}")

        # Create the policy
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            device=device,
        )

        # Cache modality config for observation construction
        self.modality_config = modality_config

        # Get action horizon from modality config
        self.action_horizon = len(modality_config["action"].delta_indices)
        print(f"  Action horizon: {self.action_horizon}")

        print("Model loaded successfully!")

    def reset(self):
        """Reset policy state if needed."""
        # GR00T policy doesn't require explicit reset like ACT
        pass

    def _extract_numpy(self, obs_dict: dict, key: str, expected_dim: int) -> np.ndarray | None:
        """Extract numpy array from observation dict."""
        if key not in obs_dict:
            return None
        val = obs_dict[key]
        if isinstance(val, torch.Tensor):
            val = val[0].cpu().numpy() if val.dim() > 1 else val.cpu().numpy()
        val = val.flatten()[:expected_dim]
        if val.shape[0] < expected_dim:
            val = np.pad(val, (0, expected_dim - val.shape[0]), mode="constant")
        return val.astype(np.float32)

    def _prepare_observation(self, obs_dict: dict) -> Dict[str, Any]:
        """Convert Isaac Sim observation dict to GR00T expected format."""
        gr00t_obs = {}

        # ---- Images ----
        # GR00T expects: (T, H, W, C) uint8, where T is temporal horizon
        for cam_key in ["table_cam", "table_cam_side", "wrist_cam"]:
            if cam_key not in obs_dict:
                if self.verbose:
                    print(f"  [WARNING] {cam_key} NOT found in obs_dict!")
                continue

            img = obs_dict[cam_key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            # Handle batch dimension: (B, ...) -> (...)
            if img.ndim == 4:
                img = img[0]

            # Handle channel dimension: Isaac may output (C, H, W)
            if img.ndim == 3 and img.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))

            # Ensure uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # CRITICAL: horizontally flip the image to match training data orientation.
            # The Isaac Sim live rendering produces images that are mirrored left-to-right
            # compared to the encoded training videos. Without this flip, the model sees
            # objects on the opposite side and outputs actions in the wrong direction.
            img = img[:, ::-1, :].copy()

            # Add time dimension: (H, W, C) -> (1, H, W, C)
            gr00t_obs[f"video.{cam_key}"] = np.expand_dims(img, axis=0)

            if self.verbose:
                print(
                    f"  [Image] {cam_key}: shape={gr00t_obs[f'video.{cam_key}'].shape}, "
                    f"dtype={gr00t_obs[f'video.{cam_key}'].dtype}"
                )

        # ---- State ----
        # Construct state components
        eef_pos = self._extract_numpy(obs_dict, "eef_pos", 3)
        eef_quat = self._extract_numpy(obs_dict, "eef_quat", 4)
        gripper_pos = self._extract_numpy(obs_dict, "gripper_pos", 2)
        joint_pos = self._extract_numpy(obs_dict, "joint_pos", 9)

        if eef_pos is None:
            eef_pos = np.zeros(3, dtype=np.float32)
        if eef_quat is None:
            eef_quat = np.zeros(4, dtype=np.float32)
        if gripper_pos is None:
            gripper_pos = np.zeros(2, dtype=np.float32)
        if joint_pos is None:
            joint_pos = np.zeros(9, dtype=np.float32)

        # end_effector = eef_pos(3) + eef_quat(4) = 7D
        end_effector = np.concatenate([eef_pos, eef_quat], axis=0).astype(np.float32)
        # Add time dimension: (D,) -> (1, D)
        gr00t_obs["state.end_effector"] = np.expand_dims(end_effector, axis=0)

        # fingers = gripper_pos (2D)
        gr00t_obs["state.fingers"] = np.expand_dims(gripper_pos, axis=0)

        # joints = joint_pos (9D)
        gr00t_obs["state.joints"] = np.expand_dims(joint_pos, axis=0)

        if self.verbose:
            print(f"  [State] end_effector shape={end_effector.shape}")
            print(f"  [State] fingers shape={gripper_pos.shape}")
            print(f"  [State] joints shape={joint_pos.shape}")

        # ---- Language ----
        # GR00T expects a list of strings
        gr00t_obs["annotation.human.action.task_description"] = [self.task_description]

        return gr00t_obs

    def get_action(self, obs_dict: dict) -> np.ndarray:
        """Get action from GR00T policy given current observation."""
        # Convert Isaac Sim obs to GR00T format
        gr00t_obs = self._prepare_observation(obs_dict)

        if self.verbose:
            print("  [GR00T] Observation keys:", list(gr00t_obs.keys()))
            for k, v in gr00t_obs.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

        # Run inference
        with torch.inference_mode():
            action_dict = self.policy.get_action(gr00t_obs)

        # Extract action from dict
        # action_dict["action.full_action"] has shape (action_horizon, 7) or (B, action_horizon, 7)
        action_key = "action.full_action"
        if action_key not in action_dict:
            raise KeyError(
                f"Expected action key '{action_key}' not found in output. Got: {list(action_dict.keys())}"
            )

        action = action_dict[action_key]
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Remove batch dimension if present
        if action.ndim == 3:
            action = action[0]

        # Take the first action from the action chunk
        # shape: (action_horizon, 7) -> (7,)
        action = action[0]

        if self.verbose:
            print(f"  [GR00T] Raw action shape: {action.shape}")
            print(f"  [GR00T] Raw action: {action}")

        # Post-process action for Isaac Sim
        # Lock rotation to 0 for IK-Rel
        action[3:6] = 0.0

        # Gripper thresholding: binary open/close based on sign
        action[6] = 1.0 if action[6] > 0.0 else -1.0

        if self.verbose:
            print(
                f"  [Action] dx={action[0]:.4f}, dy={action[1]:.4f}, dz={action[2]:.4f}, "
                f"gripper={action[6]:.1f}"
            )

        return action.astype(np.float32)


def run_inference():
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"gr00t_inference_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from tasks.franka import place_bin_ik_rel_env_cfg  # noqa: F401

    policy = GR00TPolicyWrapper(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
        task_description=args.task_description,
        verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print("GR00T-N1.5 Place-Bin Inference - Franka on Isaac Sim")
    print("=" * 70)

    task_name = args.task
    print(f"\nLoading environment: {task_name}")

    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.concatenate_terms = False
    env = gym.make(task_name, cfg=env_cfg)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    episode_lengths = []
    if args.save_video:
        print(f"Video saving enabled. Will save to: {output_dir}")

    env.unwrapped.sim.reset()
    for ep in range(args.num_episodes):
        obs, info = env.reset()
        policy.reset()
        done = False
        step = 0
        obs_dict = obs["policy"]
        eef_pos_0 = obs_dict["eef_pos"][0].cpu().numpy()
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        print(f"  Step 0 EEF pos: [{eef_pos_0[0]:.4f}, {eef_pos_0[1]:.4f}, {eef_pos_0[2]:.4f}]")

        video_frames = []
        video_frames_side = []
        video_frames_wrist = []

        while step < args.episode_length and not done and simulation_app.is_running():
            action = policy.get_action(obs_dict)
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(args.device)
            obs, reward, terminated, truncated, info = env.step(action_tensor)

            if args.save_video:
                for frames, cam_name in [
                    (video_frames, "table_cam"),
                    (video_frames_side, "table_cam_side"),
                    (video_frames_wrist, "wrist_cam"),
                ]:
                    img = obs["policy"][cam_name][0].cpu().numpy()
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    frames.append(img)

            if terminated.ndim > 0:
                done = bool(
                    terminated[0].item()
                    if isinstance(terminated[0], torch.Tensor)
                    else terminated[0]
                )
                done = done or bool(
                    truncated[0].item() if isinstance(truncated[0], torch.Tensor) else truncated[0]
                )
            else:
                done = bool(terminated or truncated)

            obs_dict = obs["policy"]
            step += 1

            if step % 50 == 0:
                eef_pos = obs_dict.get("eef_pos", None)
                if eef_pos is not None:
                    if isinstance(eef_pos, torch.Tensor):
                        eef_pos = eef_pos[0].cpu().numpy()
                    print(
                        f"  Step {step}: EEF pos = [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]"
                    )
                else:
                    print(f"  Step {step}")

        episode_lengths.append(step)
        print(f"Episode {ep + 1} finished: {step} steps")

        if args.save_video:
            for frames, suffix in [
                (video_frames, "table_cam"),
                (video_frames_side, "table_cam_side"),
                (video_frames_wrist, "wrist_cam"),
            ]:
                if frames:
                    video_path = output_dir / f"episode_{ep + 1}_{suffix}.mp4"
                    h, w = frames[0].shape[:2]
                    out = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                    )
                    for frame in frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    print(f"  Saved video: {video_path} ({len(frames)} frames)")

    print("\n" + "=" * 70)
    print("Inference Summary")
    print("=" * 70)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Min episode length: {min(episode_lengths)} steps")
    print(f"Max episode length: {max(episode_lengths)} steps")

    env.close()
    simulation_app.close()
    print("\nInference completed!")


if __name__ == "__main__":
    run_inference()
