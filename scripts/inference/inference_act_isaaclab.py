#!/usr/bin/env python3
"""
ACT (Action Chunking Transformer) inference for Franka place-bin task on Isaac Sim.

Usage:
python scripts/inference/inference_act_isaaclab.py \
    --model_path ./pretrained_models/act_place_bin \
    --dataset_path ./datasets/lerobot/auto_collected \
    --num_episodes 3 \
    --headless \
    --save_video
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optionally add custom LeRobot to path (set LEROBOT_PATH env var if using a fork)
CUSTOM_LEROBOT_PATH = os.environ.get("LEROBOT_PATH", "")
if CUSTOM_LEROBOT_PATH and os.path.exists(CUSTOM_LEROBOT_PATH):
    sys.path.insert(0, str(CUSTOM_LEROBOT_PATH))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="ACT Inference on Isaac Sim for Franka Place-Bin")
    parser.add_argument("--task", type=str, default="Isaac-Place-Bin-Franka-IK-Rel-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained_models/act_place_bin",
        help="Path to ACT pretrained model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/lerobot/auto_collected",
        help="Path to LeRobot dataset (for normalization stats)",
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
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=100,
        help="Override n_action_steps at inference time (default: 10). "
        "Reducing this makes the policy re-plan more frequently from fresh observations.",
    )
    return parser.parse_args()


class ACTPolicyWrapper:
    """Wrapper for ACT policy to interface with Isaac Sim environment."""

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        device: str = "cuda",
        verbose: bool = False,
    ):
        self.device = device
        self.verbose = verbose

        print(f"Loading ACT model from: {model_path}")

        # Monkey-patch missing transformers class so that lerobot.policies (which auto-loads wall_x)
        # does not fail during import.
        import transformers.cache_utils

        if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
            transformers.cache_utils.SlidingWindowCache = object  # type: ignore[attr-defined]

        from lerobot.policies.act.modeling_act import ACTPolicy

        # The saved config may contain fields (e.g. use_peft) that this version of ACTConfig
        # does not recognise. Clean the config and load from a temporary directory.
        model_path_obj = Path(model_path)
        with open(model_path_obj / "config.json") as f:
            config_dict = json.load(f)

        # Remove fields not supported by the current ACTConfig / PreTrainedConfig
        unsupported = {
            "use_peft",
            "use_rabc",
            "rabc_progress_path",
            "rabc_kappa",
            "rabc_epsilon",
            "rabc_head_mode",
        }
        for key in unsupported:
            config_dict.pop(key, None)

        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="act_model_"))
        with open(temp_dir / "config.json", "w") as f:
            json.dump(config_dict, f)
        shutil.copy(model_path_obj / "model.safetensors", temp_dir / "model.safetensors")

        for f in [
            "policy_preprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor.json",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ]:
            if (model_path_obj / f).exists():
                shutil.copy(model_path_obj / f, temp_dir / f)

        self.policy = ACTPolicy.from_pretrained(temp_dir)
        self.policy.to(device)
        self.policy.eval()

        # NOTE: Temporal ensemble disabled. It was causing action collapse to near-zero.
        # The policy falls back to standard action queue behavior (n_action_steps=100).
        print("  Temporal ensemble: DISABLED (using action queue)")

        self.preprocessor = None
        self.postprocessor = None
        self._load_processors(temp_dir)

        self.chunk_size = int(self.policy.config.chunk_size)
        # Allow overriding n_action_steps at inference time without retraining.
        # The model still predicts chunk_size steps, but we only execute n_action_steps.
        self.n_action_steps = int(self.policy.config.n_action_steps)
        self.n_obs_steps = int(self.policy.config.n_obs_steps)

        print("Model loaded successfully!")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Action steps: {self.n_action_steps}")
        print(f"  Observation steps: {self.n_obs_steps}")

        self._load_normalization_stats(dataset_path)
        self.reset()

    def _load_normalization_stats(self, dataset_path: str):
        """Load mean/std stats from dataset for MEAN_STD normalization."""
        stats_path = Path(dataset_path) / "meta" / "stats.json"

        self.action_mean = None
        self.action_std = None
        self.state_mean = None
        self.state_std = None

        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)

                if "action" in stats:
                    self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
                    self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
                    # Prevent division by zero
                    self.action_std = np.where(self.action_std == 0, 1.0, self.action_std)
                    print(f"Loaded action stats: mean shape={self.action_mean.shape}")

                if "observation.state" in stats:
                    self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
                    self.state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
                    self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)
                    print(f"Loaded state stats: mean shape={self.state_mean.shape}")
                else:
                    print("Warning: observation.state stats not found in stats.json")
            except Exception as e:
                print(f"Warning: Failed to load stats.json: {e}")
        else:
            print(f"Warning: stats.json not found at {stats_path}")

        # Fallback: use identity normalization if stats not loaded
        if self.action_mean is None:
            print("Warning: Using identity action normalization")
            self.action_mean = np.zeros(7, dtype=np.float32)
            self.action_std = np.ones(7, dtype=np.float32)
        if self.state_mean is None:
            print("Warning: Using identity state normalization")
            self.state_mean = np.zeros(7, dtype=np.float32)
            self.state_std = np.ones(7, dtype=np.float32)

    def _load_processors(self, temp_dir: Path):
        from lerobot.processor.pipeline import DataProcessorPipeline

        preproc_file = temp_dir / "policy_preprocessor.json"
        if preproc_file.exists():
            try:
                self.preprocessor = DataProcessorPipeline.from_pretrained(
                    temp_dir, config_filename="policy_preprocessor.json"
                )
                print("Loaded preprocessor from model")
            except Exception as e:
                print(f"Warning: Failed to load preprocessor: {e}")

        postproc_file = temp_dir / "policy_postprocessor.json"
        if postproc_file.exists():
            try:
                self.postprocessor = DataProcessorPipeline.from_pretrained(
                    temp_dir, config_filename="policy_postprocessor.json"
                )
                print("Loaded postprocessor from model")
            except Exception as e:
                print(f"Warning: Failed to load postprocessor: {e}")

    def reset(self):
        self.policy.reset()
        self._prev_action = None

    def _prepare_batch(self, obs_dict: dict) -> dict[str, torch.Tensor]:
        """Build input batch for ACT policy from Isaac Sim observations."""
        batch = {}
        use_preprocessor = self.preprocessor is not None

        # ---- Images ----
        for key in ["table_cam", "table_cam_side", "wrist_cam"]:
            if key not in obs_dict:
                print(f"  [WARNING] {key} NOT found in obs_dict!")
                continue
            img = obs_dict[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0

            # CRITICAL FIX: horizontally flip the image to match the training data orientation.
            # The Isaac Sim live rendering produces images that are mirrored left-to-right
            # compared to the encoded training videos. Without this flip, the model sees
            # objects on the opposite side and outputs actions in the wrong direction.
            img = img[:, :, ::-1].copy()

            # Only apply ImageNet normalization manually when there is no preprocessor.
            # The preprocessor's normalizer_processor already handles MEAN_STD normalization.
            if not use_preprocessor:
                img = (img - IMAGENET_MEAN) / IMAGENET_STD

            batch[f"observation.images.{key}"] = torch.from_numpy(img).unsqueeze(0).to(self.device)
            if self.verbose:
                print(
                    f"  [Image] {key}: shape={batch[f'observation.images.{key}'].shape}, "
                    f"range=[{batch[f'observation.images.{key}'].min():.3f}, {batch[f'observation.images.{key}'].max():.3f}]"
                )

        # ---- State ----
        # Construct 18-dim state: eef_pos (3) + eef_quat (4) + gripper_pos (2) + joint_pos (9)
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

        state = np.concatenate([eef_pos, eef_quat, gripper_pos, joint_pos], axis=0).astype(
            np.float32
        )
        expected_dim = self.state_mean.shape[0]
        if state.shape[0] < expected_dim:
            state = np.pad(state, (0, expected_dim - state.shape[0]), mode="constant")
        elif state.shape[0] > expected_dim:
            state = state[:expected_dim]

        # Only normalize state manually when there is no preprocessor.
        if not use_preprocessor:
            state = self._normalize_state(state)

        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(self.device)

        if self.verbose:
            print(f"  [State] shape={state.shape}, range=[{state.min():.3f}, {state.max():.3f}]")

        return batch

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """MEAN_STD unnormalization: x * std + mean"""
        action_unnorm = action * self.action_std + self.action_mean
        # Lock rotation to 0 for IK-Rel
        action_unnorm[3:6] = 0.0
        return action_unnorm.astype(np.float32)

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

    def get_action(self, obs_dict: dict) -> np.ndarray:
        """Get action from policy given current observation."""
        batch = self._prepare_batch(obs_dict)

        # Always log step details to file for debugging
        debug_lines = []
        debug_lines.append("=" * 60)
        debug_lines.append("STEP DEBUG")
        debug_lines.append("=" * 60)
        debug_lines.append(f"Batch keys: {list(batch.keys())}")

        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
            debug_lines.append("Applied preprocessor")
            for k in ["observation.state"]:
                if k in batch:
                    debug_lines.append(
                        f"  {k} after preproc: min={batch[k].min():.4f}, max={batch[k].max():.4f}, mean={batch[k].mean():.4f}"
                    )
            for k in batch:
                if k.startswith("observation.images."):
                    debug_lines.append(
                        f"  {k} after preproc: min={batch[k].min():.4f}, max={batch[k].max():.4f}, mean={batch[k].mean():.4f}"
                    )

        with torch.no_grad():
            raw_action = self.policy.select_action(batch).squeeze().cpu().numpy()

        debug_lines.append(f"RAW action from model: {raw_action}")

        if self.postprocessor is not None:
            action_dict = {"action": torch.from_numpy(raw_action).unsqueeze(0)}
            action_dict = self.postprocessor(action_dict)
            action = action_dict["action"].squeeze().cpu().numpy()
            debug_lines.append("Applied postprocessor")
        else:
            action = self._unnormalize_action(raw_action)
            debug_lines.append("Used manual unnormalize")

        debug_lines.append(f"Action BEFORE smoothing: {action}")

        # NOTE: Temporal smoothing removed to avoid lag/response sluggishness.
        # Previously: action[:6] = 0.85 * action[:6] + 0.15 * self._prev_action[:6]

        # Gripper thresholding: binary open/close based on sign.
        action[6] = 1.0 if action[6] > 0.0 else -1.0

        self._prev_action = action.copy()

        debug_lines.append(f"FINAL action: {action}")
        debug_lines.append(f"dz = {action[2]:.6f} (negative=down, positive=up)")

        with open("/tmp/act_step_debug.log", "a") as f:
            f.write("\n".join(debug_lines) + "\n")

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
    output_dir = output_root / f"act_inference_{run_timestamp}"
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

    policy = ACTPolicyWrapper(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device=args.device,
        verbose=args.verbose,
    )
    # Override n_action_steps for more frequent re-planning
    if args.n_action_steps != policy.n_action_steps:
        print(f"Overriding n_action_steps: {policy.n_action_steps} -> {args.n_action_steps}")
        policy.n_action_steps = args.n_action_steps
        policy.policy.config.n_action_steps = args.n_action_steps

    print("\n" + "=" * 70)
    print("ACT Place-Bin Inference - Franka on Isaac Sim")
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
