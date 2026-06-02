#!/usr/bin/env python3
"""
GR00T-N1.5 Remote Inference Client for Isaac Sim

This script runs in the `isaac` environment (with IsaacLab) and connects
to a GR00T inference server running in the `isaac_gr00t` environment via ZMQ.

Architecture:
    Terminal 1 (isaac_gr00t)          Terminal 2 (isaac)
    ┌──────────────────────┐          ┌──────────────────────────┐
    │  inference_service   │ ◄─ ZMQ ──┤  gr00t_remote_client.py  │
    │  (GR00T model)       │          │  (IsaacLab + Franka env) │
    └──────────────────────┘          └──────────────────────────┘

Prerequisites:
    # (1) GR00T model checkpoint (e.g., pretrained_models/gr00t_place_bin)
    # (2) IsaacLab + GR00T inference dependencies in their respective conda envs

Quick Start:
    # Terminal 1 (isaac_gr00t environment):
    conda activate isaac_gr00t
    cd $GR00T_PATH
    python scripts/inference_service.py \
        --model-path ./pretrained_models/gr00t_place_bin \
        --server --port 5555

    # Terminal 2 (isaac environment):
    conda activate isaac
    cd vla-franka-isaaclab
    python scripts/inference/gr00t_remote_client.py \
        --server-host localhost \
        --server-port 5555 \
        --num-episodes 3 \
        --headless --save-video --no-flip --episode-length 800

Dependencies:
    # isaac environment:
    pip install zmq msgpack numpy

Task Descriptions (--task-description):
    # Single cube pick-and-place:
    "pick up the green cube and place it into the blue bin"
    "pick up the red cube and place it into the blue bin"
    # Multi-cube sequential stacking:
    "pick up the blue, red and green cubes in order and stack them in the blue bin"

Key Arguments (gr00t_remote_client.py):
    --task              IsaacLab task name (default: Isaac-Place-Bin-Franka-IK-Rel-v0)
    --server-host       GR00T server hostname (default: localhost)
    --server-port       GR00T server port (default: 5555)
    --headless          Run Isaac Sim without GUI
    --num-episodes      Number of evaluation episodes (default: 1)
    --episode-length    Max steps per episode (default: 300)
    --num-envs          Number of parallel environments (default: 1)
    --seed              Random seed (default: 1000)
    --save-video        Save camera videos per episode
    --output-dir        Video output directory (default: ./outputs/GR00T)
    --no-flip           Disable horizontal image flip (use when training data was NOT flipped)
    --action-horizon    GR00T action chunk size — smaller = more frequent re-planning (default: 4)
    --action-smoothing  EMA smoothing factor 0.0–0.9 (default: 0.3)
    --task-description  Language instruction for policy conditioning
    --verbose           Enable per-step debug logging

Advanced Usage Examples:
    # (A) Interactive debugging with GUI and verbose logging:
    python scripts/inference/gr00t_remote_client.py \
        --server-host localhost --server-port 5555 \
        --num-episodes 1 --episode-length 200 \
        --action-horizon 4 --verbose

    # (B) Headless evaluation with video recording (low smoothing for precision):
    python scripts/inference/gr00t_remote_client.py \
        --server-host localhost --server-port 5555 \
        --num-episodes 10 --episode-length 800 \
        --headless --save-video --no-flip \
        --action-horizon 4 --action-smoothing 0.1

    # (C) Multi-cube stacking task:
    python scripts/inference/gr00t_remote_client.py \
        --server-host localhost --server-port 5555 \
        --task-description "pick up the blue, red and green cubes in order and stack them in the blue bin" \
        --headless --save-video --no-flip --episode-length 1200

    # (D) Remote server:
    python scripts/inference/gr00t_remote_client.py \
        --server-host 10.0.0.100 --server-port 5555 \
        --headless --save-video --num-episodes 5
"""

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASK_DESCRIPTION = (
    "pick up the green cube and place it into the blue bin"
)
# pick up the red cube and place it into the blue bin
# pick up the blue, red and green cubes in order and stack them in the blue bin

class MsgSerializer:
    """Serialize/deserialize messages with msgpack + numpy support."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        import msgpack

        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        import msgpack

        return msgpack.unpackb(data, object_hook=MsgSerializer._decode, raw=False)

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and obj.get("__ndarray_class__"):
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class Gr00tRemoteClient:
    """ZMQ client for remote GR00T inference."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 30000,
        api_token: Optional[str] = None,
    ):
        import zmq

        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[Client] Connected to GR00T server at tcp://{host}:{port}")

        self.modality_config = self._call_endpoint("get_modality_config", requires_input=False)
        print(f"[Client] Modality config received: {list(self.modality_config.keys())}")

    def _call_endpoint(
        self, endpoint: str, data: Optional[dict] = None, requires_input: bool = True
    ) -> dict:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return self._call_endpoint("get_action", observation)

    def ping(self) -> dict:
        return self._call_endpoint("ping", requires_input=False)

    def close(self):
        self.socket.close()
        self.context.term()


class IsaacObsConverter:
    """Convert Isaac Sim observations to GR00T format."""

    def __init__(
        self,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        verbose: bool = False,
        flip_images: bool = True,
    ):
        self.task_description = task_description
        self.verbose = verbose
        self.flip_images = flip_images

    def _extract_numpy(self, obs_dict: dict, key: str, expected_dim: int) -> Optional[np.ndarray]:
        import torch

        if key not in obs_dict:
            return None
        val = obs_dict[key]
        if isinstance(val, torch.Tensor):
            val = val[0].cpu().numpy() if val.dim() > 1 else val.cpu().numpy()
        val = val.flatten()[:expected_dim]
        if val.shape[0] < expected_dim:
            val = np.pad(val, (0, expected_dim - val.shape[0]), mode="constant")
        return val.astype(np.float32)

    def convert(self, obs_dict: dict) -> Dict[str, Any]:
        gr00t_obs = {}

        for cam_key in ["table_cam", "table_cam_side", "wrist_cam"]:
            if cam_key not in obs_dict:
                if self.verbose:
                    print(f"  [WARNING] {cam_key} NOT found in obs_dict!")
                continue

            import torch

            img = obs_dict[cam_key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            if img.ndim == 4:
                img = img[0]

            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            if self.flip_images:
                img = img[:, ::-1, :].copy()
            gr00t_obs[f"video.{cam_key}"] = np.expand_dims(img, axis=0)

            if self.verbose:
                print(
                    f"  [Image] {cam_key}: shape={gr00t_obs[f'video.{cam_key}'].shape}, "
                    f"dtype={gr00t_obs[f'video.{cam_key}'].dtype}"
                )

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

        end_effector = np.concatenate([eef_pos, eef_quat], axis=0).astype(np.float32)
        gr00t_obs["state.end_effector"] = np.expand_dims(end_effector, axis=0)
        gr00t_obs["state.fingers"] = np.expand_dims(gripper_pos, axis=0)
        gr00t_obs["state.joints"] = np.expand_dims(joint_pos, axis=0)

        if self.verbose:
            print(f"  [State] end_effector shape={end_effector.shape}")
            print(f"  [State] fingers shape={gripper_pos.shape}")
            print(f"  [State] joints shape={joint_pos.shape}")

        gr00t_obs["annotation.human.action.task_description"] = [self.task_description]

        return gr00t_obs


class ActionPostProcessor:
    """Post-process GR00T action output for Isaac Sim."""

    def __init__(
        self,
        action_horizon: int = 16,
        verbose: bool = False,
        action_smoothing: float = 0.0,
    ):
        self.action_horizon = action_horizon
        self.verbose = verbose
        self.action_smoothing = action_smoothing
        self._action_chunk = None
        self._chunk_step = 0
        self._prev_action = None

    def reset(self):
        """Reset action chunk cache."""
        self._action_chunk = None
        self._chunk_step = 0
        self._prev_action = None

    def process(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        action_key = "action.full_action"
        if action_key not in action_dict:
            raise KeyError(
                f"Expected action key '{action_key}' not found. Got: {list(action_dict.keys())}"
            )

        action = action_dict[action_key]

        if action.ndim == 3:
            action = action[0]

        self._action_chunk = action
        self._chunk_step = 0

        if self.verbose:
            print(f"  [GR00T] New action chunk shape: {action.shape}")

        return self._get_next_action()

    def _get_next_action(self) -> np.ndarray:
        """Get next action from current chunk."""
        if self._action_chunk is None:
            raise RuntimeError("No action chunk available")

        if self._chunk_step >= len(self._action_chunk):
            self._chunk_step = len(self._action_chunk) - 1

        action = self._action_chunk[self._chunk_step]
        self._chunk_step += 1

        action[3:6] = 0.0
        action[6] = float(np.clip(action[6], -1.0, 1.0))

        # Action smoothing via exponential moving average
        if self.action_smoothing > 0.0 and self._prev_action is not None:
            action = (
                self.action_smoothing * self._prev_action + (1.0 - self.action_smoothing) * action
            )

        self._prev_action = action.copy()

        if self.verbose:
            print(
                f"  [Action] dx={action[0]:.4f}, dy={action[1]:.4f}, dz={action[2]:.4f}, "
                f"gripper={action[6]:.1f}"
            )

        return action.astype(np.float32)

    def has_more_actions(self) -> bool:
        """Check if current chunk has more actions."""
        if self._action_chunk is None:
            return False
        return self._chunk_step < len(self._action_chunk)


def parse_args():
    parser = argparse.ArgumentParser(description="GR00T-N1.5 Remote Inference Client for Isaac Sim")
    parser.add_argument("--task", type=str, default="Isaac-Place-Bin-Franka-IK-Rel-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--server-host", type=str, default="localhost", help="GR00T inference server host"
    )
    parser.add_argument("--server-port", type=int, default=5555, help="GR00T inference server port")
    parser.add_argument(
        "--task-description",
        type=str,
        default=DEFAULT_TASK_DESCRIPTION,
        help="Language task description for policy conditioning",
    )
    parser.add_argument("--save-video", action="store_true", help="Save episode videos")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/GR00T",
        help="Output directory for videos",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=4,
        help="GR00T action chunk size. Smaller = more frequent re-planning, better precision (default: 4)",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.3,
        help="EMA smoothing factor for actions. 0.0 = no smoothing, 0.9 = heavy smoothing (default: 0.3)",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable horizontal image flip. Use this if training data was NOT flipped.",
    )
    return parser.parse_args()


def run_inference():
    args = parse_args()

    output_dir = None
    if args.save_video:
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"gr00t_remote_inference_{run_timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 70}\n")

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("Connecting to GR00T Inference Server")
    print("=" * 70)
    client = Gr00tRemoteClient(
        host=args.server_host,
        port=args.server_port,
        timeout_ms=30000,
    )

    ping_resp = client.ping()
    print(f"[Client] Server health: {ping_resp}")

    print("\n" + "=" * 70)
    print("Launching Isaac Sim")
    print("=" * 70)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from tasks.franka import place_bin_ik_rel_env_cfg  # noqa: F401

    obs_converter = IsaacObsConverter(
        task_description=args.task_description,
        verbose=args.verbose,
        flip_images=not args.no_flip,
    )
    action_horizon = args.action_horizon
    action_processor = ActionPostProcessor(
        action_horizon=action_horizon,
        verbose=args.verbose,
        action_smoothing=args.action_smoothing,
    )
    print(f"[Client] Action horizon (chunk size): {action_horizon}")
    print(f"[Client] Action smoothing: {args.action_smoothing}")

    print("\n" + "=" * 70)
    print("GR00T-N1.5 Remote Inference - Franka on Isaac Sim")
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

    import torch

    env.unwrapped.sim.reset()
    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        obs_dict = obs["policy"]
        eef_pos_0 = obs_dict["eef_pos"][0].cpu().numpy()
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        print(f"  Step 0 EEF pos: [{eef_pos_0[0]:.4f}, {eef_pos_0[1]:.4f}, {eef_pos_0[2]:.4f}]")

        video_frames = []
        video_frames_side = []
        video_frames_wrist = []

        action_processor.reset()
        while step < args.episode_length and not done and simulation_app.is_running():
            if not action_processor.has_more_actions():
                gr00t_obs = obs_converter.convert(obs_dict)
                action_dict = client.get_action(gr00t_obs)
                action_processor.process(action_dict)

            action = action_processor._get_next_action()
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to("cuda:0")

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
            else:
                done = bool(terminated)

            obs_dict = obs["policy"]
            step += 1

            if step % 50 == 0:
                eef_pos = obs_dict["eef_pos"][0].cpu().numpy()
                print(
                    f"  Step {step} | EEF pos: [{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]"
                )

        episode_lengths.append(step)
        print(f"  Episode finished after {step} steps")

        if args.save_video and video_frames:
            import cv2

            for frames, name in [
                (video_frames, "table_cam"),
                (video_frames_side, "table_cam_side"),
                (video_frames_wrist, "wrist_cam"),
            ]:
                if frames:
                    video_path = output_dir / f"ep{ep + 1:03d}_{name}.mp4"
                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"  Saved video: {video_path}")

    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)
    print(f"Episode lengths: {episode_lengths}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    if output_dir:
        print(f"Output directory: {output_dir}")

    client.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_inference()
