#!/usr/bin/env python3
"""
GR00T-N1.5 Order Generalization Evaluation Script (Remote Client Mode)

Evaluates the model's ability to follow different color ordering instructions
across all 6 permutations of {blue, red, green}. Connects to a GR00T inference
server via ZMQ (runs in isaac_gr00t env).

Usage:
    # Terminal 1 (isaac_gr00t environment):
    conda activate isaac_gr00t
    cd $GR00T_PATH
    python scripts/inference_service.py \
        --model-path ./pretrained_models/gr00t_place_bin \
        --server --port 5555

    # Terminal 2 (isaac environment):
    conda activate isaac
    cd VLA-Franka-IsaacLab
    python scripts/inference/eval_gr00t_order_generalization.py \
        --server-host localhost \
        --server-port 5555 \
        --episodes-per-order 10 \
        --episode-length 800 \
        --headless \
        --save-video \
        --output-dir ./outputs/order_eval \
        --seed 1000
"""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLOR_ORDERS: List[Tuple[str, str, str]] = [
    ("blue", "red", "green"),
    ("blue", "green", "red"),
    ("red", "blue", "green"),
    ("red", "green", "blue"),
    ("green", "blue", "red"),
    ("green", "red", "blue"),
]

COLOR_TO_CUBE = {
    "blue": "cube_1",
    "red": "cube_2",
    "green": "cube_3",
}

CUBE_TO_COLOR = {v: k for k, v in COLOR_TO_CUBE.items()}

GRASP_DIST_THRESHOLD = 0.10      # EEF 到方块距离
LIFT_Z_THRESHOLD = 0.03          # 方块离桌高度（只要比桌面初始高度高一点就算提起）
BIN_XY_THRESHOLD = 0.12          # bin 水平范围（放宽，允许放在边缘）
BIN_Z_MIN = -0.02                # 允许方块略低于 bin 上沿
BIN_Z_MAX = 0.25                 # 允许叠放后更高的 Z
GRIPPER_OPEN_THRESHOLD = 0.035
GRIPPER_CLOSED_THRESHOLD = 0.02

COLOR_ANNOTATIONS = {
    "blue": (255, 100, 50),
    "red": (50, 50, 255),
    "green": (50, 200, 50),
    "white": (255, 255, 255),
    "yellow": (50, 255, 255),
}

DEFAULT_TASK_DESCRIPTION = (
    "pick up the blue, red and green cubes in order and stack them in the blue bin"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    order_prompt: str
    episode: int
    actual_order: List[str] = field(default_factory=list)
    stage1_ok: bool = False
    stage2_ok: bool = False
    stage3_ok: bool = False
    stack_ok: bool = False
    order_correct: bool = False
    steps: int = 0
    grasp_events: List[Dict[str, Any]] = field(default_factory=list)
    place_events: List[Dict[str, Any]] = field(default_factory=list)
    color_misgrasp: bool = False


@dataclass
class OrderSummary:
    order_prompt: str
    episodes: int = 0
    stage_1_success: int = 0
    stage_2_success: int = 0
    stage_3_success: int = 0
    order_correct: int = 0
    stack_success: int = 0
    color_misgrasp: int = 0
    total_steps: int = 0


# ---------------------------------------------------------------------------
# Remote client classes (adapted from gr00t_remote_client.py)
# ---------------------------------------------------------------------------
class MsgSerializer:
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
    def __init__(
        self,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        verbose: bool = False,
        flip_images: bool = False,
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
        gr00t_obs["annotation.human.action.task_description"] = [self.task_description]
        return gr00t_obs


class ActionPostProcessor:
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
        if self._action_chunk is None:
            raise RuntimeError("No action chunk available")
        if self._chunk_step >= len(self._action_chunk):
            self._chunk_step = len(self._action_chunk) - 1
        action = self._action_chunk[self._chunk_step]
        self._chunk_step += 1
        action[3:6] = 0.0
        action[6] = float(np.clip(action[6], -1.0, 1.0))
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
        if self._action_chunk is None:
            return False
        return self._chunk_step < len(self._action_chunk)


# ---------------------------------------------------------------------------
# Episode Tracker
# ---------------------------------------------------------------------------
class EpisodeTracker:
    def __init__(self, env, expected_order: Tuple[str, str, str]):
        self.env = env
        self.expected_order = expected_order
        self.grasped: Dict[str, bool] = {"cube_1": False, "cube_2": False, "cube_3": False}
        self.lifted: Dict[str, bool] = {"cube_1": False, "cube_2": False, "cube_3": False}
        self.placed: Dict[str, bool] = {"cube_1": False, "cube_2": False, "cube_3": False}
        self.actual_order: List[str] = []
        self.grasp_events: List[Dict[str, Any]] = []
        self.place_events: List[Dict[str, Any]] = []
        self.stage_success = [False, False, False]
        self.color_misgrasp = False
        self._bin_pos = None
        self._prev_gripper_open = True
        self.grasp_counter: Dict[str, int] = {"cube_1": 0, "cube_2": 0, "cube_3": 0}
        self.place_counter: Dict[str, int] = {"cube_1": 0, "cube_2": 0, "cube_3": 0}
        # 记录每个方块的初始 Z 高度（用于高度突变检测）
        self.initial_z: Dict[str, float] = {
            cn: float(self._get_scene_pos(cn)[2])
            for cn in ["cube_1", "cube_2", "cube_3"]
        }

    def _get_scene_pos(self, name: str) -> np.ndarray:
        obj = self.env.scene[name]
        return obj.data.root_pos_w[0].cpu().numpy()

    def _get_eef_pos(self, obs_dict: dict) -> np.ndarray:
        import torch
        val = obs_dict["eef_pos"]
        if isinstance(val, torch.Tensor):
            val = val[0].cpu().numpy()
        return val.flatten()[:3].astype(np.float32)

    def _get_gripper_state(self, obs_dict: dict) -> Tuple[bool, float]:
        """Returns (is_open, avg_value)."""
        import torch
        val = obs_dict.get("gripper_pos", None)
        if val is None:
            return True, 1.0
        if isinstance(val, torch.Tensor):
            val = val[0].cpu().numpy()
        val = val.flatten()[:2].astype(np.float32)
        avg = float(np.mean(val))
        if avg > 0.4:
            return avg > 0.5, avg
        return avg > GRIPPER_OPEN_THRESHOLD, avg

    def update(self, step: int, obs_dict: dict):
        eef_pos = self._get_eef_pos(obs_dict)
        gripper_open, gripper_avg = self._get_gripper_state(obs_dict)
        bin_pos = self._get_scene_pos("blue_sorting_bin")
        self._bin_pos = bin_pos

        # ---------- 抓取检测：用"高度突变 + 夹爪闭合"判断 ----------
        for cube_name in ["cube_1", "cube_2", "cube_3"]:
            if self.lifted[cube_name]:
                continue
            cube_pos = self._get_scene_pos(cube_name)
            z_delta = cube_pos[2] - self.initial_z[cube_name]

            # 夹爪闭合判断
            if gripper_avg > 0.4:
                gripper_closed = gripper_avg < 0.5
            else:
                gripper_closed = gripper_avg < 0.05

            # 抓取条件：方块比初始位置明显抬高（> 0.015m）且夹爪闭合
            if z_delta > 0.015 and gripper_closed:
                self.grasp_counter[cube_name] += 1
                if self.grasp_counter[cube_name] >= 3:
                    self.lifted[cube_name] = True
                    self.grasped[cube_name] = True
                    color = CUBE_TO_COLOR[cube_name]
                    self.actual_order.append(color)
                    self.grasp_events.append({
                        "step": step,
                        "color": color,
                        "cube_name": cube_name,
                        "eef_pos": eef_pos.copy().tolist(),
                        "cube_pos": cube_pos.copy().tolist(),
                    })
                    stage_idx = len(self.actual_order) - 1
                    if stage_idx < 3:
                        if color != self.expected_order[stage_idx]:
                            self.color_misgrasp = True
            else:
                self.grasp_counter[cube_name] = 0

        # ---------- 放置检测 ----------
        for cube_name in ["cube_1", "cube_2", "cube_3"]:
            if self.lifted[cube_name] and not self.placed[cube_name]:
                cube_pos = self._get_scene_pos(cube_name)
                color = CUBE_TO_COLOR[cube_name]
                xy_dist = np.linalg.norm(cube_pos[:2] - bin_pos[:2])
                z_rel = cube_pos[2] - bin_pos[2]
                in_bin = xy_dist < BIN_XY_THRESHOLD and BIN_Z_MIN < z_rel < BIN_Z_MAX
                if in_bin:
                    self.place_counter[cube_name] += 1
                    if self.place_counter[cube_name] >= 5:
                        self.placed[cube_name] = True
                        self.place_events.append({
                            "step": step,
                            "color": color,
                            "cube_name": cube_name,
                            "cube_pos": cube_pos.copy().tolist(),
                        })
                        try:
                            stage_idx = self.expected_order.index(color)
                            self.stage_success[stage_idx] = True
                        except ValueError:
                            pass
                else:
                    self.place_counter[cube_name] = 0

        self._prev_gripper_open = gripper_open

    def get_result(self, total_steps: int) -> EpisodeResult:
        order_prompt = ",".join(self.expected_order)
        order_correct = self.actual_order == list(self.expected_order)

        cube_positions = {}
        for cube_name in ["cube_1", "cube_2", "cube_3"]:
            cube_positions[cube_name] = self._get_scene_pos(cube_name)

        bin_pos = self._get_scene_pos("blue_sorting_bin")
        all_in_bin = True
        for cube_name in ["cube_1", "cube_2", "cube_3"]:
            cp = cube_positions[cube_name]
            xy_dist = np.linalg.norm(cp[:2] - bin_pos[:2])
            z_rel = cp[2] - bin_pos[2]
            if not (xy_dist < BIN_XY_THRESHOLD and BIN_Z_MIN < z_rel < BIN_Z_MAX):
                all_in_bin = False
                break

        c1, c2, c3 = cube_positions["cube_1"], cube_positions["cube_2"], cube_positions["cube_3"]
        xy_aligned = (
            np.linalg.norm(c1[:2] - c2[:2]) < 0.03
            and np.linalg.norm(c2[:2] - c3[:2]) < 0.03
        )
        z_1_2 = c2[2] - c1[2]
        z_2_3 = c3[2] - c2[2]
        z_stacked = 0.025 < z_1_2 < 0.075 and 0.025 < z_2_3 < 0.075

        # 必须真正发生过抓取且三个都放置了才算 stack_ok
        all_placed = all(self.placed.values())
        stack_ok = len(self.actual_order) == 3 and all_placed and all_in_bin and xy_aligned and z_stacked

        return EpisodeResult(
            order_prompt=order_prompt,
            episode=0,
            actual_order=self.actual_order.copy(),
            stage1_ok=self.stage_success[0],
            stage2_ok=self.stage_success[1],
            stage3_ok=self.stage_success[2],
            stack_ok=stack_ok,
            order_correct=order_correct,
            steps=total_steps,
            grasp_events=self.grasp_events.copy(),
            place_events=self.place_events.copy(),
            color_misgrasp=self.color_misgrasp,
        )


# ---------------------------------------------------------------------------
# Keyframe helpers
# ---------------------------------------------------------------------------
def annotate_keyframe(
    frame: np.ndarray,
    step: int,
    eef_pos: np.ndarray,
    cube_positions: Dict[str, np.ndarray],
    stage_text: str,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    def put_text(text, pos, color=COLOR_ANNOTATIONS["white"]):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    put_text(f"Step: {step}", (10, 25))
    put_text(f"EEF: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]", (10, 45))

    y_offset = h - 20
    for cube_name, color in [("cube_1", "blue"), ("cube_2", "red"), ("cube_3", "green")]:
        pos = cube_positions[cube_name]
        put_text(
            f"{color}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
            (10, y_offset),
            COLOR_ANNOTATIONS[color],
        )
        y_offset -= 20

    put_text(stage_text, (w - 300, h - 20), COLOR_ANNOTATIONS["yellow"])
    return img


def save_keyframe(
    frame: np.ndarray,
    output_dir: Path,
    order_slug: str,
    episode: int,
    event_type: str,
    color: str,
    step: int,
    eef_pos: np.ndarray,
    cube_positions: Dict[str, np.ndarray],
    stage_text: str,
):
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    annotated = annotate_keyframe(frame, step, eef_pos, cube_positions, stage_text)
    fname = f"{order_slug}_ep{episode:03d}_{event_type}_{color}.png"
    cv2.imwrite(str(keyframes_dir / fname), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


def save_final_keyframe(
    frame: np.ndarray,
    output_dir: Path,
    order_slug: str,
    episode: int,
    step: int,
    eef_pos: np.ndarray,
    cube_positions: Dict[str, np.ndarray],
    result: EpisodeResult,
):
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    stage_text = (
        f"Stack: {'OK' if result.stack_ok else 'FAIL'} | "
        f"Order: {'OK' if result.order_correct else 'FAIL'} | "
        f"Actual: {result.actual_order}"
    )
    annotated = annotate_keyframe(frame, step, eef_pos, cube_positions, stage_text)
    fname = f"{order_slug}_ep{episode:03d}_final.png"
    cv2.imwrite(str(keyframes_dir / fname), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GR00T Order Generalization Evaluation")
    parser.add_argument("--task", type=str, default="Isaac-Place-Bin-Franka-IK-Rel-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--episodes-per-order", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--server-host", type=str, default="localhost")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/order_eval",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--orders",
        type=str,
        default="all",
        help='e.g. "blue,red,green;red,blue,green" or "all"',
    )
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--action-smoothing", type=float, default=0.3)
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Enable horizontal image flip. Only use this if your training data WAS flipped.",
    )
    return parser.parse_args()


def build_task_description(order: Tuple[str, str, str]) -> str:
    c1, c2, c3 = order
    return f"pick up the {c1}, {c2} and {c3} cubes in order and stack them in the blue bin"


def parse_orders_arg(orders_str: str) -> List[Tuple[str, str, str]]:
    if orders_str.lower() == "all":
        return COLOR_ORDERS
    result = []
    for part in orders_str.split(";"):
        colors = [c.strip().lower() for c in part.split(",")]
        if len(colors) == 3:
            result.append(tuple(colors))  # type: ignore
    return result if result else COLOR_ORDERS


def format_order_slug(order: Tuple[str, str, str]) -> str:
    return "_".join(order)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_evaluation():
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"order_eval_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    orders_to_test = parse_orders_arg(args.orders)
    print(f"Testing {len(orders_to_test)} color order(s):")
    for o in orders_to_test:
        print(f"  {' -> '.join(o)}")

    # Connect to GR00T server first
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

    # Launch Isaac Sim
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

    task_name = args.task
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.concatenate_terms = False
    env = gym.make(task_name, cfg=env_cfg)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    all_results: List[EpisodeResult] = []
    order_summaries: Dict[str, OrderSummary] = {}

    for order_idx, order in enumerate(orders_to_test):
        order_seed = args.seed + order_idx * 1000
        np.random.seed(order_seed)

        task_desc = build_task_description(order)
        order_slug = format_order_slug(order)

        print(f"\n{'=' * 70}")
        print(f"Order {order_idx + 1}/{len(orders_to_test)}: {' -> '.join(order)}")
        print(f"Prompt: {task_desc}")
        print(f"Seed: {order_seed}")
        print(f"{'=' * 70}")

        obs_converter = IsaacObsConverter(
            task_description=task_desc,
            verbose=args.verbose,
            flip_images=args.flip,
        )
        action_processor = ActionPostProcessor(
            action_horizon=args.action_horizon,
            verbose=args.verbose,
            action_smoothing=args.action_smoothing,
        )

        summary = OrderSummary(order_prompt=",".join(order))

        for ep in range(args.episodes_per_order):
            obs, info = env.reset()
            done = False
            step = 0
            obs_dict = obs["policy"]

            tracker = EpisodeTracker(env.unwrapped, expected_order=order)
            action_processor.reset()
            video_frames = []

            while step < args.episode_length and not done and simulation_app.is_running():
                if not action_processor.has_more_actions():
                    gr00t_obs = obs_converter.convert(obs_dict)
                    action_dict = client.get_action(gr00t_obs)
                    action_processor.process(action_dict)

                action = action_processor._get_next_action()
                action_tensor = torch.from_numpy(action).float().unsqueeze(0).to("cuda:0")
                obs, reward, terminated, truncated, info = env.step(action_tensor)

                if args.save_video:
                    img = obs["policy"]["table_cam"][0].cpu().numpy()
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    video_frames.append(img)

                obs_dict = obs["policy"]
                tracker.update(step, obs_dict)

                # Keyframes on events (disabled — only final frame is saved)

                if terminated.ndim > 0:
                    done = bool(
                        terminated[0].item()
                        if isinstance(terminated[0], torch.Tensor)
                        else terminated[0]
                    )
                else:
                    done = bool(terminated)

                step += 1

            # Final result
            result = tracker.get_result(step)
            result.episode = ep + 1
            all_results.append(result)

            summary.episodes += 1
            if result.stage1_ok:
                summary.stage_1_success += 1
            if result.stage2_ok:
                summary.stage_2_success += 1
            if result.stage3_ok:
                summary.stage_3_success += 1
            if result.order_correct:
                summary.order_correct += 1
            if result.stack_ok:
                summary.stack_success += 1
            if result.color_misgrasp:
                summary.color_misgrasp += 1
            summary.total_steps += step

            print(
                f"  Ep {ep + 1}/{args.episodes_per_order}: "
                f"steps={step} | "
                f"actual_order={result.actual_order} | "
                f"stages=[{int(result.stage1_ok)},{int(result.stage2_ok)},{int(result.stage3_ok)}] | "
                f"order_ok={result.order_correct} | "
                f"stack_ok={result.stack_ok} | "
                f"misgrasp={result.color_misgrasp}"
            )

            # Final keyframe
            if args.save_video and video_frames:
                eef_pos = obs_dict["eef_pos"][0].cpu().numpy().flatten()[:3]
                cube_positions = {
                    cn: env.unwrapped.scene[cn].data.root_pos_w[0].cpu().numpy()
                    for cn in ["cube_1", "cube_2", "cube_3"]
                }
                save_final_keyframe(
                    video_frames[-1], output_dir, order_slug, ep + 1, step,
                    eef_pos, cube_positions, result,
                )

            # Save video
            if args.save_video and video_frames:
                videos_dir = output_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"{order_slug}_ep{ep + 1:03d}.mp4"
                h, w = video_frames[0].shape[:2]
                out = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                )
                for frame in video_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"    Video saved: {video_path}")

        order_summaries[order_slug] = summary

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    json_results = {}
    for slug, s in order_summaries.items():
        json_results[slug] = {
            "episodes": s.episodes,
            "stage_1_success": s.stage_1_success,
            "stage_1_success_rate": s.stage_1_success / s.episodes if s.episodes else 0,
            "stage_2_success": s.stage_2_success,
            "stage_2_success_rate": s.stage_2_success / s.episodes if s.episodes else 0,
            "stage_3_success": s.stage_3_success,
            "stage_3_success_rate": s.stage_3_success / s.episodes if s.episodes else 0,
            "order_correct": s.order_correct,
            "order_accuracy": s.order_correct / s.episodes if s.episodes else 0,
            "stack_success": s.stack_success,
            "stack_success_rate": s.stack_success / s.episodes if s.episodes else 0,
            "color_misgrasp": s.color_misgrasp,
            "color_misgrasp_rate": s.color_misgrasp / s.episodes if s.episodes else 0,
            "avg_steps": s.total_steps / s.episodes if s.episodes else 0,
        }

    json_path = output_dir / "order_eval_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON results saved: {json_path}")

    csv_path = output_dir / "order_eval_episodes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "order_prompt", "episode", "expected_order", "actual_order",
            "stage1_ok", "stage2_ok", "stage3_ok",
            "order_correct", "stack_ok", "color_misgrasp", "steps"
        ])
        for r in all_results:
            writer.writerow([
                r.order_prompt, r.episode, r.order_prompt,
                ",".join(r.actual_order),
                r.stage1_ok, r.stage2_ok, r.stage3_ok,
                r.order_correct, r.stack_ok, r.color_misgrasp, r.steps
            ])
    print(f"CSV episodes saved: {csv_path}")

    summary_path = output_dir / "order_eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GR00T Order Generalization Evaluation Summary\n")
        f.write(f"Timestamp: {run_timestamp}\n")
        f.write(f"Server: {args.server_host}:{args.server_port}\n")
        f.write(f"Episodes per order: {args.episodes_per_order}\n")
        f.write(f"Episode length: {args.episode_length}\n")
        f.write("=" * 70 + "\n\n")

        for slug, s in order_summaries.items():
            n = s.episodes
            f.write(f"Order: {slug.replace('_', ' -> ')}\n")
            f.write(f"  Episodes: {n}\n")
            f.write(f"  Stage 1 success: {s.stage_1_success}/{n} ({s.stage_1_success/n*100:.1f}%)\n")
            f.write(f"  Stage 2 success: {s.stage_2_success}/{n} ({s.stage_2_success/n*100:.1f}%)\n")
            f.write(f"  Stage 3 success: {s.stage_3_success}/{n} ({s.stage_3_success/n*100:.1f}%)\n")
            f.write(f"  Order correct:   {s.order_correct}/{n} ({s.order_correct/n*100:.1f}%)\n")
            f.write(f"  Stack success:   {s.stack_success}/{n} ({s.stack_success/n*100:.1f}%)\n")
            f.write(f"  Color misgrasp:  {s.color_misgrasp}/{n} ({s.color_misgrasp/n*100:.1f}%)\n")
            f.write(f"  Avg steps:       {s.total_steps/n:.1f}\n")
            f.write("\n")

    print(f"Text summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    for slug, s in order_summaries.items():
        n = s.episodes
        print(f"\n{slug.replace('_', ' -> ')}:")
        print(f"  Stage 1: {s.stage_1_success/n*100:.1f}% | "
              f"Stage 2: {s.stage_2_success/n*100:.1f}% | "
              f"Stage 3: {s.stage_3_success/n*100:.1f}%")
        print(f"  Order accuracy: {s.order_correct/n*100:.1f}% | "
              f"Stack success: {s.stack_success/n*100:.1f}%")

    client.close()
    env.close()
    simulation_app.close()
    print("\nDone!")


if __name__ == "__main__":
    run_evaluation()
