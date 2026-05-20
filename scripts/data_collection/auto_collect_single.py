#!/usr/bin/env python3
"""
自动化数据采集脚本 - 机械臂自动执行 pick-place 任务
直接Output LeRobot v3.0 格式数据集

使用方法:
    python scripts/data_collection/auto_collect_single.py \
        --num_episodes 10 \
        --output_dir datasets/lerobot/auto_collected_red \
        --repo_id local/franka_place_bin \
        --use_videos \
        --overwrite  \
        --target_color red 

特点:
    - 自动生成平滑轨迹
    - 记录连续动作数据
    - 直接Output LeRobot 格式（无需后续转换）
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# LeRobot import (try installed first, then fallback to local src)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    _lerobot_path = str(project_root / "lerobot" / "src")
    if os.path.exists(_lerobot_path) and _lerobot_path not in sys.path:
        sys.path.insert(0, _lerobot_path)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Automated data collection for pick-place task")
parser.add_argument("--task", type=str, default="Isaac-Place-Bin-Franka-IK-Rel-v0")
parser.add_argument("--num_episodes", type=int, default=10)

# LeRobot dataset output
parser.add_argument(
    "--output_dir",
    type=str,
    default="datasets/lerobot/auto_collected",
    help="Output LeRobot 数据集目录",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="local/franka_place_bin",
    help="HuggingFace-style repo ID",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="pick up cubes and place them into the blue bin",
    help="任务语言描述",
)
parser.add_argument("--fps", type=int, default=30, help="录制帧率")
parser.add_argument(
    "--use_videos",
    action="store_true",
    default=False,
    help="将图像存储为 mp4 视频（体积更小，处理更慢）",
)
parser.add_argument("--steps_per_episode", type=int, default=180)
parser.add_argument(
    "--save_failed",
    action="store_true",
    default=False,
    help="同时保存失败的 episode（默认只保存Success的）",
)
parser.add_argument("--robot_type", type=str, default="franka")
parser.add_argument(
    "--target_color",
    type=str,
    default="green",
    choices=["blue", "red", "green"],
    help="目标方块颜色 (blue=cube_1, red=cube_2, green=cube_3)",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="覆盖已存在的数据集目录",
)
parser.add_argument(
    "--save_trajectory",
    action="store_true",
    default=False,
    help="保存EEF轨迹为 numpy 文件 (.npy)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from collections.abc import Sequence

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Environment registration
# =============================================================================


def register_local_tasks(task_root: str) -> None:
    """注册本地任务环境"""
    if not os.path.isdir(task_root):
        return
    if task_root not in sys.path:
        sys.path.insert(0, task_root)

    from gymnasium.envs.registration import registry as gym_registry

    local_task_ids = {
        "Isaac-Lift-Cube-Franka-IK-Rel-v0",
        "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "Isaac-Place-Bin-Franka-IK-Rel-v0",
        "Isaac-Place-Bin-Franka-IK-Rel-Mimic-v0",
        "Isaac-Place-One-Cube-Franka-IK-Rel-v0",
        "Isaac-Place-One-Cube-Franka-IK-Rel-Mimic-v0",
    }

    for task_id in local_task_ids:
        if task_id in gym_registry:
            del gym_registry[task_id]

    for module_name in ("franka",):
        try:
            __import__(module_name)
        except Exception as exc:
            logger.warning(f"Failed to import local task module '{module_name}': {exc}")


# =============================================================================
# Image helpers
# =============================================================================


def prepare_image_for_lerobot(img_tensor) -> np.ndarray:
    """将 Isaac Sim 图像转换为 LeRobot 格式 (H, W, C) uint8.

    Isaac Sim 通常Output (C, H, W) float32 [0,1] 或 [0,255].
    LeRobot 期望 (H, W, C) uint8.
    """
    img = img_tensor.cpu().numpy()

    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img


# =============================================================================
# LeRobot features
# =============================================================================


def create_features(action_dim: int, image_keys: list[str], image_shapes: dict) -> dict:
    """创建 LeRobot features 字典.

    observation.state 使用 18 维格式以兼容推理脚本:
        eef_pos(3) + eef_quat(4) + gripper_pos(2) + joint_pos(9) = 18
    """
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (18,),
            "names": [
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
                "finger_0",
                "finger_1",
            ]
            + [f"joint_{i}" for i in range(9)],
        },
    }

    for img_key in image_keys:
        c, h, w = image_shapes[img_key]
        features[f"observation.images.{img_key}"] = {
            "dtype": "video" if args_cli.use_videos else "image",
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
        }

    return features


# =============================================================================
# Pick-place controller
# =============================================================================


class PickPlaceController:
    """闭环 pick-place 控制器（基于实时 eef 位置的比例控制）"""

    POS_GAIN = 0.4
    MAX_STEP = 0.03
    XY_THRESH = 0.05
    Z_THRESH = 0.05
    WAIT_STEPS = 15

    def __init__(
        self,
        eef_pos: np.ndarray,
        cube_pos: np.ndarray,
        bin_pos: np.ndarray,
        z_grasp_offset: float = 0.0,
        z_place_offset: float = 0.0,
        xy_place_range: float = 0.03,
    ):
        self.step_count = 0
        self.phase_idx = 0
        self.wait_counter = 0

        z_offset = 0.107

        approach = cube_pos.copy()
        min_approach_z = 0.25 + z_offset
        approach[2] = max(eef_pos[2], min_approach_z)

        grasp = cube_pos.copy()
        grasp[2] = 0.0 + z_offset + z_grasp_offset

        lift = cube_pos.copy()
        lift[2] = 0.25 + z_offset

        approach_place = bin_pos.copy()
        approach_place[2] = 0.25 + z_offset

        place = bin_pos.copy()
        place[0] += np.random.uniform(-xy_place_range, xy_place_range)
        place[1] += np.random.uniform(-xy_place_range, xy_place_range)
        place[2] = 0.05 + z_offset + z_place_offset

        self.phases = [
            ("approach", approach, 1.0, "move"),
            ("grasp", grasp, 1.0, "move"),
            ("close_gripper", grasp, -1.0, "wait"),
            ("lift", lift, -1.0, "move"),
            ("approach_place", approach_place, -1.0, "move"),
            ("place", place, -1.0, "move"),
            ("open_gripper", place, 1.0, "wait"),
        ]
        self.phase_names = [p[0] for p in self.phases]

    def get_action(self, eef_pos: np.ndarray) -> np.ndarray:
        if self.done:
            return np.zeros(7, dtype=np.float32)

        _, target_pos, gripper_cmd, phase_type = self.phases[self.phase_idx]
        self.step_count += 1

        action = np.zeros(7, dtype=np.float32)

        if phase_type == "wait":
            self.wait_counter += 1
            if self.wait_counter >= self.WAIT_STEPS:
                self.phase_idx += 1
                self.wait_counter = 0
            action[6] = gripper_cmd
            return action

        error = target_pos - eef_pos
        action[:3] = np.clip(error * self.POS_GAIN, -self.MAX_STEP, self.MAX_STEP)
        action[3:6] = 0.0
        action[6] = gripper_cmd

        xy_dist = np.linalg.norm(error[:2])
        z_dist = abs(error[2])

        # Grasp phases need stricter Z tolerance; place phases can be looser
        z_thresh = 0.015 if self.phase_idx in (1, 2) else self.Z_THRESH

        if xy_dist < self.XY_THRESH and z_dist < z_thresh:
            self.phase_idx += 1

        return action

    @property
    def done(self) -> bool:
        return self.phase_idx >= len(self.phases)

    @property
    def current_phase_name(self) -> str:
        if self.done:
            return "done"
        return self.phase_names[self.phase_idx]


# =============================================================================
# Data quality
# =============================================================================


def analyze_data_quality(actions: np.ndarray) -> dict:
    pos_actions = actions[:, :3]
    magnitude = np.linalg.norm(pos_actions, axis=1)

    return {
        "num_frames": len(actions),
        "mean_magnitude": float(magnitude.mean()),
        "max_magnitude": float(magnitude.max()),
        "std_magnitude": float(magnitude.std()),
        "near_zero_ratio": float(np.mean(magnitude < 0.001)),
        "effective_ratio": float(np.mean(magnitude >= 0.01)),
    }


def check_cube_in_bin(
    cube_pos: np.ndarray, bin_pos: np.ndarray, xy_threshold: float = 0.06
) -> bool:
    xy_dist = np.linalg.norm(cube_pos[:2] - bin_pos[:2])
    z_diff = cube_pos[2] - bin_pos[2]
    in_bin_xy = xy_dist < xy_threshold
    in_bin_z = z_diff > 0.005
    return in_bin_xy and in_bin_z


# =============================================================================
# Main
# =============================================================================


def main():
    args = args_cli

    # 颜色到 cube 名称的映射
    COLOR_TO_CUBE = {
        "blue": "cube_1",
        "red": "cube_2",
        "green": "cube_3",
    }
    target_cube_name = COLOR_TO_CUBE[args.target_color]

    # 根据目标颜色自动更新任务描述
    task_description = args.task_description
    if args.target_color in task_description:
        pass  # 用户已经在描述中提到了颜色
    else:
        task_description = f"pick up the {args.target_color} cube and place it into the blue bin"

    local_task_root = os.path.join(os.path.dirname(__file__), "..", "..", "tasks")
    register_local_tasks(local_task_root)

    print(f"Creating environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.overwrite:
            import shutil

            shutil.rmtree(output_dir)
            print(f"  Deleted old dataset: {output_dir}")
        else:
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.parent / f"{output_dir.name}_{ts}"
            print(f"  Output dir exists, auto using new dir: {output_dir}")

    print("\nProbing environment observation format...")
    env.reset()
    obs, _ = env.reset()
    obs_dict = obs["policy"]

    image_keys = []
    for key in ("table_cam", "table_cam_side", "wrist_cam"):
        if key in obs_dict:
            image_keys.append(key)

    action_dim = 7

    image_shapes = {}
    for img_key in image_keys:
        img = prepare_image_for_lerobot(obs_dict[img_key][0])
        h, w, c = img.shape
        image_shapes[img_key] = (c, h, w)
        print(f"  {img_key}: {h}x{w}x{c}")

    features = create_features(action_dim, image_keys, image_shapes)
    print(f"\nFeatures:")
    for k, v in features.items():
        print(f"  {k}: shape={v['shape']}, dtype={v['dtype']}")

    print(f"\nCreating LeRobot dataset...")
    print(f"  Dir: {output_dir}")
    print(f"  Repo ID: {args.repo_id}")
    print(f"  FPS: {args.fps}")
    print(f"  Video mode: {args.use_videos}")
    print(f"  Target color: {args.target_color} ({target_cube_name})")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=str(output_dir),
        robot_type=args.robot_type,
        use_videos=args.use_videos and len(image_keys) > 0,
        image_writer_threads=4 if len(image_keys) > 0 else 0,
    )

    print(f"\nStarting automated collection: {args.num_episodes} episodes")
    print("=" * 60)

    success_count = 0
    failed_count = 0
    total_saved = 0
    attempt = 0
    max_attempts = args.num_episodes * 5
    all_stats = []

    initial_positions = {
        "eef": [],
        "bin": [],
        target_cube_name: [],
    }

    while success_count < args.num_episodes and attempt < max_attempts:
        attempt += 1
        status_msg = f"Success: {success_count}"
        if args.save_failed:
            status_msg += f", Failed (saved): {failed_count}"
        print(f"\nAttempt {attempt}/{max_attempts} ({status_msg}/{args.num_episodes})")

        env.reset()

        cube_1_pos = env.scene["cube_1"].data.root_pos_w[0].cpu().numpy()
        cube_2_pos = env.scene["cube_2"].data.root_pos_w[0].cpu().numpy()
        cube_3_pos = env.scene["cube_3"].data.root_pos_w[0].cpu().numpy()
        bin_pos = env.scene["blue_sorting_bin"].data.root_pos_w[0].cpu().numpy()

        robot = env.scene["robot"]
        ee_idx = robot.body_names.index("panda_hand")
        eef_pos = robot.data.body_state_w[0, ee_idx, :3].cpu().numpy()

        target_cube_pos = {
            "cube_1": cube_1_pos,
            "cube_2": cube_2_pos,
            "cube_3": cube_3_pos,
        }[target_cube_name]

        print(f"  [Step 0] {target_cube_name}: [{target_cube_pos[0]:.3f}, {target_cube_pos[1]:.3f}, {target_cube_pos[2]:.3f}]")
        print(f"  [Step 0] Bin:  [{bin_pos[0]:.3f}, {bin_pos[1]:.3f}, {bin_pos[2]:.3f}]")
        print(f"  [Step 0] EEF:  [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

        initial_positions["eef"].append(eef_pos.copy())
        initial_positions["bin"].append(bin_pos.copy())
        initial_positions[target_cube_name].append(target_cube_pos.copy())

        print(f"  Target: {target_cube_name} ({args.target_color})")

        z_grasp_offset = np.random.uniform(-0.01, 0.01)
        z_place_offset = np.random.uniform(-0.01, 0.01)
        xy_place_range = np.random.uniform(0.02, 0.05)
        controller = PickPlaceController(
            eef_pos=eef_pos,
            cube_pos=target_cube_pos,
            bin_pos=bin_pos,
            z_grasp_offset=z_grasp_offset,
            z_place_offset=z_place_offset,
            xy_place_range=xy_place_range,
        )
        place_pos = controller.phases[5][1]
        print(f"  Place target: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")
        print(f"  Place offset: XY=±{xy_place_range * 100:.1f}cm, Z={z_place_offset * 100:.1f}cm")

        episode_frames = []
        eef_trajectory = []
        max_steps = args.steps_per_episode * 2
        step = 0

        while not controller.done and step < max_steps:
            action = controller.get_action(eef_pos)
            step += 1

            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(env.device)
            obs, _, _, _, _ = env.step(action_tensor)

            eef_pos = obs["policy"]["eef_pos"][0].cpu().numpy()
            eef_trajectory.append(eef_pos.copy())

            obs_dict = obs["policy"]
            frame = {
                "action": action.astype(np.float32),
                "task": task_description,
            }

            eef_pos_arr = obs_dict["eef_pos"][0].cpu().numpy().flatten()[:3]
            eef_quat_arr = obs_dict["eef_quat"][0].cpu().numpy().flatten()[:4]
            gripper_pos_arr = (
                obs_dict["gripper_pos"][0].cpu().numpy().flatten()[:2]
                if "gripper_pos" in obs_dict
                else np.zeros(2, dtype=np.float32)
            )
            joint_pos_arr = (
                obs_dict["joint_pos"][0].cpu().numpy().flatten()[:9]
                if "joint_pos" in obs_dict
                else np.zeros(9, dtype=np.float32)
            )

            state = np.concatenate(
                [eef_pos_arr, eef_quat_arr, gripper_pos_arr, joint_pos_arr]
            ).astype(np.float32)
            frame["observation.state"] = state

            for img_key in image_keys:
                img = prepare_image_for_lerobot(obs_dict[img_key][0])
                frame[f"observation.images.{img_key}"] = img

            episode_frames.append(frame)
            env.sim.render()

        actions_arr = np.array([f["action"] for f in episode_frames], dtype=np.float32)
        stats = analyze_data_quality(actions_arr)

        print(f"  Total steps: {step}, Final phase: {controller.current_phase_name}")
        if len(actions_arr) > 0:
            print(
                f"    dx: [{actions_arr[:, 0].min() * 100:.2f}, {actions_arr[:, 0].max() * 100:.2f}] cm"
            )
            print(
                f"    dy: [{actions_arr[:, 1].min() * 100:.2f}, {actions_arr[:, 1].max() * 100:.2f}] cm"
            )
            print(
                f"    dz: [{actions_arr[:, 2].min() * 100:.2f}, {actions_arr[:, 2].max() * 100:.2f}] cm"
            )
            print(f"    Avg magnitude: {np.linalg.norm(actions_arr[:, :3], axis=1).mean() * 100:.2f} cm")

        final_target_cube_pos = env.scene[target_cube_name].data.root_pos_w[0].cpu().numpy()
        success = check_cube_in_bin(final_target_cube_pos, bin_pos)

        print(
            f"  Final {target_cube_name}: [{final_target_cube_pos[0]:.3f}, {final_target_cube_pos[1]:.3f}, {final_target_cube_pos[2]:.3f}]"
        )
        xy_dist = np.linalg.norm(final_target_cube_pos[:2] - bin_pos[:2])
        print(f"  XY Distance: {xy_dist * 100:.1f} cm")

        if success or args.save_failed:
            for frame in episode_frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            total_saved += 1

            if args.save_trajectory and eef_trajectory:
                traj_path = output_dir / f"trajectory_{total_saved - 1:03d}.npy"
                np.save(traj_path, np.array(eef_trajectory))
                print(f"  Trajectory saved: {traj_path}")

            if success:
                success_count += 1
                print(f"  ✓ Success! {target_cube_name} placed in bin")
            else:
                failed_count += 1
                print("  ✗ Failed! Data saved")

            print(f"  Frames: {stats['num_frames']}")
            print(f"  Avg magnitude: {stats['mean_magnitude'] * 100:.2f} cm")
            print(f"  Effective movement: {stats['effective_ratio'] * 100:.1f}%")
            print(f"  Episode: {total_saved - 1}")
            all_stats.append(stats)
        else:
            print("  ✗ Failed! Discarded")

    print("\nSaving dataset (computing stats, encoding video)...")
    dataset.finalize()

    if args.use_videos:
        images_dir = output_dir / "images"
        if images_dir.exists():
            import shutil

            shutil.rmtree(images_dir)

    print("\n" + "=" * 60)
    print("Collection complete！")
    print(f"  Success: {success_count}")
    if args.save_failed:
        print(f"  Failed (saved): {failed_count}")
    print(f"  Total attempts: {attempt}")
    print(f"  Total saved: {total_saved}")
    if all_stats:
        print(f"  Avg frames: {np.mean([s['num_frames'] for s in all_stats]):.0f}")
        print(f"  Avg effective movement: {np.mean([s['effective_ratio'] for s in all_stats]) * 100:.1f}%")

    if initial_positions["eef"]:
        print(f"\n  Initial position stats:")
        for key, label in [("eef", "EEF"), ("bin", "Bin"), (target_cube_name, target_cube_name.replace("_", " ").title())]:
            arr = np.array(initial_positions[key])
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            ranges = maxs - mins
            std = np.std(arr, axis=0)
            print(f"    {label} Range: [{ranges[0]:.4f}, {ranges[1]:.4f}, {ranges[2]:.4f}] m")
            print(f"    {label} Std: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}] m")
            print(f"    {label} min: [{mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}]")
            print(f"    {label} max: [{maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f}]")

    print(f"  Output: {output_dir}")
    print("=" * 60)

    env.close()
    simulation_app.close()
    print("\nDone！")


if __name__ == "__main__":
    main()
