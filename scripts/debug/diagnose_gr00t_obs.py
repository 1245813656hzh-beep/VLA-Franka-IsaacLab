#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def save_comparison_image(env_img, dataset_img, cam_name, output_path, flip_hint="unknown"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(env_img)
    axes[0].set_title(f"Env Obs (with flip)\n{cam_name}")
    axes[0].axis("off")

    env_img_no_flip = env_img[:, ::-1, :]
    axes[1].imshow(env_img_no_flip)
    axes[1].set_title(f"Env Obs (NO flip)\n{cam_name}")
    axes[1].axis("off")

    if dataset_img is not None:
        axes[2].imshow(dataset_img)
    else:
        axes[2].text(
            0.5, 0.5, "No dataset\nimage loaded", ha="center", va="center", transform=axes[2].transAxes
        )
    axes[2].set_title(f"Dataset (training)\n{cam_name}\n{flip_hint}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def diagnose_observations(env_obs_dict, dataset_sample, action_with_flip, action_without_flip, output_dir, step=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"DIAGNOSIS STEP {step}")
    print("=" * 70)

    cam_names = ["table_cam", "table_cam_side", "wrist_cam"]
    for cam_name in cam_names:
        if cam_name not in env_obs_dict:
            continue

        env_img = env_obs_dict[cam_name]
        if isinstance(env_img, torch.Tensor):
            env_img = env_img[0].cpu().numpy()
        if env_img.ndim == 4:
            env_img = env_img[0]
        if env_img.ndim == 3 and env_img.shape[0] == 3:
            env_img = np.transpose(env_img, (1, 2, 0))
        if env_img.dtype != np.uint8:
            if env_img.max() <= 1.0:
                env_img = (env_img * 255).astype(np.uint8)
            else:
                env_img = env_img.astype(np.uint8)

        dataset_img = None
        flip_hint = "unknown"
        if dataset_sample is not None:
            video_key = f"video.{cam_name}"
            if video_key in dataset_sample:
                dataset_img_raw = dataset_sample[video_key]
                if isinstance(dataset_img_raw, np.ndarray):
                    if dataset_img_raw.ndim == 5:
                        dataset_img = dataset_img_raw[0, 0]
                    elif dataset_img_raw.ndim == 4:
                        dataset_img = dataset_img_raw[0]
                    elif dataset_img_raw.ndim == 3:
                        dataset_img = dataset_img_raw
                elif isinstance(dataset_img_raw, torch.Tensor):
                    dataset_img = dataset_img_raw[0, 0].cpu().numpy()

                if dataset_img is not None and dataset_img.dtype != np.uint8:
                    if dataset_img.max() <= 1.0:
                        dataset_img = (dataset_img * 255).astype(np.uint8)
                    else:
                        dataset_img = dataset_img.astype(np.uint8)

                if dataset_img is not None:
                    env_gray = np.mean(env_img, axis=2)
                    env_gray_flip = np.mean(env_img[:, ::-1, :], axis=2)
                    dataset_gray = np.mean(dataset_img, axis=2)

                    if env_gray.shape != dataset_gray.shape:
                        try:
                            from PIL import Image
                            dataset_pil = Image.fromarray(dataset_img)
                            dataset_pil = dataset_pil.resize((env_img.shape[1], env_img.shape[0]), Image.BILINEAR)
                            dataset_img = np.array(dataset_pil)
                            dataset_gray = np.mean(dataset_img, axis=2)
                        except ImportError:
                            pass

                    if env_gray.shape == dataset_gray.shape:
                        diff_no_flip = np.mean(np.abs(env_gray.astype(float) - dataset_gray.astype(float)))
                        diff_with_flip = np.mean(np.abs(env_gray_flip.astype(float) - dataset_gray.astype(float)))
                        if diff_with_flip < diff_no_flip:
                            flip_hint = "TRAINING WAS FLIPPED (keep flip)"
                        else:
                            flip_hint = "TRAINING WAS NOT FLIPPED (use --no-flip)"
                        print(f"  {cam_name}: diff_no_flip={diff_no_flip:.1f}, diff_with_flip={diff_with_flip:.1f}")

        output_path = output_dir / f"step{step:03d}_{cam_name}_comparison.png"
        save_comparison_image(env_img, dataset_img, cam_name, output_path, flip_hint)

    print("\n  STATE COMPARISON:")
    for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
        if key in env_obs_dict:
            val = env_obs_dict[key]
            if isinstance(val, torch.Tensor):
                val = val[0].cpu().numpy()
            print(f"    Env {key}: {val.flatten()[:6]}")

    if dataset_sample is not None:
        for key in ["state.end_effector", "state.fingers", "state.joints"]:
            if key in dataset_sample:
                val = dataset_sample[key]
                if isinstance(val, np.ndarray):
                    if val.ndim >= 2:
                        val = val[0]
                    print(f"    Dataset {key}: {val.flatten()[:6]}")
                elif isinstance(val, torch.Tensor):
                    val = val[0].cpu().numpy()
                    print(f"    Dataset {key}: {val.flatten()[:6]}")

    print("\n  ACTION COMPARISON:")
    print(f"    With flip:    {action_with_flip[:8] if action_with_flip is not None else 'N/A'}")
    print(f"    Without flip: {action_without_flip[:8] if action_without_flip is not None else 'N/A'}")
    if action_with_flip is not None and action_without_flip is not None:
        diff = np.abs(action_with_flip - action_without_flip)
        print(f"    Action diff:  {diff[:8]}")
        print(f"    Max diff:     {np.max(diff):.6f}")

    print(f"\n  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--server-host", type=str, default="localhost")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--output-dir", type=str, default="./debug_output")
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--task", type=str, default="Isaac-Place-Bin-Franka-IK-Rel-v0")
    args = parser.parse_args()

    print("=" * 70)
    print("GR00T Observation Diagnostic Tool")
    print("=" * 70)

    from scripts.inference.gr00t_remote_client import (
        ActionPostProcessor,
        Gr00tRemoteClient,
        IsaacObsConverter,
    )

    client = Gr00tRemoteClient(host=args.server_host, port=args.server_port, timeout_ms=30000)
    modality_config = client.get_modality_config()
    action_horizon = len(modality_config["action"]["delta_indices"])
    print(f"Action horizon: {action_horizon}")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.franka import place_bin_ik_rel_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    env = gym.make(args.task, cfg=env_cfg)

    dataset_sample = None
    if args.dataset_path:
        try:
            print(f"\nLoading dataset from {args.dataset_path}...")
            from gr00t.data.dataset import LeRobotSingleDataset
            dataset = LeRobotSingleDataset(
                dataset_path=args.dataset_path,
                modality_configs=modality_config,
                transforms=None,
                embodiment_tag="new_embodiment",
            )
            dataset_sample = dataset[0]
            print(f"Dataset loaded! Keys: {list(dataset_sample.keys())}")
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")

    obs_converter_with_flip = IsaacObsConverter(flip_images=True)
    obs_converter_no_flip = IsaacObsConverter(flip_images=False)
    action_processor = ActionPostProcessor(action_horizon=action_horizon)

    obs, info = env.reset(seed=42)
    obs_dict = obs["policy"]

    for step in range(args.num_steps):
        gr00t_obs_with = obs_converter_with_flip.convert(obs_dict)
        action_dict_with = client.get_action(gr00t_obs_with)
        action_with = action_dict_with.get("action.full_action", None)
        if action_with is not None and isinstance(action_with, np.ndarray):
            if action_with.ndim == 3:
                action_with = action_with[0, 0]
            elif action_with.ndim == 2:
                action_with = action_with[0]

        gr00t_obs_without = obs_converter_no_flip.convert(obs_dict)
        action_dict_without = client.get_action(gr00t_obs_without)
        action_without = action_dict_without.get("action.full_action", None)
        if action_without is not None and isinstance(action_without, np.ndarray):
            if action_without.ndim == 3:
                action_without = action_without[0, 0]
            elif action_without.ndim == 2:
                action_without = action_without[0]

        diagnose_observations(obs_dict, dataset_sample, action_with, action_without, args.output_dir, step=step)

        if action_with is not None:
            action_tensor = torch.from_numpy(action_with).float().unsqueeze(0).to("cuda:0")
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            obs_dict = obs["policy"]

    env.close()
    simulation_app.close()
    client.close()

    print("\n" + "=" * 70)
    print(f"Done! Check: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import torch
    main()
