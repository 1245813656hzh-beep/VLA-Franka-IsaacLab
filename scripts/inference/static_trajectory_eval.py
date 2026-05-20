#!/usr/bin/env python
"""
ACT 模型静态轨迹评估脚本（Static Trajectory Analysis）
支持 LeRobot 数据集格式

功能：
- 离线评估 ACT 模型性能（无需运行物理环境）
- Teacher Forcing：将真实观测值输入模型，预测动作并与真实动作对比
- 生成 16 维动作轨迹对比图（GT vs Pred）
- 计算 Unnormalized MSE 指标

LeRobot 数据集结构：
    dataset/
    ├── meta/
    │   ├── info.json       # 数据集元信息（features, fps 等）
    │   ├── stats.json      # 归一化统计（mean, std）
    │   └── episodes/       # episode 元数据
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet  # 动作和状态数据
    └── videos/             # 视频数据（可选）

使用方法：
    CUDA_VISIBLE_DEVICES=1   \
    python scripts/inference/static_trajectory_eval.py \
        --dataset-path ./datasets/lerobot/auto_collected \
        --model-path ./pretrained_models/act_place_bin \
        --output-dir ./eval_results \
        --use-images 
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# LeRobot 视频解码
try:
    from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec

    VIDEO_DECODER_AVAILABLE = True
except ImportError:
    VIDEO_DECODER_AVAILABLE = False
    print("警告：无法导入 lerobot 视频解码工具，将仅使用 state 进行推理")


# =============================================================================
# LeRobot 数据集加载模块
# =============================================================================


class LeRobotDatasetLoader:
    """LeRobot 格式数据集加载器"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        self.info = None
        self.stats = None
        self.episodes_df = None
        self.features = None
        self.fps = 30
        self.camera_keys = []

    def load(self):
        """加载数据集元信息和数据"""
        print(f"\n加载 LeRobot 数据集：{self.dataset_path}")

        # 1. 加载 info.json
        info_file = self.meta_path / "info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                self.info = json.load(f)
            print(f"  机器人：{self.info.get('robot_type', 'unknown')}")
            print(f"  Episodes: {self.info.get('total_episodes', 0)}")
            print(f"  Total frames: {self.info.get('total_frames', 0)}")
            self.fps = self.info.get("fps", 30)
            print(f"  FPS: {self.fps}")
            self.features = self.info.get("features", {})

            # 提取相机 keys
            self.camera_keys = [
                k for k in self.features.keys() if k.startswith("observation.images.")
            ]
            if self.camera_keys:
                print(f"  相机：{', '.join(self.camera_keys)}")
        else:
            raise FileNotFoundError(f"未找到 info.json: {info_file}")

        # 2. 加载 stats.json
        stats_file = self.meta_path / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.stats = json.load(f)
            print(f"  加载归一化统计：{stats_file}")

# 提取 action 的归一化统计
            if "action" in self.stats:
                self.action_stats = {
                    "mean": np.array(self.stats["action"]["mean"]),
                    "std": np.array(self.stats["action"]["std"]),
                }
                print(
                    f"  Action 归一化：mean=[{self.action_stats['mean'].min():.3f}~{self.action_stats['mean'].max():.3f}], "
                    f"std=[{self.action_stats['std'].min():.3f}~{self.action_stats['std'].max():.3f}]"
                )
            else:
                self.action_stats = None
                print("  譕告：stats 中没有 action 字段")

            # 提取 observation.state 的归一化统计（用于输入归一化）
            if "observation.state" in self.stats:
                self.state_stats = {
                    "mean": np.array(self.stats["observation.state"]["mean"]),
                    "std": np.array(self.stats["observation.state"]["std"]),
                }
                print(
                    f"  State 归一化：mean=[{self.state_stats['mean'].min():.3f}~{self.state_stats['mean'].max():.3f}], "
                    f"std=[{self.state_stats['std'].min():.3f}~{self.state_stats['std'].max():.3f}]"
                )
            else:
                self.state_stats = None
                print("  譕告：stats 中没有 observation.state 字段")
        else:
            print(f"  警告：未找到 stats.json")
            self.stats = None
            self.action_stats = None
            self.state_stats = None

        # 3. 加载 episodes
        episodes_dir = self.meta_path / "episodes"
        if episodes_dir.exists():
            parquet_files = list(episodes_dir.rglob("*.parquet"))
            if parquet_files:
                print(f"  找到 episodes 文件：{len(parquet_files)} 个")
                self.episodes_df = pd.concat(
                    [pd.read_parquet(f) for f in parquet_files], ignore_index=True
                )
                print(f"  加载 episodes: {len(self.episodes_df)} episodes")
            else:
                print(f"  警告：episodes 目录为空")

        # 4. 检查 features
        if self.features:
            print(f"\n  Features:")
            for key, val in self.features.items():
                if isinstance(val, dict) and "shape" in val:
                    print(f"    {key}: shape={val['shape']}")

        # 创建 metadata 对象
        class SimpleMetadata:
            def __init__(self, info, stats, features, fps, camera_keys):
                self.info = info
                self.stats = stats
                self.features = features
                self.fps = fps
                self.camera_keys = camera_keys

        return SimpleMetadata(
            self.info, self.stats, self.features, self.fps, self.camera_keys
        )

    def get_episode_data(self, episode_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """加载指定 episode 的数据"""
        if self.episodes_df is None:
            raise ValueError("请先调用 load() 加载数据集")

        ep_row = self.episodes_df[self.episodes_df["episode_index"] == episode_idx]
        if len(ep_row) == 0:
            raise ValueError(f"未找到 episode {episode_idx}")

        chunk_idx = int(ep_row["data/chunk_index"].values[0])
        file_idx = int(ep_row["data/file_index"].values[0])
        from_frame = int(ep_row["dataset_from_index"].values[0])
        n_frames = int(ep_row["length"].values[0])

        parquet_file = (
            self.data_path / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        )
        if not parquet_file.exists():
            raise FileNotFoundError(f"未找到数据文件：{parquet_file}")

        df = pd.read_parquet(parquet_file)
        ep_df = df.iloc[from_frame : from_frame + n_frames].reset_index(drop=True)

        # LeRobot 格式：action 和 observation.state 是对象列，每个元素是 numpy 数组
        action_col = ep_df["action"]
        state_col = ep_df["observation.state"]

        # 将对象数组转换为二维 numpy 数组
        actions = np.stack(action_col.values).astype(np.float32)
        qpos = np.stack(state_col.values).astype(np.float32)

        print(f"\n  Episode {episode_idx}:")
        print(f"    qpos shape: {qpos.shape}")
        print(f"    actions shape: {actions.shape}")

        return qpos, actions

    def get_episode_video_info(self, episode_idx: int) -> dict:
        """获取 episode 的视频文件信息"""
        if self.episodes_df is None:
            raise ValueError("请先调用 load() 加载数据集")

        ep_row = self.episodes_df[self.episodes_df["episode_index"] == episode_idx]
        if len(ep_row) == 0:
            raise ValueError(f"未找到 episode {episode_idx}")

        ep_row = ep_row.iloc[0]
        video_info = {}

        for cam_key in self.camera_keys:
            chunk_idx = int(ep_row[f"videos/{cam_key}/chunk_index"])
            file_idx = int(ep_row[f"videos/{cam_key}/file_index"])
            from_timestamp = float(ep_row[f"videos/{cam_key}/from_timestamp"])

            video_path = (
                self.dataset_path
                / "videos"
                / cam_key
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.mp4"
            )

            if not video_path.exists():
                print(f"  警告：视频文件不存在 {video_path}")
                continue

            video_info[cam_key] = {
                "path": video_path,
                "from_timestamp": from_timestamp,
            }

        return video_info

    def load_video_frames(
        self,
        video_info: dict,
        frame_indices: list[int],
        backend: str = None,
    ) -> dict[str, torch.Tensor]:
        """
        加载指定帧的视频帧

        Args:
            video_info: get_episode_video_info 返回的视频信息
            frame_indices: 要加载的帧索引列表
            backend: 视频解码后端 ("torchcodec", "pyav"), 默认自动选择

        Returns:
            dict[camera_key -> torch.Tensor]: 形状 (N, C, H, W) 的帧，值范围 [0,1]
        """
        if not VIDEO_DECODER_AVAILABLE:
            return {}

        # 计算 timestamps (从 episode 起始位置开始)
        timestamps = [idx / self.fps for idx in frame_indices]

        frames_dict = {}
        for cam_key, info in video_info.items():
            # 计算相对于视频文件起始的 timestamps
            shifted_ts = [info["from_timestamp"] + ts for ts in timestamps]

            # 如果指定了特定后端，则先尝试该后端
            if backend is not None:
                try:
                    # 解码视频帧
                    frames = decode_video_frames(
                        info["path"],
                        shifted_ts,
                        tolerance_s=1.0 / self.fps * 0.5,  # 容忍半帧的误差
                        backend=backend,
                    )
                    frames_dict[cam_key] = frames
                    continue  # 成功则继续下一个摄像头
                except Exception as e:
                    print(
                        f"  警告: 使用指定后端 '{backend}' 无法解码 {info['path']}: {e}"
                    )
                    # 不返回空字典，尝试其他后端

            # 如果没有指定 backend 或指定的 backend 失败，则尝试自动选择
            backends_to_try = ["torchcodec", "pyav", "opencv", "imageio"]  # 尝试顺序
            success = False

            for attempt_backend in backends_to_try:
                try:
                    frames = decode_video_frames(
                        info["path"],
                        shifted_ts,
                        tolerance_s=1.0 / self.fps * 0.5,  # 容忍半帧的误差
                        backend=attempt_backend,
                    )
                    frames_dict[cam_key] = frames
                    success = True
                    # print(f"  成功使用 {attempt_backend} 解码 {info['path']}")
                    break  # 成功则退出尝试循环
                except Exception as e:
                    continue  # 尝试下一个后端

            # 如果所有后端都无法解码，则创建合适的占位符
            if not success:
                print(f"  警告: 无法使用任何后端解码 {info['path']}，创建占位符张量")
                # 从特征集中获取预期的图像尺寸
                if hasattr(self, "features") and cam_key in self.features:
                    shape_info = self.features[cam_key]["shape"]
                    height, width, channels = shape_info
                    # 创建占位符：(N, C, H, W) 格式
                    placeholder_img = torch.zeros(
                        len(frame_indices), channels, height, width, dtype=torch.float32
                    )
                    frames_dict[cam_key] = placeholder_img
                    print(
                        f"    为 {cam_key} 创建了形状为 [{channels}, {height}, {width}] 的占位符张量"
                    )
                else:
                    # 默认尺寸
                    placeholder_img = torch.zeros(
                        len(frame_indices), 3, 480, 640, dtype=torch.float32
                    )
                    frames_dict[cam_key] = placeholder_img
                    print(f"    为 {cam_key} 创建了默认形状 [3, 480, 640] 的占位符张量")

        return frames_dict

    def get_action_dim(self) -> int:
        if "features" in self.info and "action" in self.info["features"]:
            shape = self.info["features"]["action"]["shape"]
            return shape[0] if len(shape) > 0 else 14
        return 16


def load_lerobot_data(dataset_path: str, episode_idx: int = 0):
    """从 LeRobot 数据集加载单条轨迹数据"""
    loader = LeRobotDatasetLoader(dataset_path)
    metadata = loader.load()
    qpos, actions = loader.get_episode_data(episode_idx)
    return qpos, actions, metadata


# =============================================================================
# ACT 模型加载模块
# =============================================================================


def load_act_model(
    model_path: str,
    metadata: dict,
    dataset_stats: dict = None,
    device: str = "cuda",
) -> tuple[torch.nn.Module, object, object, object]:
    """加载 ACT 模型及其预处理器和后处理器
    
    Returns:
        policy: ACT 模型
        policy_config: 模型配置
        preprocessor: 输入预处理器（归一化）
        postprocessor: 输出后处理器（反归一化）
    """
    print(f"\n加载 ACT 模型：{model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在：{model_path}")

    config_file = Path(model_path) / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"未找到 config.json: {config_file}")

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
    except ImportError as e:
        raise ImportError("请确保已安装 lerobot") from e

    print("加载配置...")

    # 临时修复：移除不支持的 use_peft 字段
    import shutil

    config_backup = config_file.with_suffix(".json.bak")
    shutil.copy(config_file, config_backup)

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # 移除不支持的字段
    removed_fields = []
    if "use_peft" in config_dict:
        removed_fields.append("use_peft")
        del config_dict["use_peft"]

    if removed_fields:
        print(f"  临时移除不支持的字段：{', '.join(removed_fields)}")
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=4)

    try:
        policy_config = PreTrainedConfig.from_pretrained(model_path)
        policy_config.pretrained_path = Path(model_path)
        policy_config.device = device
    finally:
        # 恢复原始配置
        if config_backup.exists():
            shutil.move(config_backup, config_file)

    print(f"\n模型配置：")
    print(f"  类型：{policy_config.type}")
    print(f"  chunk_size: {policy_config.chunk_size}")
    print(f"  device: {policy_config.device}")

    print("\n创建策略模型...")
    policy = make_policy(policy_config, ds_meta=metadata)

    from safetensors.torch import load_file as load_safetensors

    model_safetensors = Path(model_path) / "model.safetensors"

    if model_safetensors.exists():
        state_dict = load_safetensors(str(model_safetensors), device="cpu")
    else:
        model_pytorch = Path(model_path) / "pytorch_model.bin"
        state_dict = torch.load(model_pytorch, map_location="cpu")

    processed_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(processed_state_dict, strict=False)
    policy.to(device)
    policy.eval()

    print(
        f"模型加载完成：{sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M 参数"
    )
    
    # 创建预处理器和后处理器
    preprocessor = None
    postprocessor = None
    
    # 首先尝试从模型目录加载预处理器配置
    preprocessor_json = Path(model_path) / "policy_preprocessor.json"
    postprocessor_json = Path(model_path) / "policy_postprocessor.json"
    
    if preprocessor_json.exists() and postprocessor_json.exists():
        print("\n从模型目录加载预处理器/后处理器...")
        try:
            from lerobot.processor import PolicyProcessorPipeline
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                model_path, config_filename="policy_preprocessor.json"
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                model_path, config_filename="policy_postprocessor.json"
            )
            print("  预处理器/后处理器加载成功")
        except Exception as e:
            print(f"  加载预处理器失败: {e}")
    
    # 如果从模型目录加载失败，尝试使用 dataset_stats 创建
    if preprocessor is None and dataset_stats is not None:
        print("\n使用 dataset_stats 创建预处理器/后处理器...")
        try:
            import torch
            tensor_stats = {}
            for key, stat_dict in dataset_stats.items():
                tensor_stats[key] = {
                    "mean": torch.tensor(stat_dict["mean"], dtype=torch.float32),
                    "std": torch.tensor(stat_dict["std"], dtype=torch.float32),
                }
            
            preprocessor, postprocessor = make_act_pre_post_processors(
                config=policy_config,
                dataset_stats=tensor_stats
            )
            print("  预处理器/后处理器创建成功")
        except Exception as e:
            print(f"  创建预处理器失败: {e}")
    
    if preprocessor is None:
        print("\n警告：未能创建预处理器，将使用原始数据直接推理（可能导致性能下降）")

    return policy, policy_config, preprocessor, postprocessor


# =============================================================================
# Teacher Forcing 推理模块
# =============================================================================


def run_teacher_forcing_inference(
    policy: torch.nn.Module,
    qpos_data: np.ndarray,
    actions_data: np.ndarray,
    config,
    preprocessor=None,
    postprocessor=None,
    loader: LeRobotDatasetLoader = None,
    video_info: dict = None,
    device: str = "cuda",
    chunk_size: int = 100,
    use_images: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Teacher Forcing 推理
    
    Args:
        policy: ACT 模型
        qpos_data: 真实观测 qpos [T, action_dim]（未归一化）
        actions_data: 真实动作 [T, action_dim]（未归一化）
        config: 模型配置
        preprocessor: 输入预处理器（归一化）
        postprocessor: 输出后处理器（反归一化）
        loader: LeRobotDatasetLoader 实例（用于加载视频）
        video_info: 视频文件信息
        device: 运行设备
        chunk_size: 动作 chunk 大小
        use_images: 是否使用图像输入
    """
    print("\n" + "=" * 70)
    print("Teacher Forcing 推理")
    print("=" * 70)

    T, action_dim = qpos_data.shape
    predicted_actions = np.zeros((T, action_dim), dtype=np.float32)

    camera_keys = []
    if use_images and loader is not None and video_info is not None:
        camera_keys = loader.camera_keys
        if not camera_keys:
            print("  警告：数据集没有图像数据，将仅使用 state 进行推理")
            use_images = False
        elif not VIDEO_DECODER_AVAILABLE:
            print("  警告：视频解码器不可用，将仅使用 state 进行推理")
            use_images = False
        else:
            print(f"  使用图像输入：{', '.join(camera_keys)}")

    print(f"\n推理参数：")
    print(f"  时间步数 T: {T}")
    print(f"  动作维度：{action_dim}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  设备：{device}")
    print(f"  使用预处理器: {preprocessor is not None}")
    print(f"  使用后处理器: {postprocessor is not None}")

    if preprocessor is not None:
        test_batch = {"observation.state": torch.from_numpy(qpos_data[0]).float()}
        if use_images:
            for cam_key in camera_keys:
                test_batch[cam_key] = torch.zeros(3, 480, 640, dtype=torch.float32)
        processed_test = preprocessor(test_batch)
        print(f"\n  [调试] 预处理后 observation.state 形状: {processed_test['observation.state'].shape}")
        print(f"  [调试] 预处理后 observation.state 均值: {processed_test['observation.state'].mean().item():.4f}")
        print(f"  [调试] 预处理后 observation.state 标准差: {processed_test['observation.state'].std().item():.4f}")

    policy.eval()
    policy_device = next(policy.parameters()).device
    
    image_features = getattr(config, 'image_features', {}) or {}
    if image_features and not use_images:
        print(f"  注意：模型需要图像输入，但未启用 --use-images，将使用占位符图像")

    with torch.no_grad():
        for t in range(T):
            if (t + 1) % 50 == 0:
                print(f"  推理进度：{t + 1}/{T} ({100 * (t + 1) / T:.1f}%)")

            batch = {"observation.state": torch.from_numpy(qpos_data[t]).float()}

            if use_images and video_info:
                try:
                    frames = loader.load_video_frames(video_info, [t])
                    for cam_key in camera_keys:
                        if cam_key in frames:
                            batch[cam_key] = frames[cam_key][0]
                except Exception as e:
                    print(f"  警告：无法解码视频帧: {str(e)}")
                    for cam_key in camera_keys:
                        if hasattr(loader, "features") and cam_key in loader.features:
                            shape_info = loader.features[cam_key]["shape"]
                            height, width, channels = shape_info
                            batch[cam_key] = torch.zeros(channels, height, width, dtype=torch.float32)
                        else:
                            batch[cam_key] = torch.zeros(3, 480, 640, dtype=torch.float32)
            
            elif image_features:
                for cam_key in image_features.keys():
                    if hasattr(loader, "features") and cam_key in loader.features:
                        shape_info = loader.features[cam_key]["shape"]
                        height, width, channels = shape_info
                        batch[cam_key] = torch.zeros(channels, height, width, dtype=torch.float32)
                    else:
                        batch[cam_key] = torch.zeros(3, 480, 640, dtype=torch.float32)

            if preprocessor is not None:
                processed_batch = preprocessor(batch)
                for key in ["observation.state"] + list(image_features.keys()):
                    if key in processed_batch and torch.is_tensor(processed_batch[key]):
                        if processed_batch[key].device != policy_device:
                            processed_batch[key] = processed_batch[key].to(policy_device)
                        if processed_batch[key].dim() == 1:
                            processed_batch[key] = processed_batch[key].unsqueeze(0)
            else:
                processed_batch = {}
                for key, value in batch.items():
                    if not torch.is_tensor(value):
                        value = torch.from_numpy(np.asarray(value)).float()
                    if value.device != policy_device:
                        value = value.to(policy_device)
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                    elif value.dim() == 3:
                        value = value.unsqueeze(0)
                    processed_batch[key] = value

            action_chunk = policy.predict_action_chunk(processed_batch)
            
            action_tensor = action_chunk[:, 0, :]
            
            if t == 0:
                print(f"  [调试] 模型输出动作均值: {action_tensor.mean().item():.4f}")
                print(f"  [调试] 模型输出动作标准差: {action_tensor.std().item():.4f}")
            
            if postprocessor is not None:
                action_batch = {"action": action_tensor}
                action_output = postprocessor(action_batch)
                if isinstance(action_output, dict):
                    pred_action = action_output["action"].cpu().numpy()[0]
                else:
                    pred_action = action_output.cpu().numpy()[0]
                if t == 0:
                    print(f"  [调试] 后处理后动作均值: {pred_action.mean():.4f}")
                    print(f"  [调试] 后处理后动作标准差: {pred_action.std():.4f}")
            else:
                pred_action = action_tensor.cpu().numpy()[0]
                if hasattr(loader, "action_stats") and loader.action_stats is not None:
                    pred_action = pred_action * loader.action_stats["std"] + loader.action_stats["mean"]

            predicted_actions[t] = pred_action

    print(f"\n推理完成")

    actions_data_unnorm = actions_data.copy()
    if preprocessor is None and hasattr(loader, "action_stats") and loader.action_stats is not None:
        actions_data_unnorm = actions_data * loader.action_stats["std"] + loader.action_stats["mean"]
        print(f"已对真实动作数据进行反归一化")

    mse = np.mean((predicted_actions - actions_data_unnorm) ** 2)
    mae = np.mean(np.abs(predicted_actions - actions_data_unnorm))
    rmse = np.sqrt(mse)

    print(f"\n评估指标（Unnormalized）：")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    per_dim_mse = np.mean((predicted_actions - actions_data_unnorm) ** 2, axis=0)

    return predicted_actions, mse, per_dim_mse


# =============================================================================
# 绘图模块
# =============================================================================


def plot_trajectory(
    ground_truth_actions: np.ndarray,
    predicted_actions: np.ndarray,
    unnormalized_mse: float,
    per_dim_mse: np.ndarray,
    output_path: str,
    action_dim: int = 16,
    inference_point_interval: int = 10,
    episode_id: int = 0,
    modalities: str = "right_arm, right_gripper, left_arm, left_gripper",
):
    """绘制轨迹对比图"""
    print(f"\n绘制轨迹对比图...")

    T = ground_truth_actions.shape[0]
    time_steps = np.arange(T)

    fig = plt.figure(figsize=(14, 40))
    gs = GridSpec(action_dim, 1, figure=fig, hspace=0.3)

    for dim in range(action_dim):
        ax = fig.add_subplot(gs[dim, 0])

        ax.plot(
            time_steps,
            ground_truth_actions[:, dim],
            color="#1f77b4",
            linewidth=1.5,
            label="gt action",
            zorder=2,
        )
        ax.plot(
            time_steps,
            predicted_actions[:, dim],
            color="#ff7f0e",
            linewidth=1.5,
            label="pred action",
            zorder=2,
        )

        inference_points = list(range(0, T, inference_point_interval))
        if T - 1 not in inference_points:
            inference_points.append(T - 1)

        ax.scatter(
            inference_points,
            ground_truth_actions[inference_points, dim],
            color="red",
            s=15,
            label="inference point",
            zorder=3,
        )

        ax.set_title(
            f"Action Dimension {dim} (MSE: {per_dim_mse[dim]:.4f})",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)

        dim_min = min(
            ground_truth_actions[:, dim].min(), predicted_actions[:, dim].min()
        )
        dim_max = max(
            ground_truth_actions[:, dim].max(), predicted_actions[:, dim].max()
        )
        dim_range = dim_max - dim_min
        if dim_range > 0:
            ax.set_ylim(dim_min - 0.05 * dim_range, dim_max + 0.15 * dim_range)

        ax.grid(True, alpha=0.3, linestyle="--")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Trajectory Analysis - ID: {episode_id}\n"
        f"Modalities: {modalities}\n"
        f"Unnormalized MSE: {unnormalized_mse:.6f}",
        fontsize=12,
        fontweight="bold",
        color="#1f77b4",
        y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图片已保存：{output_path}")


# =============================================================================
# 主函数
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="ACT 静态轨迹评估（LeRobot 格式）")

    parser.add_argument(
        "--dataset-path", type=str, required=True, help="LeRobot 数据集路径"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="ACT 模型 checkpoint 路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./eval_results", help="结果输出目录"
    )
    parser.add_argument(
        "--episode-idx", type=int, default=0, help="要评估的 episode 索引"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="运行设备（cuda/cpu）"
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="Action Chunk 大小")
    parser.add_argument(
        "--inference-interval", type=int, default=10, help="Inference point 间隔"
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="right_arm, right_gripper, left_arm, left_gripper",
        help="模态描述",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="是否使用图像输入（默认仅使用 state）",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default=None,
        help="视频解码后端（torchcodec/pyav，默认自动选择）",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告：CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{output_dir}")

    # 步骤 1: 加载数据
    print("\n" + "=" * 70)
    print("步骤 1: 加载 LeRobot 数据集")
    print("=" * 70)

    loader = LeRobotDatasetLoader(args.dataset_path)
    metadata = loader.load()
    qpos, actions = loader.get_episode_data(args.episode_idx)
    action_dim = actions.shape[1]

    # 获取视频信息（如果使用图像）
    video_info = {}
    if args.use_images:
        print("\n加载视频信息...")
        video_info = loader.get_episode_video_info(args.episode_idx)
        print(f"找到 {len(video_info)} 个视频文件")

    # 步骤 2: 加载模型
    print("\n" + "=" * 70)
    print("步骤 2: 加载 ACT 模型")
    print("=" * 70)
    
    dataset_stats = None
    if hasattr(loader, 'stats') and loader.stats is not None:
        dataset_stats = loader.stats
        print(f"使用数据集 stats 进行归一化")
    
    policy, config, preprocessor, postprocessor = load_act_model(
        args.model_path, metadata, dataset_stats, args.device
    )

    if hasattr(config, "chunk_size"):
        args.chunk_size = config.chunk_size

    # 步骤 3: Teacher Forcing 推理
    print("\n" + "=" * 70)
    print("步骤 3: Teacher Forcing 推理")
    print("=" * 70)
    predicted_actions, mse, per_dim_mse = run_teacher_forcing_inference(
        policy=policy,
        qpos_data=qpos,
        actions_data=actions,
        config=config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        loader=loader if args.use_images else None,
        video_info=video_info if args.use_images else None,
        device=args.device,
        chunk_size=args.chunk_size,
        use_images=args.use_images,
    )

    # 步骤 4: 绘图
    print("\n" + "=" * 70)
    print("步骤 4: 绘制轨迹图")
    print("=" * 70)

    output_image = output_dir / f"eval_ep{args.episode_idx}_trajectory.png"
    plot_trajectory(
        actions,
        predicted_actions,
        mse,
        per_dim_mse,
        str(output_image),
        action_dim,
        args.inference_interval,
        args.episode_idx,
        args.modalities,
    )

    # 保存指标
    metrics = {
        "episode_idx": args.episode_idx,
        "dataset": args.dataset_path,
        "model": args.model_path,
        "frames": int(qpos.shape[0]),
        "action_dim": action_dim,
        "mse": float(mse),
        "mae": float(np.mean(np.abs(predicted_actions - actions))),
        "rmse": float(np.sqrt(mse)),
        "per_dim_mse": [float(x) for x in per_dim_mse],
    }
    metrics_file = output_dir / f"eval_ep{args.episode_idx}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("评估完成")
    print("=" * 70)
    print(f"\n结果：")
    print(f"  Unnormalized MSE: {mse:.6f}")
    print(f"  轨迹图：{output_image}")
    print(f"  指标文件：{metrics_file}")


if __name__ == "__main__":
    main()
