#!/usr/bin/env python3
"""
在本地用 LeRobot 训练 ACT 的脚本。

特点：
- 训练入口直接复用 `lerobot.scripts.lerobot_train.train`
- 策略配置直接复用 `lerobot.policies.act.ACTConfig`
- ACT 的主要参数默认值全部来自 `ACTConfig`（可通过命令行覆盖）

示例：
python scripts/model_training/train_act_with_lerobot.py \
  --dataset-root ./datasets/lerobot/auto_collected \
  --output-dir ./outputs/act_place_bin \
  --steps 50000 \
  --batch-size 64 \
  --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 允许直接从本地源码导入 lerobot（如需使用自定义 fork，可设置 LEROBOT_PATH 环境变量）
_lerobot_env = os.environ.get("LEROBOT_PATH", "")
if _lerobot_env:
    LEROBOT_SRC = Path(_lerobot_env)
    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train


def build_parser() -> argparse.ArgumentParser:
    act_defaults = ACTConfig()

    parser = argparse.ArgumentParser(
        description="使用 LeRobot 训练 ACT（参数默认值来自 ACTConfig）"
    )

    # ---------------- Dataset / Training ----------------
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="数据集 repo_id（本地数据集可不填；若 --dataset-root 直接指向数据集目录会自动推断）",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="本地路径。可传数据集上级目录，或直接传包含 meta/info.json 的数据集目录",
    )
    parser.add_argument("--streaming", action="store_true", help="是否使用 streaming 数据加载")

    parser.add_argument("--output-dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--job-name", type=str, default="act_train", help="任务名")
    parser.add_argument(
        "--policy-repo-id",
        type=str,
        default="intern/act_local",
        help="用于 policy.repo_id（训练配置校验必填），例如 intern/act_417",
    )

    parser.add_argument(
        "--device", type=str, default=act_defaults.device or "cuda", help="设备，如 cuda/cpu"
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--eval-freq", type=int, default=0, help="离线训练可设 0 关闭 env eval")

    # ---------------- ACT main params (from ACTConfig) ----------------
    parser.add_argument("--chunk-size", type=int, default=100, help="ACT chunk size (default: 10)")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=100,
        help="Number of action steps to execute per chunk (default: 10)",
    )

    parser.add_argument("--vision-backbone", type=str, default=act_defaults.vision_backbone)
    parser.add_argument(
        "--pretrained-backbone-weights",
        type=str,
        default=act_defaults.pretrained_backbone_weights,
        help="例如 ResNet18_Weights.IMAGENET1K_V1；传 none 可关闭",
    )
    parser.add_argument(
        "--replace-final-stride-with-dilation",
        action="store_true",
        default=bool(act_defaults.replace_final_stride_with_dilation),
    )

    parser.add_argument("--pre-norm", action="store_true", default=act_defaults.pre_norm)
    parser.add_argument("--dim-model", type=int, default=act_defaults.dim_model)
    parser.add_argument("--n-heads", type=int, default=act_defaults.n_heads)
    parser.add_argument("--dim-feedforward", type=int, default=act_defaults.dim_feedforward)
    parser.add_argument(
        "--feedforward-activation", type=str, default=act_defaults.feedforward_activation
    )
    parser.add_argument("--n-encoder-layers", type=int, default=act_defaults.n_encoder_layers)
    parser.add_argument("--n-decoder-layers", type=int, default=act_defaults.n_decoder_layers)

    parser.add_argument("--use-vae", action="store_true", default=act_defaults.use_vae)
    parser.add_argument("--latent-dim", type=int, default=act_defaults.latent_dim)
    parser.add_argument(
        "--n-vae-encoder-layers", type=int, default=act_defaults.n_vae_encoder_layers
    )
    parser.add_argument(
        "--temporal-ensemble-coeff", type=float, default=act_defaults.temporal_ensemble_coeff
    )

    parser.add_argument("--dropout", type=float, default=act_defaults.dropout)
    parser.add_argument(
        "--kl-weight", type=float, default=10.0, help="VAE KL loss weight (default: 1.0)"
    )

    parser.add_argument("--optimizer-lr", type=float, default=act_defaults.optimizer_lr)
    parser.add_argument(
        "--optimizer-weight-decay", type=float, default=act_defaults.optimizer_weight_decay
    )
    parser.add_argument(
        "--optimizer-lr-backbone", type=float, default=act_defaults.optimizer_lr_backbone
    )

    # ---------------- WandB ----------------
    parser.add_argument("--wandb-enable", action="store_true", help="启用 wandb")
    parser.add_argument("--wandb-project", type=str, default="lerobot")

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="使用的 GPU ID，单卡如 '0'，多卡如 '0,1'。多卡时会自动启用分布式训练（torchrun）。",
    )

    return parser


def normalize_none_string(v: str | None) -> str | None:
    if v is None:
        return None
    return None if v.lower() == "none" else v


def resolve_local_dataset_args(
    dataset_repo_id: str | None, dataset_root: str | None
) -> tuple[str, str | None]:
    """解析本地数据集参数。

    规则：
     1) 若 `dataset_root/meta/info.json` 存在，说明 `dataset_root` 直接是数据集目录：
         - repo_id 若未提供，则自动使用目录名
         - root 直接使用该目录（LeRobot 在本地模式下会直接读取 root/meta/info.json）
    2) 否则按原样使用 root，但要求必须显式提供 repo_id。
    """
    if dataset_root is None:
        if dataset_repo_id is None:
            raise ValueError(
                "必须提供 --dataset-repo-id，或提供 --dataset-root 指向本地 LeRobot 数据集目录。"
            )
        return dataset_repo_id, None

    root_path = Path(dataset_root).expanduser().resolve()
    is_dataset_dir = (root_path / "meta" / "info.json").is_file()

    if is_dataset_dir:
        resolved_repo_id = dataset_repo_id or root_path.name
        resolved_root = str(root_path)
        return resolved_repo_id, resolved_root

    if dataset_repo_id is None:
        raise ValueError(
            "当 --dataset-root 不是数据集目录（缺少 meta/info.json）时，必须提供 --dataset-repo-id。"
        )
    return dataset_repo_id, str(root_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---------- 多卡自动启动分布式训练 ----------
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    if len(gpu_ids) > 1 and "LOCAL_RANK" not in os.environ:
        # 当前不在分布式子进程中，用 torchrun 重新启动
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(len(gpu_ids)),
            "--nnodes",
            "1",
            __file__,
        ] + sys.argv[1:]
        print(
            f"[INFO] 检测到多卡请求，自动启动分布式训练: GPUs={args.gpu_ids}, nproc={len(gpu_ids)}"
        )
        subprocess.run(cmd, env=env)
        return

    # 单卡时限制可见 GPU；分布式子进程中 torchrun 已处理好设备
    if len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    # -------------------------------------------

    resolved_repo_id, resolved_root = resolve_local_dataset_args(
        args.dataset_repo_id, args.dataset_root
    )

    policy_cfg = ACTConfig(
        # 核心类型与设备
        repo_id=args.policy_repo_id,
        push_to_hub=False,
        device=args.device,
        # ACT 主参数
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        vision_backbone=args.vision_backbone,
        pretrained_backbone_weights=normalize_none_string(args.pretrained_backbone_weights),
        replace_final_stride_with_dilation=args.replace_final_stride_with_dilation,
        pre_norm=args.pre_norm,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        feedforward_activation=args.feedforward_activation,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_vae=args.use_vae,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        dropout=args.dropout,
        kl_weight=args.kl_weight,
        optimizer_lr=args.optimizer_lr,
        optimizer_weight_decay=args.optimizer_weight_decay,
        optimizer_lr_backbone=args.optimizer_lr_backbone,
        # input_features / output_features 不在此手填，训练时由 dataset metadata 自动推断
        input_features={},
        output_features={},
    )

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=resolved_repo_id,
            root=resolved_root,
            streaming=args.streaming,
        ),
        policy=policy_cfg,
        output_dir=Path(args.output_dir),
        job_name=args.job_name,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        save_checkpoint=True,
        wandb=WandBConfig(enable=args.wandb_enable, project=args.wandb_project),
    )

    # 直接调用 LeRobot 官方训练入口
    train(train_cfg)


if __name__ == "__main__":
    main()
