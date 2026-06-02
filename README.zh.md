# VLA-Franka-IsaacLab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | **中文**

基于 [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) / [IsaacLab](https://github.com/isaac-sim/IsaacLab) 的 VLA (Vision-Language-Action) 机械臂仿真部署工具链，支持 Franka 机器人在仿真环境中的**自动化数据采集**与**VLA 模型推理**。

<p align="center">
  <b>数据采集 → 模型训练 → 推理部署</b>
</p>

## 演示 — 蓝、红、绿三色方块堆叠

<p align="center">
  <video src="blue_red_green/ep002_table_cam.mp4" controls muted width="32%"></video>
  <video src="blue_red_green/ep002_table_cam_side.mp4" controls muted width="32%"></video>
  <video src="blue_red_green/ep002_wrist_cam.mp4" controls muted width="32%"></video>
</p>

<p align="center">
  <sub><b>左：</b>桌面视角（正面）&nbsp;|&nbsp; <b>中：</b>侧面视角 &nbsp;|&nbsp; <b>右：</b>腕部视角</sub>
</p>

## 功能特性

- 🤖 **自动化数据采集**
  - 单方块 Pick-Place 任务自动化轨迹生成与采集
  - 多方块顺序堆叠采集
  - 直接输出 [LeRobot](https://github.com/huggingface/lerobot) v3.0 格式数据集
  - 内置闭环控制器，自动判断成功率

- 🧠 **VLA 模型推理**
  - 支持 **GR00T-N1.5** (NVIDIA) 本地推理
  - 支持 **ACT** (Action Chunking Transformer) 本地推理
  - 支持 **GR00T 远程推理**（ZMQ 服务解耦，避免 CUDA 冲突）

- 🎯 **自定义 IsaacLab 环境**
  - **Place-Bin**（放方块到篮子）— 核心自定义环境
  - 多摄像头配置（俯视 Table Cam / 侧视 Side Cam / 腕部 Wrist Cam）
  - 支持键盘遥操作

## 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| Ubuntu | 22.04 | 必需 |
| NVIDIA GPU | VRAM >= 12GB | 用于 Isaac Sim + 模型推理 |
| NVIDIA Driver | >= 535 | CUDA 12.x 需要 |
| Isaac Sim | 4.2+ | 仿真引擎，需先独立安装 |
| IsaacLab | >= 0.48.0 | 机器人学习框架 |
| Python | 3.10 / 3.11 | 3.11 用于 IsaacLab，3.10 用于 GR00T |
| CUDA | 12.x | 与 Isaac Sim 版本匹配 |

> ⚠️ **版本兼容性警告**：本项目在以下版本组合中验证通过：
> - IsaacLab 0.48.5 + isaaclab-mimic 1.0.15 + isaaclab-tasks 0.11.8
> - LeRobot 0.4.4
> - transformers 4.40+ (isaac 环境), 4.45~4.51 (gr00t 环境)

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/1245813656hzh-beep/VLA-Franka-IsaacLab.git
cd VLA-Franka-IsaacLab
```

### 2. 安装 NVIDIA Isaac Sim

这是最大的前置依赖，**必须先完成**：

```bash
# 方式一：Omniverse Launcher (推荐)
# 下载 https://www.nvidia.com/omniverse 并安装 Isaac Sim 4.2+

# 方式二：pip 安装
pip install isaacsim==4.2.0 --extra-index-url https://pypi.nvidia.com
```

设置环境变量：
```bash
export ISAACSIM_PATH=/path/to/isaac-sim  # 例如 ~/isaacsim
```

### 3. 安装 IsaacLab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
# 或使用 pip
pip install isaaclab isaaclab-tasks isaaclab-assets isaaclab-mimic
```

### 4. 安装项目依赖

本项目涉及**多个 conda 环境**（因为依赖冲突）。推荐按功能创建独立环境：

#### 环境 A — IsaacLab 仿真环境（数据采集 + ACT 推理）

```bash
conda create -n isaac python=3.11
conda activate isaac
pip install -e .
pip install -r requirements-isaac.txt
python scripts/debug/check_installation.py
```

#### 环境 B — GR00T 推理服务

```bash
conda create -n isaac_gr00t python=3.10
conda activate isaac_gr00t
pip install -e .

# 安装 GR00T-N1.5（必须从源码）
mkdir -p third_party
git clone https://github.com/NVIDIA/GR00T.git third_party/GR00T-N1.5
cd third_party/GR00T-N1.5
pip install -e .

pip install -r ../../requirements-gr00t.txt
python -c "import gr00t; print('GR00T OK')"
```

#### 环境 C — ACT 训练环境

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -e ".[train]"
pip install -r requirements-train.txt
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobot OK')"
```

> 如果你只需要数据采集和 ACT 推理，可以只创建 **环境 A**。

## 快速开始

### 自动化数据采集

#### 单方块 Pick-Place

```bash
python scripts/data_collection/auto_collect_single.py \
    --num_episodes 100 \
    --output_dir datasets/lerobot/franka_place_bin \
    --repo_id local/franka_place_bin \
    --use_videos \
    --target_color green
```

参数说明：
- `--num_episodes`: 目标成功 episode 数量
- `--target_color`: 目标方块颜色 (`blue`, `red`, `green`)
- `--use_videos`: 将图像存储为 mp4（体积小，推荐）
- `--save_failed`: 同时保存失败的 episode

#### 多方块堆叠

```bash
python scripts/data_collection/auto_collect_stack.py \
    --num_episodes 100 \
    --output_dir datasets/lerobot/franka_stack \
    --repo_id local/franka_stack \
    --use_videos
```

抓取顺序：蓝(cube_1) → 红(cube_2) → 绿(cube_3)，依次堆叠到篮子中。

### 人工遥操作采集

```bash
python scripts/data_collection/record_demos.py \
    --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
    --dataset_file datasets/franka_place_bin.hdf5 \
    --teleop_device keyboard \
    --step_hz 30
```

键盘控制映射：
- **I/K**: X 轴前后 | **J/L**: Y 轴左右 | **U/O**: Z 轴上下
- **N/M**: X 旋转 | **T/G**: Y 旋转 | **Y/B**: Z 旋转
- **P**: 夹爪开关 | **E**: 导出 episode | **R**: 重置环境

### 模型训练

#### ACT 训练（使用 LeRobot）

```bash
python scripts/model_training/train_act_with_lerobot.py \
    --dataset-root ./datasets/lerobot/franka_place_bin \
    --output-dir ./outputs/act_place_bin \
    --steps 50000 \
    --batch-size 64 \
    --device cuda
```

### VLA 模型推理

#### GR00T-N1.5 本地推理

```bash
python scripts/inference/inference_gr00t_isaaclab.py \
    --model_path ./pretrained_models/gr00t_place_bin \
    --num_episodes 5 \
    --save_video \
    --headless
```

#### ACT 本地推理

```bash
python scripts/inference/inference_act_isaaclab.py \
    --model_path ./pretrained_models/act_place_bin \
    --dataset_path ./datasets/lerobot/franka_place_bin \
    --num_episodes 5 \
    --save_video \
    --headless
```

> ACT 推理需要数据集路径来加载归一化统计量（`meta/stats.json`）。

#### GR00T 远程推理（推荐，避免环境冲突）

**Terminal 1 — 启动 GR00T 推理服务：**

```bash
conda activate isaac_gr00t
cd $GR00T_PATH
python scripts/inference_service.py \
    --model-path ./pretrained_models/gr00t_place_bin \
    --server --port 5555
```

**Terminal 2 — 启动 IsaacLab 客户端：**

```bash
conda activate isaac
cd VLA-Franka-IsaacLab
python scripts/inference/gr00t_remote_client.py \
    --server-host localhost \
    --server-port 5555 \
    --num-episodes 3 \
    --save_video
```

客户端与服务端通过 ZMQ + msgpack 通信，observation 序列化后传输，action 结果返回执行。

## 项目结构

```
VLA-Franka-IsaacLab/
├── scripts/
│   ├── data_collection/          # 数据采集脚本
│   │   ├── auto_collect_single.py    # 单方块自动采集
│   │   ├── auto_collect_stack.py     # 堆叠自动采集
│   │   └── record_demos.py           # 人工遥操作采集
│   ├── inference/                # 推理脚本
│   │   ├── inference_gr00t_isaaclab.py   # GR00T 本地推理
│   │   ├── inference_act_isaaclab.py     # ACT 本地推理
│   │   ├── gr00t_remote_client.py        # GR00T 远程客户端
│   │   └── static_trajectory_eval.py     # 静态轨迹评估
│   ├── model_training/           # 训练脚本
│   │   └── train_act_with_lerobot.py
│   └── debug/                    # 调试工具
│       └── check_installation.py     # 安装验证
├── tasks/
│   └── franka/                   # 自定义 IsaacLab 环境
│       ├── place_bin_ik_rel_env_cfg.py   # ⭐ 核心自定义环境
│       └── ...                           # 其他 IsaacLab 模板环境
├── configs/                      # 推理配置示例
│   ├── gr00t_place_bin.yaml
│   └── act_place_bin.yaml
├── docs/                         # 详细文档
├── requirements-isaac.txt        # IsaacLab 环境依赖
├── requirements-gr00t.txt        # GR00T 环境依赖
├── requirements-train.txt        # 训练环境依赖
├── pyproject.toml
├── Makefile
└── README.md
```

## 自定义任务环境

本项目的**核心自定义环境**是 `Isaac-Place-Bin-Franka-IK-Rel-v0`，在 `tasks/franka/place_bin_ik_rel_env_cfg.py` 中定义：

| 任务 ID | 说明 | 控制方式 |
|---|---|---|
| `Isaac-Place-Bin-Franka-IK-Rel-v0` | 放方块到篮子 | IK 相对控制 |

该环境在 IsaacLab 标准模板基础上做了以下扩展：
- 三摄像头观测（table_cam, table_cam_side, wrist_cam）
- EEF 位姿、夹爪状态、关节位置观测
- 无终止条件（便于数据采集和推理评估）
- 支持 Mimic 子任务标注（grasp / lift / place）

> `tasks/franka/` 下还包含其他来自 IsaacLab 模板的环境（Stack、Lift 等），如需使用可直接注册，但本项目所有脚本默认仅使用 **Place-Bin** 环境。

## 常见问题

### 1. `ModuleNotFoundError: No module named 'gr00t'`

GR00T 未安装。安装方式见上文 [环境 B — GR00T 推理服务](#环境-b--gr00t-推理服务)。

### 2. 图像方向与训练时不一致

Isaac Sim 实时渲染的图像可能与训练视频存在水平镜像差异。推理脚本中已经默认进行了水平翻转（`img[:, ::-1, :]`）。如果训练数据本身已经翻转过，可以禁用：

```bash
python scripts/inference/gr00t_remote_client.py --no-flip
```

### 3. 动作输出接近 0

ACT 的 temporal ensemble 可能导致动作塌陷。推理脚本中已禁用 temporal ensemble，改用 action queue。如需调整，可修改 `n_action_steps` 参数：

```bash
python scripts/inference/inference_act_isaaclab.py --n-action-steps 10
```

### 4. GR00T 和 IsaacLab 环境冲突

GR00T 依赖的 transformers / torch 版本可能与 IsaacLab 不兼容。推荐方案：
- 使用远程推理架构（`gr00t_remote_client.py` + `inference_service.py`）
- 或分别创建 conda 环境

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

项目中使用的第三方库：
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) - BSD-3-Clause
- [LeRobot](https://github.com/huggingface/lerobot) - Apache-2.0
- [GR00T-N1.5](https://github.com/NVIDIA/GR00T) - 请参考官方许可证
