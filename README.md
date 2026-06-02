# VLA-Franka-IsaacLab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [中文](README.zh.md)

A VLA (Vision-Language-Action) data collection and inference toolkit for the Franka robot on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) / [IsaacLab](https://github.com/isaac-sim/IsaacLab).

<p align="center">
  <b>Data Collection → Training → Inference</b>
</p>

## Demo — Blue, Red & Green Cube Stacking

<p align="center">
  <video src="blue_red_green/ep002_table_cam.mp4" controls muted width="32%"></video>
  <video src="blue_red_green/ep002_table_cam_side.mp4" controls muted width="32%"></video>
  <video src="blue_red_green/ep002_wrist_cam.mp4" controls muted width="32%"></video>
</p>

<p align="center">
  <sub><b>Left:</b> Table Cam (front) &nbsp;|&nbsp; <b>Middle:</b> Side Cam &nbsp;|&nbsp; <b>Right:</b> Wrist Cam</sub>
</p>

## Features

- 🤖 **Automated Data Collection**
  - Single-cube pick-and-place with scripted trajectory generation
  - Multi-cube stacking in the same scene
  - Direct LeRobot v3.0 dataset output
  - Built-in closed-loop controller with success checking

- 🧠 **VLA Model Inference**
  - **GR00T-N1.5** (NVIDIA) local inference
  - **ACT** (Action Chunking Transformer) local inference
  - **GR00T remote inference** via ZMQ (decouples model server from simulation)

- 🎯 **Custom IsaacLab Environment**
  - **Place-Bin** task — place scattered cubes into a blue sorting bin
  - Multi-camera setup: Table Cam / Side Cam / Wrist Cam
  - Keyboard teleoperation support

## Requirements

| Component | Version | Note |
|---|---|---|
| Ubuntu | 22.04 | Required |
| NVIDIA GPU | >= 12 GB VRAM | For Isaac Sim + model inference |
| NVIDIA Driver | >= 535 | CUDA 12.x required |
| Isaac Sim | 4.2+ | Simulation engine; install separately |
| IsaacLab | >= 0.48.0 | Robot learning framework |
| Python | 3.10 / 3.11 | 3.11 for IsaacLab, 3.10 for GR00T |
| CUDA | 12.x | Match your Isaac Sim version |

> ⚠️ **Version Compatibility**: This project is validated with:
> - IsaacLab 0.48.5 + isaaclab-mimic 1.0.15 + isaaclab-tasks 0.11.8
> - LeRobot 0.4.4
> - transformers 4.40+ (isaac env), 4.45~4.51 (gr00t env)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/1245813656hzh-beep/VLA-Franka-IsaacLab.git
cd VLA-Franka-IsaacLab
```

### 2. Install NVIDIA Isaac Sim

This is the largest prerequisite and **must be done first**:

```bash
# Option 1: Omniverse Launcher (recommended)
# Download from https://www.nvidia.com/omniverse and install Isaac Sim 4.2+

# Option 2: pip
pip install isaacsim==4.2.0 --extra-index-url https://pypi.nvidia.com
```

Set the environment variable:
```bash
export ISAACSIM_PATH=/path/to/isaac-sim  # e.g. ~/isaacsim
```

### 3. Install IsaacLab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
# Or via pip:
pip install isaaclab isaaclab-tasks isaaclab-assets isaaclab-mimic
```

### 4. Install project dependencies

This project uses **multiple conda environments** due to dependency conflicts. Create them by function:

#### Env A — IsaacLab Simulation (data collection + ACT inference)

```bash
conda create -n isaac python=3.11
conda activate isaac
pip install -e .
pip install -r requirements-isaac.txt
python scripts/debug/check_installation.py
```

#### Env B — GR00T Inference Server

```bash
conda create -n isaac_gr00t python=3.10
conda activate isaac_gr00t
pip install -e .

# Install GR00T-N1.5 from source
mkdir -p third_party
git clone https://github.com/NVIDIA/GR00T.git third_party/GR00T-N1.5
cd third_party/GR00T-N1.5
pip install -e .

pip install -r ../../requirements-gr00t.txt
python -c "import gr00t; print('GR00T OK')"
```

#### Env C — ACT Training

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -e ".[train]"
pip install -r requirements-train.txt
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobot OK')"
```

> If you only need data collection and ACT inference, **Env A alone is sufficient**.

## Quick Start

### Automated Data Collection

#### Single-cube pick-and-place

```bash
python scripts/data_collection/auto_collect_single.py \
    --num_episodes 100 \
    --output_dir datasets/lerobot/franka_place_bin \
    --repo_id local/franka_place_bin \
    --use_videos \
    --target_color green
```

Key arguments:
- `--num_episodes`: target number of successful episodes
- `--target_color`: target cube color (`blue`, `red`, `green`)
- `--use_videos`: store images as mp4 (smaller size, recommended)
- `--save_failed`: also save failed episodes

#### Multi-cube stacking

```bash
python scripts/data_collection/auto_collect_stack.py \
    --num_episodes 100 \
    --output_dir datasets/lerobot/franka_stack \
    --repo_id local/franka_stack \
    --use_videos
```

Pick order: blue → red → green, stacked sequentially into the bin.

### Human Teleoperation

```bash
python scripts/data_collection/record_demos.py \
    --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
    --dataset_file datasets/franka_place_bin.hdf5 \
    --teleop_device keyboard \
    --step_hz 30
```

Keyboard mapping:
- **I/K**: X forward/back | **J/L**: Y left/right | **U/O**: Z up/down
- **N/M**: X rotation | **T/G**: Y rotation | **Y/B**: Z rotation
- **P**: gripper toggle | **E**: export episode | **R**: reset

### Model Training

#### ACT with LeRobot

```bash
python scripts/model_training/train_act_with_lerobot.py \
    --dataset-root ./datasets/lerobot/franka_place_bin \
    --output-dir ./outputs/act_place_bin \
    --steps 50000 \
    --batch-size 64 \
    --device cuda
```

### VLA Inference

#### GR00T-N1.5 local inference

```bash
python scripts/inference/inference_gr00t_isaaclab.py \
    --model_path ./pretrained_models/gr00t_place_bin \
    --num_episodes 5 \
    --save_video \
    --headless
```

#### ACT local inference

```bash
python scripts/inference/inference_act_isaaclab.py \
    --model_path ./pretrained_models/act_place_bin \
    --dataset_path ./datasets/lerobot/franka_place_bin \
    --num_episodes 5 \
    --save_video \
    --headless
```

> ACT inference requires the dataset path to load normalization stats (`meta/stats.json`).

#### GR00T remote inference (recommended)

**Terminal 1 — Start GR00T server:**

```bash
conda activate isaac_gr00t
cd $GR00T_PATH
python scripts/inference_service.py \
    --model-path ./pretrained_models/gr00t_place_bin \
    --server --port 5555
```

**Terminal 2 — Start IsaacLab client:**

```bash
conda activate isaac
cd VLA-Franka-IsaacLab
python scripts/inference/gr00t_remote_client.py \
    --server-host localhost \
    --server-port 5555 \
    --num-episodes 3 \
    --save_video
```

## Project Structure

```
VLA-Franka-IsaacLab/
├── scripts/
│   ├── data_collection/
│   │   ├── auto_collect_single.py
│   │   ├── auto_collect_stack.py
│   │   └── record_demos.py
│   ├── inference/
│   │   ├── inference_gr00t_isaaclab.py
│   │   ├── inference_act_isaaclab.py
│   │   ├── gr00t_remote_client.py
│   │   └── static_trajectory_eval.py
│   ├── model_training/
│   │   └── train_act_with_lerobot.py
│   └── debug/
│       └── check_installation.py
├── tasks/
│   └── franka/
│       ├── place_bin_ik_rel_env_cfg.py   # ⭐ Core custom environment
│       └── ...                           # Other IsaacLab template tasks
├── configs/
│   ├── gr00t_place_bin.yaml
│   └── act_place_bin.yaml
├── docs/
├── requirements-isaac.txt
├── requirements-gr00t.txt
├── requirements-train.txt
├── pyproject.toml
├── Makefile
└── README.md
```

## Custom Task Environment

The **core custom environment** of this project is `Isaac-Place-Bin-Franka-IK-Rel-v0`, defined in `tasks/franka/place_bin_ik_rel_env_cfg.py`:

| Task ID | Description | Control |
|---|---|---|
| `Isaac-Place-Bin-Franka-IK-Rel-v0` | Pick up cubes and place them into a blue sorting bin | IK relative pose |

Extensions over the IsaacLab standard template:
- Three camera observations (table_cam, table_cam_side, wrist_cam)
- EEF pose, gripper state, and joint position observations
- No termination conditions (for data collection and inference evaluation)
- Mimic subtask annotation support (grasp / lift / place)

> `tasks/franka/` also contains other IsaacLab template environments (Stack, Lift, etc.). They can be registered if needed, but all scripts in this repo default to the **Place-Bin** environment only.

## FAQ

### 1. `ModuleNotFoundError: No module named 'gr00t'`

GR00T is not installed. See [Install GR00T-N1.5](#env-b--gr00t-inference-server) above.

### 2. Image orientation mismatch between training and inference

Isaac Sim live rendering may produce horizontally mirrored images compared to training videos. The inference scripts apply a horizontal flip by default (`img[:, ::-1, :]`). If your training data was already flipped, disable it:

```bash
python scripts/inference/gr00t_remote_client.py --no-flip
```

### 3. Action output near zero

ACT's temporal ensemble can cause action collapse. Our inference scripts disable temporal ensemble and use action queue. Adjust `n_action_steps` if needed:

```bash
python scripts/inference/inference_act_isaaclab.py --n-action-steps 10
```

### 4. GR00T and IsaacLab dependency conflicts

GR00T requires different `transformers` / `torch` versions than IsaacLab. Solutions:
- Use remote inference (`gr00t_remote_client.py` + `inference_service.py`)
- Or maintain separate conda environments

## License

This project is licensed under the [MIT License](LICENSE).

Third-party libraries used:
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) - BSD-3-Clause
- [LeRobot](https://github.com/huggingface/lerobot) - Apache-2.0
- [GR00T-N1.5](https://github.com/NVIDIA/GR00T) - Refer to official license
