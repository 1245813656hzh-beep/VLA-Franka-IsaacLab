# 环境搭建指南

## 前置要求

- Ubuntu 22.04 LTS
- NVIDIA GPU (推荐 RTX 4090 / A6000 / A100，VRAM >= 12GB)
- NVIDIA Driver >= 535
- CUDA 12.x

## 安装 Isaac Sim

参考 [NVIDIA Isaac Sim 官方文档](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html)。

推荐通过 Omniverse Launcher 安装，或使用 pip：

```bash
pip install isaacsim==4.2.0 --extra-index-url https://pypi.nvidia.com
```

## 安装 IsaacLab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

## 安装本项目

```bash
cd /path/to/vla-franka-isaaclab
pip install -e .
```

## 安装 LeRobot

```bash
# 方式一：pip 直接安装
pip install lerobot

# 方式二：从源码安装（如需修改）
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[aloha]"
```

如果使用方式二，可以将 `lerobot` 目录放在本项目根目录下，脚本会自动识别。

## 安装 GR00T-N1.5（可选）

```bash
mkdir -p third_party
git clone https://github.com/NVIDIA/GR00T.git third_party/GR00T-N1.5
cd third_party/GR00T-N1.5
pip install -e .
```

或设置环境变量：

```bash
export GR00T_PATH=/path/to/GR00T-N1.5
```

## 验证安装

```bash
python scripts/debug/quick_env_check.py --task Isaac-Place-Bin-Franka-IK-Rel-v0
```

如果环境能正常启动并显示相机画面，说明安装成功。
