# 推理指南

## GR00T-N1.5 本地推理

### 准备模型

将微调好的 GR00T 模型放到 `pretrained_models/` 目录：

```
pretrained_models/
└── gr00t_place_bin/
    ├── config.json
    ├── model.safetensors
    └── ...
```

### 运行推理

```bash
python scripts/inference/inference_gr00t_isaaclab.py \
    --model_path ./pretrained_models/gr00t_place_bin \
    --num_episodes 5 \
    --save_video \
    --headless \
    --task_description "pick up the green cube and place it into the blue bin"
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | - | GR00T 模型目录（必填） |
| `--num_episodes` | 5 | 推理 episode 数 |
| `--episode_length` | 250 | 每 episode 最大步数 |
| `--denoising_steps` | 4 | GR00T action head 去噪步数 |
| `--task_description` | - | 语言任务描述 |
| `--save_video` | False | 保存视频 |
| `--headless` | False | 无头模式 |
| `--verbose` | False | 详细调试日志 |

### 图像翻转说明

Isaac Sim 实时渲染的图像与训练视频存在水平镜像差异。`inference_gr00t_isaaclab.py` 中已经默认进行水平翻转：

```python
img = img[:, ::-1, :].copy()
```

如果训练数据已经预处理过翻转，需要移除这行代码。

## ACT 本地推理

### 准备模型

ACT 模型通过 LeRobot 训练保存，目录结构：

```
pretrained_models/
└── act_place_bin/
    ├── config.json
    ├── model.safetensors
    ├── policy_preprocessor.json
    └── policy_postprocessor.json
```

### 运行推理

```bash
python scripts/inference/inference_act_isaaclab.py \
    --model_path ./pretrained_models/act_place_bin \
    --dataset_path ./datasets/lerobot/franka_place_bin \
    --num_episodes 5 \
    --save_video \
    --headless \
    --n_action_steps 100
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | - | ACT 模型目录（必填） |
| `--dataset_path` | - | 数据集目录（用于加载 stats.json 归一化统计量） |
| `--n_action_steps` | 100 | 每个 chunk 执行的动作步数。越小 = 重规划越频繁 |
| `--episode_length` | 300 | 每 episode 最大步数 |

### 归一化统计量

ACT 推理需要从训练数据集中加载 `meta/stats.json` 进行动作反归一化：

```json
{
  "action": {
    "mean": [...],
    "std": [...]
  },
  "observation.state": {
    "mean": [...],
    "std": [...]
  }
}
```

如果 `stats.json` 不存在，脚本会回退到 identity normalization 并打印警告。

## 调试工具

### 检查环境观测

```bash
python scripts/debug/diagnose_gr00t_obs.py --task Isaac-Place-Bin-Franka-IK-Rel-v0
```

### 对比训练与推理的观测差异

```bash
python scripts/debug/debug_env_vs_training.py \
    --dataset_path ./datasets/lerobot/franka_place_bin
```

### 单步 ACT 调试

```bash
python scripts/debug/debug_act_step.py \
    --model_path ./pretrained_models/act_place_bin \
    --dataset_path ./datasets/lerobot/franka_place_bin
```
