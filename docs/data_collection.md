# 数据采集指南

## 自动化数据采集

### 单方块 Pick-Place

```bash
python scripts/data_collection/auto_collect_single.py \
    --num_episodes 100 \
    --output_dir datasets/lerobot/franka_place_bin \
    --repo_id local/franka_place_bin \
    --use_videos \
    --target_color green \
    --overwrite
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_episodes` | 10 | 目标成功 episode 数量 |
| `--target_color` | green | 目标方块：blue / red / green |
| `--use_videos` | False | 存为 mp4（推荐，体积小） |
| `--save_failed` | False | 是否保存失败 episode |
| `--steps_per_episode` | 180 | 每 episode 最大步数 |
| `--overwrite` | False | 覆盖已有数据集 |

**采集原理：**

脚本内置 `PickPlaceController`，使用闭环比例控制生成平滑轨迹：
1. Approach（接近方块上方）
2. Grasp（下降抓取）
3. Close gripper（关闭夹爪，等待）
4. Lift（抬起）
5. Approach place（接近放置点上方）
6. Place（下降放置）
7. Open gripper（打开夹爪，等待）

每步根据当前 EEF 位置与目标点的误差计算动作，成功率通常 > 90%。

### 多方块堆叠

```bash
python scripts/data_collection/auto_collect_stack.py \
    --num_episodes 100 \
    --output_dir datasets/lerobot/franka_stack \
    --repo_id local/franka_stack \
    --use_videos \
    --overwrite
```

抓取顺序：蓝(cube_1) → 红(cube_2) → 绿(cube_3)，依次堆叠到篮子中。

成功判定条件：
- 三个方块都在篮子中（XY 距离 < 3cm）
- 三个方块 XY 对齐（两两之间 < 2cm）
- Z 高度合理递增（每层约 4.0~5.5cm）

### 人工遥操作采集

如果需要更高质量的人类演示数据：

```bash
python scripts/data_collection/record_demos.py \
    --task Isaac-Place-Bin-Franka-IK-Rel-v0 \
    --dataset_file datasets/franka_place_bin.hdf5 \
    --teleop_device keyboard \
    --step_hz 30 \
    --num_demos 50
```

键盘控制映射：
- **I/K**: X 前后
- **J/L**: Y 左右
- **U/O**: Z 上下
- **N/M**: X 旋转
- **T/G**: Y 旋转
- **Y/B**: Z 旋转
- **P**: 夹爪开关
- **E**: 导出当前 episode（标记为成功）
- **R**: 重置环境
- **F9**: 开始/暂停录制

## 数据格式

自动化采集脚本直接输出 **LeRobot v3.0** 格式：

```
datasets/lerobot/franka_place_bin/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── stats.json         # 归一化统计量
│   └── tasks.jsonl        # 任务描述
├── data/
│   ├── chunk-000/         # 数据分片
│   └── ...
└── videos/                # 视频文件（--use_videos 时）
    ├── table_cam/
    ├── table_cam_side/
    └── wrist_cam/
```

Features 定义：
- `action`: (7,) - [dx, dy, dz, droll, dpitch, dyaw, gripper]
- `observation.state`: (18,) - [eef_pos(3), eef_quat(4), gripper_pos(2), joint_pos(9)]
- `observation.images.{cam}`: 摄像头图像 (3, 224, 224)

## 数据质量检查

采集完成后，可以查看统计信息：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/franka_place_bin", root="datasets/lerobot/franka_place_bin")
print(f"Episodes: {ds.num_episodes}")
print(f"Frames: {ds.num_frames}")
print(f"FPS: {ds.fps}")
```
