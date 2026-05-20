# 远程推理架构

## 为什么需要远程推理？

GR00T-N1.5 依赖的 `transformers`、`torch` 等库版本可能与 IsaacLab 的依赖冲突。例如：

- IsaacLab 需要 `transformers < 4.40`
- GR00T 需要 `transformers >= 4.45`

如果安装在同一环境中，会导致导入错误。

**解决方案**：将 GR00T 推理服务与 IsaacLab 仿真环境解耦，通过 ZMQ 网络通信。

```
+-------------------+     ZMQ (TCP)      +-------------------+
|  GR00T Server     | <----------------> |  IsaacLab Client  |
|  (gr00t env)      |   observation      |  (isaac env)      |
|  GPU: CUDA 12.1   |   action           |  GPU: CUDA 12.4   |
|  transformers 4.45|                    |  transformers 4.38|
+-------------------+                    +-------------------+
```

## 服务端启动

在 `gr00t` conda 环境中启动推理服务：

```bash
conda activate gr00t
cd $GR00T_PATH  # GR00T-N1.5 安装目录

python scripts/inference_service.py \
    --model-path /path/to/gr00t_place_bin \
    --server \
    --port 5555
```

服务启动后会监听 `tcp://*:5555`，暴露以下端点：

| 端点 | 说明 |
|---|---|
| `get_modality_config` | 获取模态配置 |
| `get_action` | 输入 observation，输出 action |
| `ping` | 健康检查 |

## 客户端启动

在 `isaac` conda 环境中运行仿真客户端：

```bash
conda activate isaac
cd vla-franka-isaaclab

python scripts/inference/gr00t_remote_client.py \
    --server-host localhost \
    --server-port 5555 \
    --num-episodes 3 \
    --save_video \
    --headless \
    --action-horizon 4 \
    --action-smoothing 0.3
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server-host` | localhost | GR00T 服务主机地址 |
| `--server-port` | 5555 | GR00T 服务端口 |
| `--action-horizon` | 4 | GR00T action chunk 大小 |
| `--action-smoothing` | 0.3 | EMA 动作平滑因子（0=不平滑） |
| `--no-flip` | False | 禁用图像水平翻转 |
| `--episode-length` | 300 | 每 episode 最大步数 |

## 跨机器部署

如果服务端和客户端在不同机器上：

**服务端机器：**

```bash
python scripts/inference_service.py \
    --model-path ./pretrained_models/gr00t_place_bin \
    --server \
    --port 5555 \
    --host 0.0.0.0
```

**客户端机器：**

```bash
python scripts/inference/gr00t_remote_client.py \
    --server-host 192.168.1.100 \
    --server-port 5555 \
    --num-episodes 3
```

> 确保防火墙允许 5555 端口通信。

## 性能优化

### Action Chunking

GR00T 每次推理输出一个 action chunk（默认 16 步）。客户端可以：

1. **逐帧请求**（`--action-horizon 1`）：每步都向服务端请求，精度高但延迟大
2. **Chunk 复用**（默认）：请求一次，复用 chunk 中的多步动作，延迟小但可能累积误差

推荐设置：
- 本地测试：`--action-horizon 4`，平衡精度与延迟
- 远程部署：`--action-horizon 8~16`，减少网络往返

### 图像压缩

如果网络带宽有限，可以在 `IsaacObsConverter.convert()` 中降低图像分辨率：

```python
img = cv2.resize(img, (112, 112))  # 从 224x224 降为 112x112
```

## 故障排查

### 连接超时

```
zmq.error.Again: Resource temporarily unavailable
```

- 检查服务端是否启动：`python -c "import zmq; ..."` 测试连通性
- 检查防火墙设置
- 增大客户端超时：`Gr00tRemoteClient(timeout_ms=60000)`

### 图像尺寸不匹配

```
Server error: observation image shape mismatch
```

- 检查客户端和服务端的 `data_config` 是否一致
- 确认摄像头分辨率与训练时相同（默认 224x224）

### 动作方向错误

- 检查 `--no-flip` 参数
- 对比训练和推理的图像方向是否一致
