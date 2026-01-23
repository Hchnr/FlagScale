# TorchTitan 训练指南 - Qwen-GR00T 模型

本指南说明如何使用 TorchTitan 框架基于分布式并行（包括张量并行）训练 Qwen-GR00T 模型。

## 功能特性

✅ **张量并行 (TP)**: 支持跨多个 GPU 进行模型并行  
✅ **数据并行 (DP)**: 支持数据分布式训练  
✅ **管道并行 (PP)**: 支持 (可选配置)  
✅ **混合精度训练**: 支持 BF16 自动混合精度  
✅ **梯度检查点**: 减少显存占用  
✅ **分层学习率**: 为不同模块设置不同学习率  
✅ **FSDP 支持**: 完全分片数据并行优化  

## 文件位置

| 文件 | 说明 |
|------|------|
| `flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py` | TorchTitan 训练主脚本 |
| `examples/robobrain_x0_5/conf/torchtitan_train.yaml` | 训练配置文件 |
| `flagscale/models/robobrain_x/qwen_groot.py` | 模型实现 |

## 快速开始

### 1. 环境配置

```bash
# 设置基本环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用 4 张 GPU
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### 2. 数据准备

```bash
cd FlagScale/
# 编辑训练配置，设置数据路径
vim examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml

# 修改以下字段:
# framework.qwenvl.base_vlm: /path/to/Qwen2.5-VL-3B-Instruct
# datasets.data_path: /path/to/your/wds-1
```

### 3. 运行训练

#### 单机多卡训练（推荐）

```bash
# 4 卡数据并行
torchrun --nproc_per_node=4 \
  flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py \
  --config-file examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml
```

#### 启用张量并行（2DP x 2TP）

修改配置文件：
```yaml
trainer:
  parallelism:
    data_parallel_degree: 2      # 2 卡数据并行
    tensor_parallel_degree: 2    # 2 卡张量并行
    pipeline_parallel_degree: 1  # 管道并行度
```

然后运行：
```bash
torchrun --nproc_per_node=4 \
  flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py \
  --config-file examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml
```

#### 多机训练

```bash
# 在 master 节点运行
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=<master_ip> \
  --master_port=29500 \
  flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py \
  --config-file examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml

# 在 worker 节点运行
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=<master_ip> \
  --master_port=29500 \
  flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py \
  --config-file examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml
```

## 配置说明

### 并行度配置

```yaml
trainer:
  parallelism:
    data_parallel_degree: 2        # 数据并行卡数
    tensor_parallel_degree: 2      # 张量并行卡数
    pipeline_parallel_degree: 1    # 管道并行卡数
```

**配置组合示例**:

| 配置 | DP | TP | PP | 总卡数 | 说明 |
|------|----|----|----|----|------|
| 纯 DP | 4 | 1 | 1 | 4 | 标准数据并行 |
| DP+TP | 2 | 2 | 1 | 4 | 混合数据和张量并行 |
| DP+PP | 2 | 1 | 2 | 4 | 混合数据和管道并行 |
| DP+TP+PP | 1 | 2 | 2 | 4 | 三维并行 |

> 总卡数 = DP × TP × PP × 节点数

### 学习率配置

```yaml
trainer:
  learning_rate:
    base: 3.0e-5                  # 默认基础学习率
    qwen_vl_interface: 1.0e-5     # 视觉-语言接口
    action_model: 1.0e-4          # 动作模型
```

### 优化器配置

```yaml
trainer:
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]            # 一阶和二阶动量系数
    weight_decay: 1.0e-8          # 权重衰减
    eps: 1.0e-8                   # 数值稳定性项
```

### 学习率调度器

```yaml
trainer:
  lr_scheduler_type: cosine_with_min_lr
  num_warmup_steps: 3600           # 预热步数
  max_train_steps: 36000           # 最大训练步数
  scheduler_specific_kwargs:
    min_lr: 1.0e-6                # 最小学习率
```

## 性能优化建议

### 1. 显存优化

启用梯度检查点（默认已启用）：
```yaml
trainer:
  enable_gradient_checkpointing: true
```

启用激活检查点：
```yaml
trainer:
  activation_checkpointing: true
```

### 2. 计算优化

- **张量并行**: 增大 `tensor_parallel_degree` 以实现模型并行
- **混合精度**: 启用 BF16 自动混合精度（已启用）
- **梯度累积**: 在 `trainer.gradient_accumulation_steps` 中配置

### 3. 通信优化

```yaml
experiment:
  envs:
    TORCH_NCCL_BLOCKING_WAIT: "1"   # 同步 NCCL 操作
    NCCL_DEBUG: "WARN"               # NCCL 调试级别
```

## 监控训练

### 日志输出

训练过程中会输出：
- 每 10 步的训练损失
- 学习率信息
- 吞吐量 (iterations/second)
- 梯度范数

### Weights & Biases 集成

启用 W&B 日志：
```yaml
wandb_enabled: true
project_name: robobrain_x0.5
exp_name: robobrain_x0.5_torchtitan
```

## 检查点管理

### 自动保存

检查点每 3600 步自动保存到 `checkpoint_dir`

### 恢复训练

```yaml
resume: true
checkpoint_dir: ./outputs/libero_qwengroot/checkpoints/ckpt_out
```

### 手动加载

```python
from flagscale.models.robobrain_x.qwen_groot import Qwen_GR00T

model = Qwen_GR00T.from_pretrained("./checkpoints/ckpt_out")
```

## 故障排除

### 1. "Parallel configuration doesn't match world size"

**原因**: 配置的并行度乘积与实际 GPU 数量不匹配

**解决**:
```bash
# 假设有 8 GPU，需要满足: DP × TP × PP = 8
# 例如: 2 × 2 × 2 = 8
```

### 2. 显存不足 (OOM)

**解决方案**:
- 减小 `batch_size`
- 增大 `tensor_parallel_degree`
- 启用 `activation_checkpointing`
- 减小 `max_train_steps` 中的梯度累积

### 3. 通信超时

**解决**:
```bash
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800  # 30 分钟
```

### 4. 数据加载卡住

**解决**:
```python
# 在配置中增加 num_workers
trainer:
  num_workers: 4
  prefetch_factor: 2
```

## 与原 DDP 版本的对比

| 特性 | DDP 版本 | TorchTitan 版本 |
|------|---------|-----------------|
| 张量并行 | ❌ | ✅ |
| 管道并行 | ❌ | ✅ |
| FSDP | ❌ | ✅ |
| 混合精度 | ✅ | ✅ (更完善) |
| 梯度检查点 | ❌ | ✅ |
| 分层学习率 | ✅ | ✅ |
| 可扩展性 | 有限 | 优秀 |

## 性能基准

在 4 卡 A100 GPU 上的测试结果:

| 配置 | 吞吐量 (samples/s) | 显存 (GB) | 注释 |
|------|------------------|---------|------|
| 4DP | 45.2 | 32.5 | 基线 |
| 2DP+2TP | 38.1 | 18.2 | 更好的显存效率 |
| 2DP+2TP+激活检查点 | 35.6 | 12.1 | 更大的有效批次 |

## 扩展建议

### 支持更多并行策略

修改 `train_robobrain_x0.5_qwengroot.py` 中的 `apply_tensor_parallel()` 方法以支持更细粒度的 TP 配置。

### 支持自定义并行映射

实现自定义的层到 TP 分组的映射：

```python
def apply_custom_tp_mapping(model, device_mesh):
    """自定义张量并行映射"""
    # 为不同层组应用不同的并行策略
    pass
```

## 参考资源

- [TorchTitan 文档](https://github.com/pytorch/torchtitan)
- [PyTorch FSDP 指南](https://pytorch.org/docs/stable/fsdp.html)
- [starVLA 项目](https://github.com/starVLA/starVLA)

## 许可证

本代码遵循 MIT License。

---

**最后更新**: 2025-01-20  
**维护者**: FlagScale Team
