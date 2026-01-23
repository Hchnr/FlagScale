# TorchTitan 训练 - 快速参考

## 🚀 快速开始

```bash
# 方案 1: 使用启动脚本（推荐）
./examples/robobrain_x0_5/launch_torchtitan_training.sh --num-gpus 4

# 方案 2: 直接使用 torchrun
torchrun --nproc_per_node=4 \
  flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py \
  --config-file examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml
```

## 📋 常用配置

### 2DP × 2TP (4 GPU)
```yaml
trainer:
  parallelism:
    data_parallel_degree: 2
    tensor_parallel_degree: 2
    pipeline_parallel_degree: 1
batch_size: 2
```

### 4DP × 1TP (4 GPU, 纯 DP)
```yaml
trainer:
  parallelism:
    data_parallel_degree: 4
    tensor_parallel_degree: 1
    pipeline_parallel_degree: 1
batch_size: 4
```

### 2DP × 2TP × 2PP (8 GPU)
```yaml
trainer:
  parallelism:
    data_parallel_degree: 2
    tensor_parallel_degree: 2
    pipeline_parallel_degree: 2
batch_size: 1
```

## 🛠️ 环境变量

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
```

## 📊 监控训练

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f outputs/libero_qwengroot_torchtitan/checkpoints/train.log

# 打开 W&B 仪表板
# https://wandb.ai/your-entity/robobrain_x0.5
```

## ✅ 验证配置

```bash
# 检查配置文件
cat examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml

# 测试模型加载
python -c "from flagscale.models.robobrain_x.qwen_groot import Qwen_GR00T; print('Model loaded OK')"
```

## 🔧 常见问题排查

| 问题 | 解决方案 |
|------|--------|
| OOM 显存不足 | 增大 TP 度数，减小 batch_size |
| GPU 利用率低 | 增大 batch_size，检查数据加载 |
| 通信超时 | 设置 NCCL_TIMEOUT=1800 |
| 数据加载慢 | 增加 num_workers，设置 prefetch_factor |

## 📈 性能优化

### 显存优化
```yaml
trainer:
  enable_gradient_checkpointing: true  # 减少 ~30% 显存
  activation_checkpointing: true       # 进一步优化
```

### 通信优化
```yaml
trainer:
  parallelism:
    tensor_parallel_degree: 2  # 跨少数 GPU
```

### 计算优化
```yaml
trainer:
  enable_mixed_precision_training: true
  optimizer:
    betas: [0.9, 0.95]  # 较好的收敛性
```

## 📚 详细文档

- **训练指南**: [TORCHTITAN_TRAINING.md](./TORCHTITAN_TRAINING.md)
- **DDP vs TorchTitan**: [DDP_vs_TORCHTITAN.md](./DDP_vs_TORCHTITAN.md)
- **配置示例**: [conf/train/torchtitan_config_examples.yaml](./conf/train/torchtitan_config_examples.yaml)

## 🎯 下一步

1. ✅ 配置数据路径
2. ✅ 调整并行度
3. ✅ 运行训练
4. ✅ 监控日志
5. ✅ 保存检查点

## 💡 提示

- 开始时使用小的并行度（2DP × 2TP）
- 根据显存使用情况逐步调整
- 监视显存避免 OOM
- 定期保存检查点
- 使用 W&B 跟踪实验

---

**更新时间**: 2025-01-20  
**支持**: TorchTitan, PyTorch 2.0+
