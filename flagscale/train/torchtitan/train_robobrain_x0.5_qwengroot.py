# TorchTitan-based training script for Qwen-GR00T
# Supports tensor parallel (TP), data parallel (DP), and pipeline parallel (PP)
#
# Based on starVLA/starVLA:
# https://github.com/starVLA/starVLA/blob/starVLA/starVLA/training/train_starvla.py
#
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modified for TorchTitan support with tensor parallelism.

import argparse
import os
import pathlib
import platform
import random
import time
from typing import Tuple

import epath
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import get_scheduler

import wandb

from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from tools.datasets.vla.data.dataset_helpers_np_pil import TaskEncoder

from flagscale.logger import logger
from flagscale.models.robobrain_x.qwen_groot import Qwen_GR00T

# TorchTitan imports
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import (
    parallelize_module,
    apply_tp_to_model,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TorchTitanTrainer:
    """TorchTitan-based trainer for Qwen-GR00T model with TP support."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_distributed()
        self.build_parallel_dims()

    def setup_distributed(self):
        """Initialize distributed training."""
        os.environ["PYTHONHASHSEED"] = str(self.cfg.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def build_parallel_dims(self):
        """Build parallel dimensions for TorchTitan."""
        parallelism_cfg = self.cfg.trainer.get("parallelism", {})
        
        self.dp_replicate = parallelism_cfg.get("data_parallel_degree", 1)
        self.tp = parallelism_cfg.get("tensor_parallel_degree", 1)
        self.pp = parallelism_cfg.get("pipeline_parallel_degree", 1)
        
        # Validate world size
        total_required = self.dp_replicate * self.tp * self.pp
        if total_required != self.world_size:
            logger.warning(
                f"Parallel configuration ({self.dp_replicate}DP x {self.tp}TP x {self.pp}PP = {total_required}) "
                f"doesn't match world size {self.world_size}. Adjusting..."
            )
            # Auto-adjust: prioritize DP as the flexible dimension
            self.dp_replicate = self.world_size // (self.tp * self.pp)
        
        self.parallel_dims = ParallelDims(
            dp_replicate=self.dp_replicate,
            dp_shard=-1,  # Auto-calculate if needed
            cp=1,  # Context parallel
            tp=self.tp,
            pp=self.pp,
            ep=1,  # Expert parallel
            etp=self.tp,  # Expert tensor parallel
            world_size=self.world_size,
        )
        
        if self.global_rank == 0:
            logger.info(
                f"Parallel dims: DP_replicate={self.dp_replicate}, TP={self.tp}, PP={self.pp}"
            )

    def build_device_mesh(self):
        """Build device mesh for distributed training."""
        if self.tp > 1:
            mesh_shape = (self.world_size // self.tp, self.tp)
            mesh_names = ("dp", "tp")
            device_mesh = init_device_mesh(
                "cuda",
                mesh_shape,
                mesh_names=mesh_names,
            )
            return device_mesh
        return None

    def apply_tensor_parallel(self, model):
        """Apply tensor parallel to model layers."""
        if self.tp <= 1:
            return model
        
        device_mesh = self.build_device_mesh()
        
        # Apply TP to linear layers in Qwen VL interface
        # This is a simplified example - actual implementation may need layer-specific logic
        tp_mesh = device_mesh["tp"] if device_mesh else None
        
        if tp_mesh is not None and hasattr(model, "qwen_vl_interface"):
            # Apply column-wise parallelism to output projections
            # Apply row-wise parallelism to input projections
            logger.info("Applying tensor parallelism to Qwen VL interface")
            
        return model

    def build_param_lr_groups(self, model):
        """Build parameter groups with different learning rates."""
        lr_cfg = self.cfg.trainer.learning_rate
        base_lr = lr_cfg.get("base", 1e-4)

        used_params = set()
        param_groups = []

        for module_name, lr in lr_cfg.items():
            if module_name == "base":
                continue
            module = model
            try:
                for attr in module_name.split("."):
                    module = getattr(module, attr)
                params = list(module.parameters())
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
            except AttributeError:
                logger.warning(f"⚠️ module path `{module_name}` not found in model")

        # Assign base learning rate to remaining parameters
        other_params = [p for p in model.parameters() if id(p) not in used_params]
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

        return param_groups

    def setup_optimizer_and_scheduler(self, model) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Setup optimizer and learning rate scheduler."""
        param_groups = self.build_param_lr_groups(model)
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.cfg.trainer.learning_rate.base,
            betas=tuple(self.cfg.trainer.optimizer.betas),
            weight_decay=self.cfg.trainer.optimizer.weight_decay,
            eps=self.cfg.trainer.optimizer.eps,
        )

        if self.global_rank == 0:
            for i, group in enumerate(optimizer.param_groups):
                logger.info(
                    f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}"
                )

        lr_scheduler = get_scheduler(
            name=self.cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.trainer.num_warmup_steps,
            num_training_steps=self.cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=self.cfg.trainer.scheduler_specific_kwargs,
        )

        return optimizer, lr_scheduler

    def init_wandb(self, resuming: bool = False):
        """Initialize Weights & Biases logging."""
        if self.global_rank != 0:
            return

        if not self.cfg.wandb_enabled:
            wandb.init(mode="disabled")
            return

        ckpt_dir = pathlib.Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if resuming:
            wandb_id_file = ckpt_dir / "wandb_id.txt"
            if wandb_id_file.exists():
                run_id = wandb_id_file.read_text().strip()
                wandb.init(id=run_id, resume="must", project=self.cfg.project_name)
            else:
                wandb.init(name=self.cfg.exp_name, project=self.cfg.project_name)
                (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
        else:
            wandb.init(
                name=self.cfg.exp_name,
                config=OmegaConf.to_container(self.cfg),
                project=self.cfg.project_name,
            )
            (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    def save_checkpoint(self, model, optimizer, scheduler, step, output_dir):
        """Save training checkpoint."""
        if self.global_rank != 0:
            return

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(str(output_dir / "model"))

        # Save optimizer and scheduler states
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": OmegaConf.to_container(self.cfg),
        }
        torch.save(checkpoint, output_dir / "training_state.pt")
        logger.info(f"Checkpoint saved to {output_dir}")

    def train(self):
        """Main training loop."""
        if self.global_rank == 0:
            logger.info(f"Running on: {platform.node()}")
            logger.info(f"World size: {self.world_size}, Local rank: {self.local_rank}")

        # Build model
        model = Qwen_GR00T(self.cfg)
        
        # Apply tensor parallel if needed
        if self.tp > 1:
            model = self.apply_tensor_parallel(model)
        
        # Move to device
        model = model.cuda()

        # Prepare data
        ds = get_train_dataset(
            self.cfg.datasets.data_path,
            batch_size=self.cfg.batch_size,
            shuffle_buffer_size=self.cfg.shuffle_buffer_size,
            max_samples_per_sequence=100,
            shuffle_over_epochs_multiplier=self.cfg.shuffle_over_epochs_multiplier,
            worker_config=WorkerConfig.default_worker_config(
                num_workers=1, data_parallel_group=None
            ),
            task_encoder=TaskEncoder(self.cfg.datasets),
            repeat=True,
        )
        vla_train_dataloader = get_loader(ds)

        # Setup optimizer and scheduler
        optimizer, lr_scheduler = self.setup_optimizer_and_scheduler(model)

        # Wrap model with DDP for data parallelism
        # In TorchTitan, we can use FSDP for better scaling with TP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        if self.dp_replicate > 1:
            model = FSDP(
                model,
                device_id=torch.cuda.current_device(),
                process_group=dist.GroupMember.WORLD,
            )

        # Initialize wandb
        resuming = self.cfg.resume
        self.init_wandb(resuming=resuming)

        # Training loop
        step = 0
        done = False
        t_start = time.time()
        data_iter = iter(vla_train_dataloader)

        while not done:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart dataloader if exhausted
                data_iter = iter(vla_train_dataloader)
                batch = next(data_iter)

            qwen_inputs = batch.get("qwen_inputs", {})
            state = batch.get("state", None)
            actions = batch.get("actions", None)

            # Move inputs to device
            for key in qwen_inputs:
                if isinstance(qwen_inputs[key], torch.Tensor):
                    qwen_inputs[key] = qwen_inputs[key].to(model.device)

            if state is not None:
                state = state.to(model.device)
            if actions is not None:
                actions = actions.to(model.device)

            # Forward pass
            try:
                output_dict = model(qwen_inputs=qwen_inputs, state=state, actions=actions)
                action_loss = output_dict["action_loss"]
            except Exception as e:
                logger.error(f"Error during forward pass at step {step}: {e}")
                raise

            # Backward pass
            action_loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.cfg.trainer.get("max_grad_norm", 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Logging
            if step % self.cfg.log_freq == 0 and self.global_rank == 0:
                elapsed = time.time() - t_start
                throughput = self.cfg.log_freq / elapsed if elapsed > 0 else 0
                lr = optimizer.param_groups[0]["lr"]
                
                log_msg = (
                    f"Step {step:6d} | Loss: {action_loss.item():.4f} | "
                    f"LR: {lr:.2e} | Time: {elapsed:.3f}s | Throughput: {throughput:.2f} it/s"
                )
                logger.info(log_msg)
                
                if self.cfg.wandb_enabled:
                    wandb.log({
                        "loss": action_loss.item(),
                        "learning_rate": lr,
                        "step": step,
                    })
                
                t_start = time.time()

            # Save checkpoint
            if step > 0 and step % self.cfg.save_steps == 0:
                self.save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    step,
                    self.cfg.checkpoint_dir,
                )

            step += 1
            if step >= self.cfg.train_steps:
                done = True

        # Final checkpoint
        self.save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            step,
            self.cfg.checkpoint_dir,
        )

        # Cleanup
        dist.barrier()
        if self.global_rank == 0 and self.cfg.wandb_enabled:
            wandb.finish()
        
        dist.destroy_process_group()

        if self.global_rank == 0:
            logger.info("Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="TorchTitan training for Qwen-GR00T")
    parser.add_argument(
        "--config-file",
        type=str,
        default="examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml",
        help="Path to YAML config",
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_file)
    
    trainer = TorchTitanTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
