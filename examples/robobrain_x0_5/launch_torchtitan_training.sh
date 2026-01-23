#!/bin/bash
#
# TorchTitan Training Launch Script for Qwen-GR00T
# Supports tensor parallelism (TP), data parallelism (DP), and pipeline parallelism (PP)
#
# Usage:
#   ./launch_torchtitan_training.sh [options]
#
# Options:
#   --num-gpus NUM         Number of GPUs (default: 4)
#   --dp-degree NUM        Data parallel degree (default: 2)
#   --tp-degree NUM        Tensor parallel degree (default: 2)
#   --pp-degree NUM        Pipeline parallel degree (default: 1)
#   --batch-size NUM       Batch size per GPU (default: 2)
#   --config FILE          Config file path
#   --data-path PATH       Data path
#   --ckpt-dir PATH        Checkpoint directory
#   --disable-wandb        Disable Weights & Biases logging
#   --help                 Show this help message
#

set -e

# Default values
NUM_GPUS=4
DP_DEGREE=2
TP_DEGREE=2
PP_DEGREE=1
BATCH_SIZE=2
CONFIG_FILE="examples/robobrain_x0_5/conf/train/libero_qwengroot.yaml"
DATA_PATH=""
CKPT_DIR="./outputs/libero_qwengroot/checkpoints/ckpt_out"
DISABLE_WANDB=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << 'EOF'
TorchTitan Training Launch Script for Qwen-GR00T

Usage: ./launch_torchtitan_training.sh [options]

Options:
  --num-gpus NUM         Number of GPUs to use (default: 4)
  --dp-degree NUM        Data parallel degree (default: 2)
  --tp-degree NUM        Tensor parallel degree (default: 2)
  --pp-degree NUM        Pipeline parallel degree (default: 1)
  --batch-size NUM       Batch size per GPU (default: 2)
  --config FILE          Path to config YAML file
  --data-path PATH       Path to training data (WDS format)
  --ckpt-dir PATH        Path to save checkpoints
  --disable-wandb        Disable Weights & Biases logging
  --help                 Show this help message

Examples:
  # Basic 4-GPU training with 2DP x 2TP
  ./launch_torchtitan_training.sh --num-gpus 4

  # 8-GPU training with 4DP x 2TP
  ./launch_torchtitan_training.sh --num-gpus 8 --dp-degree 4 --tp-degree 2

  # Custom data path and checkpoint directory
  ./launch_torchtitan_training.sh --data-path /path/to/data --ckpt-dir /path/to/ckpt

  # Disable W&B logging
  ./launch_torchtitan_training.sh --disable-wandb
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dp-degree)
            DP_DEGREE="$2"
            shift 2
            ;;
        --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        --pp-degree)
            PP_DEGREE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --ckpt-dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --disable-wandb)
            DISABLE_WANDB=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate configuration
print_header "Configuration Validation"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi
print_info "Config file: $CONFIG_FILE"

# Check if we're in the right directory
if [[ ! -d "examples/robobrain_x0_5" ]]; then
    print_error "Please run this script from the FlagScale root directory"
    exit 1
fi

# Validate parallel configuration
TOTAL_REQUIRED=$((DP_DEGREE * TP_DEGREE * PP_DEGREE))
if [[ $TOTAL_REQUIRED -ne $NUM_GPUS ]]; then
    print_error "Invalid configuration: DP($DP_DEGREE) × TP($TP_DEGREE) × PP($PP_DEGREE) = $TOTAL_REQUIRED ≠ NUM_GPUS($NUM_GPUS)"
    print_info "Please adjust parallel degrees so that DP × TP × PP = $NUM_GPUS"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

# Print configuration
print_header "Training Configuration"
print_info "Number of GPUs: $NUM_GPUS"
print_info "Data Parallel (DP): $DP_DEGREE"
print_info "Tensor Parallel (TP): $TP_DEGREE"
print_info "Pipeline Parallel (PP): $PP_DEGREE"
print_info "Total: $DP_DEGREE × $TP_DEGREE × $PP_DEGREE = $TOTAL_REQUIRED"
print_info "Batch size per GPU: $BATCH_SIZE"
print_info "Config file: $CONFIG_FILE"
print_info "Checkpoint dir: $CKPT_DIR"
print_info "Weights & Biases: $([ "$DISABLE_WANDB" = true ] && echo "disabled" || echo "enabled")"

if [[ -n "$DATA_PATH" ]]; then
    print_info "Data path: $DATA_PATH"
fi

# Setup environment variables
print_header "Setting Up Environment"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
if [[ $NUM_GPUS -gt 4 ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
fi
print_info "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# TorchTitan specific environment variables
export NCCL_DEBUG="WARN"
export TORCH_NCCL_BLOCKING_WAIT="1"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export TOKENIZERS_PARALLELISM="false"
export PYTHONHASHSEED="42"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

print_info "Environment variables set"

# Create temporary config file with overrides
TEMP_CONFIG="/tmp/torchtitan_train_$$.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Add parallel configuration to temp config
cat >> "$TEMP_CONFIG" << EOF

# TorchTitan overrides
_torchtitan_parallel_overrides:
  parallelism:
    data_parallel_degree: $DP_DEGREE
    tensor_parallel_degree: $TP_DEGREE
    pipeline_parallel_degree: $PP_DEGREE

_batch_size_override: $BATCH_SIZE
_checkpoint_dir_override: $CKPT_DIR
_wandb_enabled_override: $([ "$DISABLE_WANDB" = true ] && echo "false" || echo "true")
EOF

if [[ -n "$DATA_PATH" ]]; then
    cat >> "$TEMP_CONFIG" << EOF
_data_path_override: $DATA_PATH
EOF
fi

# Launch training
print_header "Launching Training"
print_info "Starting training with $NUM_GPUS GPUs..."
print_info "Press Ctrl+C to stop training"
echo ""

# Run training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    flagscale/train/torchtitan/train_robobrain_x0.5_qwengroot.py \
    --config-file "$TEMP_CONFIG"

EXIT_CODE=$?

# Cleanup
rm -f "$TEMP_CONFIG"

# Print results
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    print_header "Training Completed Successfully!"
    print_info "Checkpoints saved to: $CKPT_DIR"
else
    print_error "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
