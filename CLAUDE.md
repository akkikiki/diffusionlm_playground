# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Environment

```bash
pip install uv
uv venv --python=python3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Common Commands

### Training
```bash
# Train diffusion model with DeepSpeed ZeRO-2
accelerate launch --config_file accelerate_configs/single_node_zero2.yaml \
    src/train.py --output-dir ./outputs --parallel_mode data_parallel \
    --model meta-llama/Llama-3.2-1B-Instruct --max-train-steps 1000 --mask_token 128255

# Train with sequence parallelism (Ulysses attention)
python src/train.py --output-dir ./outputs --parallel_mode ulysses_attn \
    --model meta-llama/Llama-3.2-3B-Instruct --dataset DKYoon/SlimPajama-6B
```

### Inference
```bash
# Run inference with trained model
python src/infer.py --prompt "Write a story that starts with 'Once upon a time'" \
    --model_name ./outputs

# Run inference with base model
python src/infer.py --prompt "What is 1 + 1?" \
    --model_name meta-llama/Llama-3.1-8B-Instruct --verbose True

# Unconditional generation
python src/infer.py --unconditional --model_name ./outputs
```

### Alternative Inference Script
```bash
# Use infer_diffllama.py for LLaMA-specific inference
python src/infer_diffllama.py --model_name diffusionfamily/diffullama \
    --flash_attn flash_attention_2 --diffusion_steps 64
```

## Architecture Overview

### Core Model Architecture
- **DiscreteDiffusionModel**: Extends LlamaForCausalLM to implement discrete diffusion for text generation
- **Location**: `src/model.py:16`
- Uses iterative denoising process instead of autoregressive generation
- Supports both conditional (with prefix) and unconditional generation

### Key Components

#### Diffusion Process (`src/model.py:39`)
- `generate_samples()`: Main sampling function implementing discrete diffusion
- Uses mask token replacement and progressive denoising over multiple steps
- Supports temperature and top-p sampling controls

#### Sequence Parallelism (`src/easy_context/`)
- Modular system supporting multiple attention parallelization strategies:
  - `zigzag_ring_attn`: Ring attention with zigzag pattern
  - `dist_flash_attn`: Distributed flash attention
  - `ulysses_attn`: Ulysses attention parallelism
  - `data_parallel`: Standard data parallelism
- Monkey patching system for runtime attention replacement

#### Dataset Handling (`src/packed_dataset.py`)
- **PackedDataset**: Handles both file-based and HuggingFace datasets
- **CombinedDataset**: Combines multiple datasets with configurable ratios
- Supports efficient packing and chunking for long sequences

#### Training Pipeline (`src/train.py`)
- Distributed training with Accelerate + DeepSpeed
- Configurable datasets via `train_data_config`
- Support for gradient checkpointing and mixed precision

### Configuration Files

#### Accelerate Configs (`accelerate_configs/`)
- `single_node_zero2.yaml`: DeepSpeed ZeRO-2 for single node
- `single_node_zero3.yaml`: DeepSpeed ZeRO-3 for single node  
- `no_deepspeed.yaml`: Standard distributed training without DeepSpeed

#### Model Configurations
- Mask token handling: Uses `<|reserved_special_token_247|>` as default mask token
- Chat template support for conditional generation
- Configurable generation length and diffusion steps

### Training Data Configuration
Default training uses UltraChat dataset:
```python
train_data_config = [
    ("HuggingFaceH4/ultrachat_200k", 1.0),
]
```

### Key Parameters
- `--parallel_mode`: Choose parallelization strategy (data_parallel, ulysses_attn, zigzag_ring_attn, dist_flash_attn)
- `--diffusion_steps`: Number of denoising steps (default: 64)
- `--mask_token`: Token ID to use for masking (default: 128255)
- `--logits_temp`: Temperature for logits sampling (default: 0.9-0.95)
- `--topp_temp`: Top-p threshold for nucleus sampling (default: 0.9)

## Important Notes
- Always use absolute paths when referencing model checkpoints
- The diffusion model requires specific mask token setup for proper functioning
- Sequence parallelism requires careful attention pattern configuration
- Training outputs are saved to `./outputs/` by default