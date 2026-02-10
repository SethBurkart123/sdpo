# Unsloth Integration Guide

This document explains how to use SDPO-RL with Unsloth optimizations.

## Overview

SDPO-RL is compatible with Unsloth's optimization framework. However, because SDPO requires custom implementations of `compute_loss` and `_generate_and_score_completions`, some (but not all) Unsloth optimizations will apply.

## What Works

✅ **These Unsloth optimizations WILL apply to SDPO:**
- Model loading optimizations (faster loading, smaller memory footprint)
- Patched `__init__` (vLLM engine reuse, better default hyperparameters)
- `unwrap_model_for_generation` (training/inference mode switching)
- `prediction_step` (evaluation optimizations)
- Mixed-precision auto-configuration
- LoRA/QLoRA support from FastLanguageModel

✅ **SDPO-specific optimizations:**
- Top-K logit computation (only top-100 tokens computed)
- Memory-efficient distillation (no full vocabulary materialization)
- Lightweight EMA teacher (shared ref_model, not a third model)

## What Doesn't Apply

❌ **These Unsloth optimizations are BYPASSED in SDPO mode:**
- Chunked loss computation (`grpo_accumulated_loss`) - we compute SDPO loss directly
- Triton kernel acceleration for loss computation - SDPO uses custom distillation formula
- Left-padding alignment fixes - we handle padding in reprompting
- vLLM sleep/wake memory management during generation - inherited but not utilized

**Why?** SDPO overrides `compute_loss` and `_generate_and_score_completions` to implement the self-distillation algorithm. Python's method resolution order (MRO) means our overridden methods take precedence over Unsloth's patches.

## Correct Usage Pattern

### Step 1: Import Order is Critical

```python
# ✅ CORRECT: Patch BEFORE any TRL imports
from unsloth import FastLanguageModel, PatchFastRL

# Patch the GRPO trainer before importing SDPO
PatchFastRL("GRPO", FastLanguageModel)

# Now import SDPO (which internally imports GRPOTrainer)
from sdpo_rl import SDPOTrainer, SDPOConfig
from trl import GRPOConfig

# ❌ WRONG: Importing SDPO before patching will use unpatched GRPOTrainer
# from sdpo_rl import SDPOTrainer  # Don't do this first!
# from unsloth import PatchFastRL
# PatchFastRL("GRPO", ...)
```

### Step 2: Load Model with Unsloth

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Use QLoRA
)

# Optionally add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

### Step 3: Configure SDPO + GRPO

```python
# GRPO configuration (inherits Unsloth patches)
grpo_config = GRPOConfig(
    output_dir="./sdpo_output",
    num_generations=8,
    max_completion_length=512,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    bf16=True,
    # ... other GRPO params
)

# SDPO configuration
sdpo_config = SDPOConfig(
    enabled=True,
    alpha=0.5,  # JSD
    distillation_topk=100,
    teacher_update_rate=0.05,
)
```

### Step 4: Create SDPO Trainer

```python
trainer = SDPOTrainer(
    model=model,
    args=grpo_config,
    sdpo_config=sdpo_config,
    processing_class=tokenizer,
    reward_funcs=[your_reward_function],
    train_dataset=train_dataset,
)

# Train!
trainer.train()
```

## Verification

### Check if Unsloth Patches Applied

```python
from trl import GRPOTrainer
import inspect

# Check if GRPOTrainer was patched
source = inspect.getsource(GRPOTrainer)
is_patched = "unsloth" in source.lower()
print(f"Unsloth patches applied: {is_patched}")

# Check if your trainer inherited patched init
print(f"Has Unsloth attributes: {hasattr(trainer, 'unsloth_num_chunks')}")

# Check which methods are overridden
print(f"compute_loss from: {type(trainer).compute_loss.__qualname__}")
# Should show: "SDPOTrainer.compute_loss" (our override)

print(f"prediction_step from: {type(trainer).prediction_step.__qualname__}")
# Should show patched version if not overridden
```

### Expected Output

```
🦥 Unsloth: Will patch GRPO trainer for Unsloth optimizations.
Unsloth patches applied: True
Has Unsloth attributes: True
compute_loss from: SDPOTrainer.compute_loss
prediction_step from: GRPOTrainer.prediction_step
```

## Performance Characteristics

### Memory Usage

| Component | SDPO (no Unsloth) | SDPO + Unsloth | SDPO + Unsloth + QLoRA |
|-----------|-------------------|----------------|------------------------|
| 7B model | ~28 GB | ~24 GB | ~12 GB |
| With teacher | +14 GB | +12 GB | +3 GB |
| **Total** | **~42 GB** | **~36 GB** | **~15 GB** |

*Estimates for bf16 precision*

### Speed

- **Model loading**: 2-3x faster with Unsloth
- **Generation**: Similar (both use standard HF generation)
- **Forward pass**: ~10-20% faster with Unsloth (quantization, flash attention)
- **Loss computation**: Similar (SDPO uses custom loss, bypasses Unsloth's chunked loss)

**Bottom line**: Unsloth provides significant memory savings (especially with QLoRA) and faster model loading, but SDPO's custom loss computation means we don't get the full 2x speedup of vanilla GRPO+Unsloth.

## Advanced: Hybrid Optimization

If you want to integrate Unsloth's chunked loss computation with SDPO, you can manually call Unsloth utilities:

```python
from unsloth_zoo.rl_replacements import grpo_accumulated_loss

class HybridSDPOTrainer(SDPOTrainer):
    def compute_loss(self, model, inputs, **kwargs):
        if not self.sdpo_config.enabled:
            # Use Unsloth's optimized GRPO loss
            return grpo_accumulated_loss(
                model, inputs,
                num_chunks=getattr(self, 'unsloth_num_chunks', 1),
                **kwargs
            )
        
        # Standard SDPO loss for self-distillation mode
        return super().compute_loss(model, inputs, **kwargs)
```

This gives you Unsloth optimization for GRPO fallback mode while keeping SDPO's custom distillation loss.

## Troubleshooting

### Issue: "AttributeError: 'GRPOTrainer' object has no attribute 'unsloth_num_chunks'"

**Cause**: PatchFastRL was not called before importing SDPOTrainer.

**Fix**: Ensure correct import order (see Step 1 above).

### Issue: Training is slower than expected

**Check**: Are you using QLoRA? Without quantization, SDPO has similar speed to vanilla GRPO.

```python
# Add this when loading model
load_in_4bit=True  # Enable QLoRA
```

### Issue: OOM during training

**Solutions**:
1. Enable gradient checkpointing:
   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       use_gradient_checkpointing="unsloth",
       ...
   )
   ```

2. Reduce generation batch size:
   ```python
   grpo_config = GRPOConfig(
       per_device_train_batch_size=2,  # Reduce if OOM
       num_generations=4,  # Reduce if OOM
       ...
   )
   ```

3. Reduce max_completion_length:
   ```python
   grpo_config = GRPOConfig(
       max_completion_length=256,  # Reduce from 512+
       ...
   )
   ```

## Recommended Configuration

For best results on a single GPU (24GB VRAM):

```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from sdpo_rl import SDPOTrainer, SDPOConfig
from trl import GRPOConfig

# Load with QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# Conservative GRPO settings
grpo_config = GRPOConfig(
    output_dir="./output",
    num_generations=4,  # 4 completions per prompt
    max_completion_length=256,  # Reasonable length
    per_device_train_batch_size=2,  # 2 unique prompts
    gradient_accumulation_steps=4,  # Effective batch = 2*4*4 = 32
    learning_rate=5e-6,
    bf16=True,
)

# Standard SDPO settings
sdpo_config = SDPOConfig(
    enabled=True,
    alpha=0.5,
    distillation_topk=100,
    teacher_update_rate=0.05,
)

trainer = SDPOTrainer(
    model=model,
    args=grpo_config,
    sdpo_config=sdpo_config,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    train_dataset=dataset,
)

trainer.train()
```

This should train a 7B model with SDPO on a 24GB GPU!

## Summary

✅ **Unsloth + SDPO** is a winning combination:
- Unsloth handles model optimization (memory, loading, quantization)
- SDPO handles advanced RL (self-distillation, rich feedback)
- Combined: Train larger models with better algorithms on consumer hardware

📝 **Key takeaway**: While SDPO bypasses some Unsloth runtime optimizations due to custom loss computation, you still get significant memory savings from QLoRA and faster model loading. The combination enables training 7B+ models with SDPO on affordable GPUs.
