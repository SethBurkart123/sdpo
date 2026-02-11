# Implementation Deviations from Reference

This document details the differences between our SDPO implementation and the reference implementation at [lasgroup/SDPO](https://github.com/lasgroup/SDPO).

## Summary

Our implementation is a **faithful reimplementation** of the SDPO algorithm adapted to work with TRL's GRPOTrainer instead of verl's distributed RL framework. All core algorithms, mathematical formulas, and behaviors match the reference exactly. The differences are purely in the integration layer and API surface.

## 1. Framework Integration

### Reference (verl)
- Built on verl, a custom distributed RL framework
- Uses Ray for distributed training  
- Custom actor-rollout-ref architecture
- Integrated with vLLM for generation

### Our Implementation (TRL)
- Built on TRL's GRPOTrainer (Hugging Face ecosystem)
- Uses Accelerate for distributed training
- Subclasses GRPOTrainer and overrides key methods
- Inherits TRL's vLLM support

**Impact**: Different API but identical core algorithm. Our implementation is more accessible to the Hugging Face community.

## 2. Reward Function Format

### Reference (verl)
- verl supports structured reward outputs with feedback directly
- Rewards and feedback flow through the pipeline naturally

### Our Implementation (TRL)
- TRL expects `list[float]` from reward functions
- We provide a wrapper pattern for feedback extraction:
  ```python
  class RewardWithFeedback:
      def __init__(self):
          self.last_feedback = []
          self.__name__ = "RewardWithFeedback"
      
      def __call__(self, prompts, completions, **kwargs):
          # Compute rewards and store feedback
          self.last_feedback = feedback_list
          return scores  # list[float] for TRL
  ```

**Impact**: Users must wrap reward functions that provide feedback, but this is a simple pattern documented in our examples.

## 3. UID Grouping Mechanism

### Reference (verl)
- Explicit `uid` field in data format
- Rollouts explicitly tagged with prompt UIDs
- `_collect_solutions_by_uid` groups by these UIDs

### Our Implementation (TRL)
- TRL doesn't have explicit UIDs
- We leverage TRL's `RepeatSampler` structure
- Indices `[i*G : (i+1)*G]` guaranteed to share a prompt
- Functionally equivalent grouping without explicit UIDs

**Impact**: No API difference for users. Grouping works identically.

## 4. Configuration Structure

### Reference (verl)
- Single YAML config file combining all settings
- Config loaded via Hydra/OmegaConf
- Nested structure: `actor_rollout_ref.actor.self_distillation.*`

### Our Implementation (TRL)
- Separate `GRPOConfig` (from TRL) and `SDPOConfig` (our addition)
- Python dataclasses instead of YAML
- Flat structure with explicit parameters

**Impact**: More Pythonic API. Same configurability, different organization.

## 5. Distributed Training Strategy

### Reference (verl)
- Ray-based distributed architecture
- Separate actor/rollout/ref workers
- Custom distributed coordination

### Our Implementation (TRL)
- Accelerate-based distributed training
- FSDP/DDP support inherited from TRL
- Standard PyTorch distributed patterns

**Impact**: Uses standard Hugging Face distributed training. May have different multi-GPU behavior but single-GPU behavior is identical.

## 6. Generation Backend

### Reference (verl)
- Native vLLM integration
- Optimized for high-throughput generation
- Custom vLLM worker management

### Our Implementation (TRL)
- TRL's vLLM support (via `use_vllm=True`)
- Also supports standard HF generation
- Inherits TRL's generation config

**Impact**: More flexible generation options. vLLM works but tested primarily with standard generation.

## 7. Logging and Metrics

### Reference (verl)
- Custom metrics logging
- Integration with wandb/tensorboard via verl
- Detailed per-function reward tracking

### Our Implementation (TRL)
- TRL's standard logging infrastructure
- Inherits TRL's wandb/tensorboard integration
- Metrics stored in `self._logs` dict

**Impact**: Standard TRL logging. Slightly different metric names but same information.

## 8. Test Coverage

### Reference (verl)
- Integration tests within verl's test suite
- Tested with verl's full RL pipeline

### Our Implementation (TRL)
- 137+ comprehensive tests (123 unit + 10 e2e + 4 GPU example smoke tests)
- Tests verify:
  - Mathematical correctness (all formulas)
  - Integration with TRL
  - Real model training (Qwen 0.5B on GPU)
  - LoRA EMA adapter operations (init, collect pairs, EMA update, callback)
- Higher test coverage than reference

**Impact**: More thorough testing of core algorithms in isolation.

## 9. Memory Optimization Strategy

### Reference (verl)
- Relies on Ray's memory management
- vLLM sleep/wake for memory efficiency
- Custom chunking strategies

### Our Implementation (TRL)
- Standard PyTorch memory management
- Optional Unsloth integration for optimization
- Top-K logits reduce memory usage intrinsically
- **LoRA EMA mode** (`teacher_mode="lora_ema"`): novel extension not in the reference. Maintains student and teacher as two LoRA adapters on a shared base model, avoiding a full deepcopy. Saves ~3-4 GB for 7B QLoRA models.

**Impact**: Different optimization strategies but similar memory footprint for core SDPO computation. LoRA EMA provides additional savings for PEFT users.

## 10.5. Chat Template Configuration

### Reference (verl)
- N/A (verl handles tokenization differently)

### Our Implementation (TRL)
- `SDPOConfig.apply_chat_template_kwargs` allows passing extra keyword arguments to `tokenizer.apply_chat_template()` during teacher prompt tokenization. This is a TRL-specific addition to support models with non-standard chat template requirements.

**Impact**: Purely additive — does not change core algorithm behavior.

## 10. Model Loading

### Reference (verl)
- Custom model loading utilities
- verl-specific model registry

### Our Implementation (TRL)
- Standard `from_pretrained` via Hugging Face
- Optional Unsloth for faster loading
- Works with any HF model

**Impact**: Broader model compatibility. Works with entire HF model hub.

---

## What Is NOT Different

The following are **identical** to the reference:

✅ All mathematical formulas (loss, KL, tail bucket, IS correction)
✅ Reprompting templates (character-for-character match)
✅ EMA update formula and timing
✅ Configuration defaults (alpha, topk, rates, thresholds)
✅ Demonstration selection logic
✅ Self-distillation mask computation
✅ Zero-coverage handling
✅ Numerical stability measures (clamping, etc.)
✅ Teacher management (2 models, not 3; or 1 model with 2 adapters in lora_ema mode)
✅ Loss replacement behavior (SDPO replaces GRPO entirely)

---

## Migration Guide: verl → TRL

If you're familiar with verl's SDPO and want to use our TRL implementation:

| verl Concept | TRL Equivalent |
|--------------|----------------|
| `SelfDistillationConfig` | `SDPOConfig` |
| `actor.yaml` config | `GRPOConfig` + `SDPOConfig` |
| `loss_mode: sdpo` | `sdpo_config.enabled = True` |
| Ray workers | Accelerate processes |
| vLLM generation | `use_vllm=True` in GRPOConfig |
| UID field in data | RepeatSampler grouping |
| Custom reward returns | `RewardWithFeedback` wrapper class |
| verl trainer | `SDPOTrainer` |

Example conversion:

**verl**:
```python
# Load config from YAML
trainer = VerL_PPOTrainer(config="sdpo.yaml")
trainer.train()
```

**TRL (ours)**:
```python
from sdpo_rl import SDPOTrainer, SDPOConfig
from trl import GRPOConfig

grpo_config = GRPOConfig(
    output_dir="./output",
    num_generations=8,
    # ... other params
)

sdpo_config = SDPOConfig(
    enabled=True,
    alpha=0.5,
    # ... other params
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

---

## Conclusion

Our implementation adapts SDPO to TRL's architecture while preserving 100% of the core algorithm. All deviations are in the integration layer, not the algorithm itself. This makes SDPO accessible to the broader Hugging Face community while maintaining mathematical fidelity to the paper.

**Key Takeaway**: If you understand the SDPO paper, you understand our implementation. The TRL integration is just a different way to invoke the same core algorithm.
