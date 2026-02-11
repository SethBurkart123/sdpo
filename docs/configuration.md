# Configuration Reference

Complete reference for `SDPOConfig` and related configuration patterns.

## SDPOConfig

All parameters match the [lasgroup/SDPO](https://github.com/lasgroup/SDPO) reference. Defaults are from the paper's experiment scripts.

```python
from sdpo_rl import SDPOConfig

SDPOConfig(
    # --- Loss mode ---
    enabled=True,                              # True: SDPO replaces GRPO loss. False: vanilla GRPO.

    # --- KL divergence ---
    alpha=0.5,                                 # 0.0=forward KL, 0.5=JSD (paper default), 1.0=reverse KL
    full_logit_distillation=True,              # Use top-K logits (True) or token-level KL (False)
    distillation_topk=100,                     # K for top-K approximation
    distillation_add_tail=True,                # Append tail bucket for residual probability mass

    # --- Importance sampling ---
    is_clip=2.0,                               # Clamp IS ratio. None disables correction.

    # --- Teacher ---
    teacher_mode="ema",                        # "ema", "frozen", or "lora_ema" (trust_region not yet implemented)
    teacher_update_rate=0.05,                  # EMA rate: teacher = (1-rate)*teacher + rate*student

    # --- Chat template ---
    apply_chat_template_kwargs={},             # Extra kwargs for tokenizer.apply_chat_template()

    # --- Demonstration selection ---
    success_reward_threshold=1.0,              # Min reward for a rollout to be a "successful" demo
    dont_reprompt_on_self_success=True,        # Exclude own response as demonstration
    remove_thinking_from_demonstration=True,   # Strip <think>...</think> from demos

    # --- Feedback ---
    include_environment_feedback=True,         # Include env feedback (test errors, etc.) in teacher prompt
    environment_feedback_only_without_solution=True,  # Only include feedback when no peer demo exists

    # --- Reprompting ---
    max_reprompt_length=10240,                 # Max tokens for teacher prompt
    reprompt_truncation="right",               # "left", "right", or "error"

    # --- Templates (customizable) ---
    reprompt_template="{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n",
    solution_template="\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
    feedback_template="\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
)
```

## Training Modes

### SDPO (default)

Self-distillation replaces GRPO loss entirely. The teacher model is an EMA copy of the student that sees enriched prompts (peer demonstrations + error feedback).

```python
sdpo_config = SDPOConfig(enabled=True)  # the default
```

### GRPO Fallback

Disable SDPO and use standard TRL GRPO. Useful for comparison experiments.

```python
sdpo_config = SDPOConfig(enabled=False)
```

### Frozen Teacher

Use the initial model weights as teacher (no EMA updates). The teacher never changes during training.

```python
sdpo_config = SDPOConfig(teacher_mode="frozen")
```

### LoRA EMA Teacher

Memory-efficient mode for PEFT/LoRA models. Instead of `deepcopy`-ing the entire model for the teacher, the trainer creates a second LoRA adapter ("sdpo_teacher") on the same base model. EMA updates only touch adapter weights.

Saves ~3-4 GB for 7B QLoRA models by eliminating base weight duplication.

```python
from peft import LoraConfig, get_peft_model

model = get_peft_model(base_model, LoraConfig(r=16, ...))
sdpo_config = SDPOConfig(teacher_mode="lora_ema")
```

**Requirements:** The model must be a PEFT model with LoRA adapters. Works with both standard PEFT and Unsloth-loaded models.

## KL Divergence Variants

The `alpha` parameter controls the KL divergence variant used in distillation:

```python
SDPOConfig(alpha=0.0)   # Forward KL: KL(teacher || student) -- mode-covering
SDPOConfig(alpha=0.5)   # JSD: symmetric Jensen-Shannon (paper default)
SDPOConfig(alpha=1.0)   # Reverse KL: KL(student || teacher) -- mode-seeking
```

The paper uses `alpha=0.5` (JSD) across its experiments. The [reference experiment scripts](https://github.com/lasgroup/SDPO) use `alpha=1.0` with `distillation_topk=20` for LiveCodeBench.

## Reward Functions

TRL requires reward functions to return `list[float]`. To provide feedback for SDPO's teacher prompts, use a callable class with a `last_feedback` attribute:

```python
class MyReward:
    """Reward function that provides scores AND feedback for SDPO."""

    def __init__(self):
        self.last_feedback: list[str] = []

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        scores = []
        self.last_feedback = []

        for prompt, completion in zip(prompts, completions):
            if is_correct(completion):
                scores.append(1.0)
                self.last_feedback.append("")
            else:
                scores.append(0.0)
                self.last_feedback.append(f"Wrong: expected {answer}, got {completion}")

        return scores  # TRL requires list[float]

reward_fn = MyReward()
trainer = SDPOTrainer(..., reward_funcs=[reward_fn])
```

The trainer checks `hasattr(rf, "last_feedback")` on each reward function after scoring. Feedback strings are injected into teacher prompts for samples that lack a successful peer demonstration.

If you don't need feedback (simpler tasks), a plain function returning `list[float]` works fine -- SDPO still uses peer demonstrations from successful rollouts.

## GRPOConfig (Base Trainer)

`SDPOTrainer` extends TRL's `GRPOTrainer`, so all `GRPOConfig` parameters apply:

```python
from trl import GRPOConfig

grpo_config = GRPOConfig(
    output_dir="./output",
    num_generations=4,              # Completions per prompt
    max_completion_length=128,      # Max tokens per completion
    temperature=0.7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    bf16=True,
    remove_unused_columns=False,    # Required for SDPO
)
```

See the [TRL GRPOConfig documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig) for all available parameters.

## GPU Requirements

| Model Size | Standard EMA | LoRA EMA | Unsloth (4-bit) |
|---|---|---|---|
| 0.5B | ~6 GB | ~4 GB | ~3.5 GB |
| 7B | ~28 GB | ~14 GB | ~10 GB |
| 14B | ~56 GB | ~30 GB | ~18 GB |

## Chat Template Configuration

For models that use thinking tokens (Qwen3, QwQ, etc.), pass extra arguments to `apply_chat_template` via the config:

```python
sdpo_config = SDPOConfig(
    apply_chat_template_kwargs={"enable_thinking": True},
)
```

These kwargs are forwarded to `tokenizer.apply_chat_template()` during teacher prompt tokenization. The trainer always passes `continue_final_message=False` in addition to any kwargs you specify.
