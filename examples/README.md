# SDPO-RL Examples

Runnable examples demonstrating SDPO training for different tasks and configurations. All use Qwen2.5-0.5B which fits on consumer GPUs.

## Quick Start

```bash
pip install sdpo-rl transformers datasets

python examples/basic_sdpo.py            # Math (addition)
python examples/sdpo_with_unsloth.py     # Reasoning with Unsloth + QLoRA
python examples/sdpo_rich_feedback.py    # Code generation with test feedback
```

## Examples

### 1. `basic_sdpo.py` -- Minimal SDPO Training

Trains on simple math problems. Demonstrates the core loop: generate completions, score with rewards, select peer demos, distill.

- Binary rewards (1.0 correct, 0.0 wrong)
- Feedback strings ("Wrong. The answer is 7, not 5.")
- EMA teacher updates
- ~5 minutes on RTX 3080

### 2. `sdpo_with_unsloth.py` -- Unsloth + QLoRA

Same idea, but with Unsloth for 2x faster training and 60% less memory. Shows the critical import order:

```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)       # Patch FIRST
from sdpo_rl import SDPOTrainer, SDPOConfig  # Import SECOND
```

- 4-bit quantization, LoRA rank 16
- 8-bit AdamW optimizer
- Thinking tag removal from demonstrations
- ~3 minutes on RTX 3080

### 3. `sdpo_rich_feedback.py` -- Code Generation with Test Feedback

SDPO's killer feature: learning from detailed error messages. The reward function executes test cases and returns specific failure info ("input=[1,2,3]: expected 6, got 0"). SDPO injects these into teacher prompts so the model learns what went wrong.

- Sandboxed code execution
- Specific test failure messages
- Multiple task types (list ops, strings, conditionals, loops)
- ~5 minutes on RTX 3080

## Configuration Patterns

### GRPOConfig (base trainer)

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

### SDPOConfig (self-distillation)

```python
from sdpo_rl import SDPOConfig

sdpo_config = SDPOConfig(
    enabled=True,                              # SDPO replaces GRPO loss
    alpha=0.5,                                 # JSD (symmetric). Paper default.
    distillation_topk=100,                     # Top-100 logits
    teacher_mode="ema",                        # EMA teacher
    teacher_update_rate=0.05,                  # teacher = 0.95*teacher + 0.05*student
    is_clip=2.0,                               # IS ratio clamp
    include_environment_feedback=True,         # Use feedback in teacher prompts
    environment_feedback_only_without_solution=True,  # Feedback only when no peer demo
    remove_thinking_from_demonstration=True,   # Strip <think> from demos
)
```

### Reward Functions

TRL expects `list[float]`. To provide feedback for SDPO, use a callable class with `last_feedback`:

```python
class MyReward:
    def __init__(self):
        self.last_feedback: list[str] = []

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        scores = []
        self.last_feedback = []
        for completion in completions:
            if is_correct(completion):
                scores.append(1.0)
                self.last_feedback.append("")
            else:
                scores.append(0.0)
                self.last_feedback.append(f"Wrong: {why}")
        return scores
```

A plain function returning `list[float]` also works if you don't need feedback.

## GPU Requirements

| Example | Min VRAM | With Unsloth |
|---|---|---|
| basic_sdpo.py | ~6 GB | -- |
| sdpo_with_unsloth.py | -- | ~3.5 GB |
| sdpo_rich_feedback.py | ~6 GB | -- |

All examples work on free GPU platforms (Google Colab T4, Kaggle).

## Troubleshooting

**OOM:** Reduce `num_generations` or `per_device_train_batch_size`, or use Unsloth.

**Import errors with Unsloth:** Make sure `PatchFastRL("GRPO", FastLanguageModel)` is called _before_ `from sdpo_rl import SDPOTrainer`.

**Reward function TypeError:** Reward functions must return `list[float]`, not `list[dict]`. Store feedback on `self.last_feedback` instead.
