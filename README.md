# sdpo-trainer

**Self-Distilled Policy Optimization (SDPO) for Hugging Face TRL**

A faithful reimplementation of SDPO ([arxiv:2601.20802](https://arxiv.org/abs/2601.20802)) from the [lasgroup/SDPO](https://github.com/lasgroup/SDPO) verl fork, ported to the Hugging Face TRL ecosystem as a drop-in `GRPOTrainer` subclass.

[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

## What is SDPO?

SDPO (Self-Distilled Policy Optimization) replaces GRPO's scalar reward loss with **token-level self-distillation** from a teacher model. On each batch:

1. **Generate** multiple completions per prompt
2. **Evaluate** with your reward function (scores + optional feedback)
3. **Select** successful peer rollouts as demonstrations
4. **Reprompt** the teacher with the task + peer demo + error feedback
5. **Distill** the student towards the teacher via top-K KL divergence
6. **Update** the EMA teacher weights

The teacher sees "here's a working solution and the error from your last attempt" while the student learns to match that improved distribution. No third model in memory -- the teacher is the `ref_model` updated via EMA.

## Quick Start

```bash
pip install sdpo-trainer
```

```python
from sdpo_trainer import SDPOTrainer, SDPOConfig
from trl import GRPOConfig

grpo_config = GRPOConfig(
    output_dir="./output",
    num_generations=4,
    max_completion_length=128,
    learning_rate=1e-5,
    bf16=True,
    remove_unused_columns=False,
)

sdpo_config = SDPOConfig()  # sensible defaults from the paper

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

### With Unsloth (2x faster, 60% less memory)

```python
from unsloth import FastLanguageModel, PatchFastRL

# CRITICAL: Patch BEFORE importing SDPOTrainer
PatchFastRL("GRPO", FastLanguageModel)

from sdpo_trainer import SDPOTrainer, SDPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)

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

## Configuration

### SDPOConfig Reference

Every parameter matches the [lasgroup/SDPO](https://github.com/lasgroup/SDPO) reference. Defaults are from the paper's experiment scripts.

```python
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
    teacher_mode="ema",                        # "ema" or "frozen" (trust_region declared but not yet implemented)
    teacher_update_rate=0.05,                  # EMA rate: teacher = (1-rate)*teacher + rate*student

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

### Reward Functions

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

### Training Modes

**SDPO (default):** Self-distillation replaces GRPO loss entirely.
```python
sdpo_config = SDPOConfig(enabled=True)  # the default
```

**GRPO fallback:** Disable SDPO, use standard TRL GRPO.
```python
sdpo_config = SDPOConfig(enabled=False)
```

**Frozen teacher:** Use initial weights as teacher (no EMA updates).
```python
sdpo_config = SDPOConfig(teacher_mode="frozen")
```

### KL Divergence Variants

```python
SDPOConfig(alpha=0.0)   # Forward KL: KL(teacher || student) -- mode-covering
SDPOConfig(alpha=0.5)   # JSD: symmetric Jensen-Shannon (paper default)
SDPOConfig(alpha=1.0)   # Reverse KL: KL(student || teacher) -- mode-seeking
```

The paper uses `alpha=0.5` for generalization experiments and `alpha=1.0` with `distillation_topk=20` for rich-feedback code tasks.

## How It Works

```
                     ┌─────────────────────────────────────────┐
                     │  For each batch:                        │
                     │                                         │
 Prompt ──┬──> Generate 4 completions ──> Reward function     │
           │         │                         │               │
           │    [comp1, comp2, comp3, comp4]   │               │
           │         │                    [1.0, 0.0, 0.0, 0.5] │
           │         │                         │               │
           │    Select best peer (comp1) ──────┘               │
           │         │                                         │
           │    Build teacher prompt:                          │
           │      "Task: {prompt}                              │
           │       Correct solution: {comp1}                   │
           │       Feedback: {error from comp2}                │
           │       Correctly solve the original question."     │
           │         │                                         │
           │    Teacher forward pass (with reprompted input)   │
           │    Student forward pass (with original input)     │
           │         │                                         │
           │    KL(student top-K || teacher top-K) ──> loss    │
           │         │                                         │
           │    EMA update: teacher <- 0.95*teacher + 0.05*student
           └─────────────────────────────────────────────────────┘
```

## Examples

See `examples/` for complete, runnable scripts:

| Example | Task | Key Feature |
|---|---|---|
| `basic_sdpo.py` | Math (addition) | Core SDPO loop with feedback |
| `sdpo_with_unsloth.py` | Reasoning | Unsloth + QLoRA + 4-bit |
| `sdpo_rich_feedback.py` | Code generation | Test execution with error messages |

```bash
python examples/basic_sdpo.py
```

## Benchmark

The `benchmark/` directory contains a full MBPP code generation benchmark (Qwen2.5-0.5B, 4-bit QLoRA, RTX 3080):

- **Correctness:** Max |our_loss - ref_loss| = 1.4e-08 across 200 training steps (verified against verl reference)
- **Performance:** SDPO 1.95% pass@1 vs GRPO 1.17% at step 200

See [benchmark/README.md](benchmark/README.md) for full results and replication instructions.

## Documentation

| Document | What it covers |
|---|---|
| [VERIFICATION.md](VERIFICATION.md) | Line-by-line verification against verl reference |
| [DEVIATIONS.md](DEVIATIONS.md) | Intentional differences from verl (TRL adaptation) |
| [HANDOVER.md](HANDOVER.md) | Architecture decisions, gotchas, implementation guide |
| [UNSLOTH_INTEGRATION.md](UNSLOTH_INTEGRATION.md) | Unsloth compatibility, import order, what works |
| [examples/README.md](examples/README.md) | Example walkthrough and customization guide |
| [benchmark/README.md](benchmark/README.md) | MBPP benchmark methodology and results |

## GPU Requirements

| Model Size | Without Unsloth | With Unsloth (4-bit) |
|---|---|---|
| 0.5B | ~6 GB | ~3.5 GB |
| 7B | ~28 GB | ~10 GB |
| 14B | ~56 GB | ~18 GB |

## Citation

```bibtex
@article{zhang2025sdpo,
  title={Reinforcement Learning via Self-Distillation},
  author={Zhang, Xueying and Guo, Yunhao and Kwok, James T. and Krause, Andreas},
  journal={arXiv preprint arXiv:2601.20802},
  year={2025}
}
```

## License

Apache 2.0

## Acknowledgments

- [lasgroup/SDPO](https://github.com/lasgroup/SDPO) -- Reference implementation
- [Hugging Face TRL](https://github.com/huggingface/trl) -- Base GRPO trainer
- [Unsloth](https://github.com/unslothai/unsloth) -- Training optimizations
