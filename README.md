# sdpo-rl

**Self-Distilled Policy Optimization (SDPO) for Hugging Face TRL**

A faithful reimplementation of SDPO ([arxiv:2601.20802](https://arxiv.org/abs/2601.20802)) from the [lasgroup/SDPO](https://github.com/lasgroup/SDPO) verl fork, ported to the **Hugging Face TRL** ecosystem as a drop-in `GRPOTrainer` subclass and included support for **Unsloth**.

[![PyPI](https://img.shields.io/pypi/v/sdpo-rl)](https://pypi.org/project/sdpo-rl/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

## What is SDPO?

### The Core Idea

- When you give a model a good example in its prompt, it performs better.
- Give it two good examples and it performs even better.
- SDPO figured out how to distill that improvement back into the base model — so it behaves as if it always has those examples, without actually needing them in the prompt.
- This process repeats, compounding the model's capability over successive rounds.

### Where Do High-Quality Examples Come From?

- **Environment feedback:** Feedback from some environment (e.g., a Python interpreter)
- **Human-authored work:** Handwritten worked examples, reasoning traces, or notes
- **Best-of-N sampling:** Prompt the model multiple times for the same task, then keep whichever answer passes your evaluation criteria
- **Distillation from a stronger model:** Stronger model reasoning traces
- **Deep research pipelines:** Answers or relevant context fetched from reputable sources

The higher the quality of your examples, the more powerful your model. Think creatively about where you can get high-quality examples. Multiple examples drive up quality. If adding an example to the context improves the answer, SDPO pushes towards higher quality; if it degrades the answer, SDPO pushes towards lower quality. Be careful about the quality you train with.

### How Do You Measure Example Quality?

- **Pass-rate scoring:** Generate hundreds of outputs per problem, measure what fraction are correct. Higher pass rate = stronger example.
- **Reward models / LLM-as-judge:** A separate model scores each output. A/B comparison scoring tends to work better than absolute ratings.
- **Manual review:** Read the output and assess whether it's higher quality.

### Additional Benefits

SDPO offers advantages such as:

- [Faster training](https://x.com/jonashuebotter/status/2016950268462608665) (denser reward signal)
- [Shorter reasoning traces](https://x.com/jonashuebotter/status/2016953855146107075) from better credit assignment ([related](https://x.com/jonashuebotter/status/2016954946390757476))
- [Reduced forgetting](https://x.com/IdanShenfeld/status/2016818112004305302) when jumping between datasets

### The Algorithm

SDPO replaces GRPO's scalar reward loss with **token-level self-distillation** from a teacher model. On each batch:

1. **Generate** multiple completions per prompt
2. **Evaluate** with your reward function (scores + optional feedback)
3. **Select** successful peer rollouts as demonstrations
4. **Reprompt** the teacher with the task + peer demo + error feedback
5. **Distill** the student towards the teacher via top-K KL divergence
6. **Update** the EMA teacher weights

The teacher sees "here's a working solution and the error from your last attempt" while the student learns to match that improved distribution. No third model in memory -- the teacher is the `ref_model` updated via EMA.

## Quick Start

```bash
pip install sdpo-rl
```

```python
from sdpo_rl import SDPOTrainer, SDPOConfig
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

from sdpo_rl import SDPOTrainer, SDPOConfig

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

Parameters match the [lasgroup/SDPO](https://github.com/lasgroup/SDPO) reference (see [Known Limitations](#known-limitations) for gaps). Defaults are from the paper's experiment scripts. See [docs/configuration.md](docs/configuration.md) for the full reference.

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

**LoRA EMA teacher:** Memory-efficient mode for PEFT/LoRA models. Keeps student and teacher as two LoRA adapters on a shared base model instead of deepcopying the entire model. Saves ~3-4 GB for 7B QLoRA.
```python
from peft import LoraConfig, get_peft_model

model = get_peft_model(base_model, LoraConfig(r=16, ...))
sdpo_config = SDPOConfig(teacher_mode="lora_ema")
```

### KL Divergence Variants

```python
SDPOConfig(alpha=0.0)   # Forward KL: KL(teacher || student) -- mode-covering
SDPOConfig(alpha=0.5)   # JSD: symmetric Jensen-Shannon (paper default)
SDPOConfig(alpha=1.0)   # Reverse KL: KL(student || teacher) -- mode-seeking
```

The paper uses `alpha=0.5` (JSD) across its experiments. The [reference experiment scripts](https://github.com/lasgroup/SDPO) use `alpha=1.0` with `distillation_topk=20` for LiveCodeBench.

## How It Works

See [docs/how-it-works.md](docs/how-it-works.md) for a deeper dive into the algorithm, architecture decisions, and numerical details.

```text
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
| --- | --- | --- |
| `basic_sdpo.py` | Math (addition) | Core SDPO loop with feedback |
| `sdpo_lora_ema.py` | Math (multiplication) | LoRA EMA teacher mode (memory-efficient) |
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
| --- | --- |
| [docs/configuration.md](docs/configuration.md) | Full SDPOConfig reference, training modes, reward functions |
| [docs/how-it-works.md](docs/how-it-works.md) | Algorithm deep-dive, architecture decisions, numerical details |
| [docs/unsloth.md](docs/unsloth.md) | Unsloth compatibility, import order, what works |
| [docs/verification.md](docs/verification.md) | Line-by-line verification against verl reference |
| [docs/deviations.md](docs/deviations.md) | Intentional differences from verl (TRL adaptation) |
| [examples/README.md](examples/README.md) | Example walkthrough and customization guide |
| [benchmark/README.md](benchmark/README.md) | MBPP benchmark methodology and results |

## GPU Requirements

| Model Size | Standard EMA | LoRA EMA | Unsloth (4-bit) |
| --- | --- | --- | --- |
| 0.5B | ~6 GB | ~4 GB | ~3.5 GB |
| 7B | ~28 GB | ~14 GB | ~10 GB |
| 14B | ~56 GB | ~30 GB | ~18 GB |

## Known Limitations

Two features from the paper / reference are not yet implemented:

1. **Hybrid SDPO+GRPO blending (Section 4.5):** The paper defines a combined advantage `A = λ·A_GRPO + (1-λ)·A_SDPO` that interpolates between GRPO and SDPO. Our library supports `enabled=True` (pure SDPO) or `enabled=False` (pure GRPO) but not lambda blending. The paper shows this hybrid helps weaker models (e.g., Qwen3-0.6B).

2. **`trust_region` teacher mode:** Declared in config validation but raises `NotImplementedError`. All paper experiments use EMA, so this is low priority.

## Citation

```bibtex
@article{hubotter2026sdpo,
  title={Reinforcement Learning via Self-Distillation},
  author={H{\"u}botter, Jonas and L{\"u}beck, Frederike and Behric, Lejs and Baumann, Anton and Bagatella, Marco and Marta, Daniel and Hakimi, Ido and Shenfeld, Idan and Kleine Buening, Thomas and Guestrin, Carlos and Krause, Andreas},
  journal={arXiv preprint arXiv:2601.20802},
  year={2026}
}
```

## License

Apache 2.0

## Acknowledgments

- [lasgroup/SDPO](https://github.com/lasgroup/SDPO) -- Reference implementation
- [Hugging Face TRL](https://github.com/huggingface/trl) -- Base GRPO trainer
- [Unsloth](https://github.com/unslothai/unsloth) -- Training optimizations
