# SDPO-Trainer Examples

This directory contains practical examples demonstrating how to use SDPO (Self-Distilled Policy Optimization) for different tasks and configurations.

## Quick Start

All examples use small models (Qwen-0.5B) that can run on consumer GPUs (10GB VRAM).

```bash
# Install dependencies
pip install sdpo-trainer transformers datasets

# Run basic example
python examples/basic_sdpo.py

# Run Unsloth example (requires unsloth)
pip install unsloth
python examples/sdpo_with_unsloth.py

# Run rich feedback example
python examples/sdpo_rich_feedback.py
```

## Examples Overview

### 1. `basic_sdpo.py` - Minimal SDPO Training

**What it does:**
- Trains a model to solve simple math problems (addition)
- Uses basic reward function (correct/incorrect)
- Demonstrates core SDPO concepts

**Key features:**
- ✓ Simple dataset (50 math problems)
- ✓ Binary rewards (1.0 for correct, 0.0 for wrong)
- ✓ Feedback strings for teacher reprompting
- ✓ EMA teacher updates

**Best for:**
- Learning SDPO basics
- Understanding the training loop
- Quick testing on new tasks

**Runtime:** ~5 minutes on RTX 3080

---

### 2. `sdpo_with_unsloth.py` - Unsloth + SDPO Integration

**What it does:**
- Same reasoning task as basic example
- Uses Unsloth for 2x faster training, 60% less memory
- Demonstrates proper Unsloth integration

**Key features:**
- ✓ 4-bit quantization (QLoRA)
- ✓ Fast generation (Unsloth optimizations)
- ✓ LoRA fine-tuning (16-rank adapters)
- ✓ 8-bit optimizer
- ✓ Thinking tag support (`<think>...</think>`)

**CRITICAL:** Import order matters!
```python
# CORRECT:
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # Patch FIRST
from sdpo_trainer import SDPOTrainer     # Import SECOND

# WRONG:
from sdpo_trainer import SDPOTrainer     # Too early!
from unsloth import PatchFastRL
```

**Best for:**
- Training larger models (Qwen-7B on 10GB GPU)
- Limited VRAM scenarios
- Production training pipelines

**Runtime:** ~3 minutes on RTX 3080 (faster than basic!)

---

### 3. `sdpo_rich_feedback.py` - Code Generation with Test Feedback

**What it does:**
- Trains a model to write Python functions
- Executes test cases and provides detailed error messages
- Demonstrates SDPO's killer feature: learning from rich feedback

**Key features:**
- ✓ Code execution with test cases
- ✓ Specific error messages ("Test failed: expected X, got Y")
- ✓ Teacher learns from both successful peers AND error feedback
- ✓ Multiple task types (list ops, string ops, conditionals, loops)

**Example feedback flow:**
```python
# Failed attempt 1: Wrong logic
"Test failed: input=[1,2,3], expected=6, got=0"

# Teacher prompt for next attempt:
"""
Original task: Write sum_list(lst)
Peer solution: def sum_list(lst): return sum(lst)  # Success!
Feedback: Test failed: input=[1,2,3], expected=6, got=0
Now try again...
"""

# Model learns from both the successful peer AND the error message
```

**Best for:**
- Code generation tasks
- Any task with executable feedback
- Understanding SDPO's true power

**Runtime:** ~5 minutes on RTX 3080

---

## Example Comparison

| Feature | basic_sdpo.py | sdpo_with_unsloth.py | sdpo_rich_feedback.py |
|---------|---------------|----------------------|----------------------|
| **Task** | Math (addition) | Reasoning (even/odd) | Code generation |
| **Feedback** | Simple strings | Simple strings | Rich error messages |
| **Optimization** | None | Unsloth + QLoRA | None |
| **Memory Usage** | ~6GB VRAM | ~3.5GB VRAM | ~6GB VRAM |
| **Runtime** | ~5 min | ~3 min | ~5 min |
| **Best for** | Learning SDPO | Production use | Understanding power |

---

## Common Configuration Patterns

### GRPO Configuration (Base Trainer)

```python
from trl import GRPOConfig

grpo_config = GRPOConfig(
    # Output
    output_dir="./output",
    
    # Generation
    num_generations=4,              # 4 completions per prompt
    max_completion_length=128,      # Max tokens per completion
    temperature=0.7,                # Sampling temperature
    
    # Training
    num_train_epochs=1,
    per_device_train_batch_size=4,  # 4 prompts per batch
    gradient_accumulation_steps=2,  # Effective batch = 4 * 2 = 8
    learning_rate=1e-5,
    
    # Optimization
    bf16=True,                      # Use bfloat16
    remove_unused_columns=False,    # Required for SDPO
)
```

### SDPO Configuration (Self-Distillation)

```python
from sdpo_trainer import SDPOConfig

sdpo_config = SDPOConfig(
    # Core SDPO
    enabled=True,                   # Enable SDPO (replaces GRPO loss)
    alpha=0.5,                      # JSD balance (0.5 = symmetric)
    distillation_topk=100,          # Top-100 logits for efficiency
    
    # Teacher
    teacher_mode="ema",             # EMA teacher (no third model)
    ema_tau=0.05,                   # EMA rate: θ_t = 0.95*θ_t + 0.05*θ_s
    ema_update_every=1,             # Update every batch
    
    # Importance Sampling
    is_coef=0.0,                    # Disable IS for on-policy
    is_clip=2.0,                    # Clip IS weights (when enabled)
    
    # Reprompting
    reprompt_teacher=True,          # Use peer demos as teacher input
    thinking_tag="<think>",         # Strip thinking tags from demos
)
```

### Reward Function Patterns

**Pattern 1: Simple Scoring (basic_sdpo.py)**
```python
def reward_fn(prompts, completions, **kwargs):
    results = []
    for prompt, completion in zip(prompts, completions):
        if is_correct(completion):
            score = 1.0
            feedback = "Correct!"
        else:
            score = 0.0
            feedback = "Wrong."
        results.append({"score": score, "feedback": feedback})
    return results
```

**Pattern 2: Rich Feedback (sdpo_rich_feedback.py)**
```python
def reward_fn(prompts, completions, **kwargs):
    results = []
    for prompt, completion in zip(prompts, completions):
        try:
            # Execute test cases
            output = execute(completion)
            if output == expected:
                results.append({"score": 1.0, "feedback": "All tests passed!"})
            else:
                # Detailed error message
                feedback = f"Test failed: expected {expected}, got {output}"
                results.append({"score": 0.0, "feedback": feedback})
        except Exception as e:
            # Exception details
            feedback = f"Error: {type(e).__name__}: {str(e)}"
            results.append({"score": 0.0, "feedback": feedback})
    return results
```

---

## GPU Requirements

| Example | Min VRAM | Recommended VRAM | CPU Mode |
|---------|----------|------------------|----------|
| basic_sdpo.py | 6GB | 10GB | Slow |
| sdpo_with_unsloth.py | 3.5GB | 6GB | Not supported |
| sdpo_rich_feedback.py | 6GB | 10GB | Slow |

**Notes:**
- All examples use Qwen-0.5B (494M parameters)
- For larger models (Qwen-7B), use Unsloth with 4-bit quantization
- CPU training is technically possible but ~50x slower

---

## Free GPU Options

All examples work on free GPU platforms:

### Google Colab (Free Tier)
- **GPU:** T4 (16GB VRAM)
- **Runtime:** All examples work
- **Setup:** Just `!pip install sdpo-trainer`

### Kaggle Notebooks
- **GPU:** T4 (16GB VRAM) or P100 (16GB VRAM)
- **Runtime:** All examples work
- **Setup:** Same as Colab

### Lambda Labs (Free Credits)
- **GPU:** Various (often A10 or better)
- **Runtime:** Very fast
- **Setup:** Same as local

---

## Extending the Examples

### Add Your Own Task

1. **Create a dataset:**
```python
dataset = Dataset.from_list([
    {"uid": "task_1", "prompt": "Your task here", "expected": "answer"},
    # ...
])
```

2. **Write a reward function:**
```python
def my_reward_fn(prompts, completions, **kwargs):
    # Your evaluation logic
    return [{"score": ..., "feedback": ...} for ...]
```

3. **Configure and train:**
```python
trainer = SDPOTrainer(
    model=model,
    args=grpo_config,
    sdpo_config=sdpo_config,
    processing_class=tokenizer,
    reward_funcs=[my_reward_fn],
    train_dataset=dataset,
)
trainer.train()
```

### Common Customizations

**Use larger models:**
```python
# With Unsloth (recommended)
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,  # 7B fits on 10GB GPU
)
```

**Adjust generation diversity:**
```python
grpo_config = GRPOConfig(
    temperature=0.9,        # Higher = more diverse
    top_p=0.95,            # Nucleus sampling
    num_generations=8,      # More attempts = more diversity
)
```

**Customize teacher prompts:**
```python
sdpo_config = SDPOConfig(
    teacher_prompt_template=(
        "Task: {prompt}\n"
        "Previous attempt feedback: {feedback}\n"
        "Example solution: {demonstration}\n"
        "Your improved solution:"
    ),
)
```

---

## Troubleshooting

### OOM (Out of Memory)

**Solution 1:** Use Unsloth
```bash
pip install unsloth
python examples/sdpo_with_unsloth.py
```

**Solution 2:** Reduce batch size
```python
grpo_config = GRPOConfig(
    per_device_train_batch_size=2,  # Smaller batch
    gradient_accumulation_steps=4,  # Keep effective batch size
)
```

**Solution 3:** Reduce generations
```python
grpo_config = GRPOConfig(
    num_generations=2,  # Fewer completions per prompt
)
```

### Slow Training

**Solution:** Use Unsloth (2x speedup)
```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'sdpo_trainer'`

**Solution:** Install the package
```bash
pip install -e .  # If in repo root
# OR
pip install sdpo-trainer  # If published to PyPI
```

---

## Next Steps

After running the examples:

1. **Read the docs:**
   - `../VERIFICATION.md` - Algorithm verification
   - `../UNSLOTH_INTEGRATION.md` - Unsloth details
   - `../HANDOVER.md` - Implementation guide

2. **Try your own task:**
   - Start with `basic_sdpo.py` as a template
   - Replace dataset and reward function
   - Adjust hyperparameters

3. **Scale up:**
   - Use Unsloth for larger models
   - Try multi-GPU training (FSDP)
   - Implement custom evaluation metrics

4. **Share your results:**
   - Open an issue with your use case
   - Contribute new examples
   - Report bugs or improvements

---

## Questions?

- **Implementation details:** See `../HANDOVER.md`
- **Algorithm verification:** See `../VERIFICATION.md`
- **Unsloth integration:** See `../UNSLOTH_INTEGRATION.md`
- **Bug reports:** Open a GitHub issue
- **General questions:** Check the main README

Happy training! 🚀
