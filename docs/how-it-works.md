# How SDPO Works

A deeper look at the SDPO algorithm and how `SDPOTrainer` implements it.

## The Algorithm

SDPO (Self-Distilled Policy Optimization) replaces GRPO's scalar reward loss with **token-level self-distillation** from a teacher model. The core insight: instead of just telling the model "this answer was good/bad" (scalar reward), show the teacher what a correct answer looks like and let the student learn to match the teacher's improved distribution.

### Training Loop

On each batch:

1. **Generate** multiple completions per prompt (e.g., 4)
2. **Evaluate** with your reward function (scores + optional feedback)
3. **Select** the best successful peer rollout as a demonstration
4. **Reprompt** the teacher with: original task + peer demo + error feedback
5. **Distill** the student towards the teacher via top-K KL divergence
6. **Update** the EMA teacher weights

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

The teacher sees "here's a working solution and the error from your last attempt" while the student learns to match that improved distribution. No third model in memory -- the teacher is the `ref_model` updated via EMA.

## Key Design Decisions

### SDPO Replaces GRPO Loss Entirely

This is the most important thing to understand. When SDPO is enabled, the GRPO clip loss (`-ratio * advantage`) is **completely replaced** by the self-distillation KL loss. There is no lambda blending -- it is purely the distillation loss.

This means `compute_loss` in `SDPOTrainer` does NOT call `super().compute_loss()`. It runs its own forward pass and computes only the distillation loss.

### Two Models, Not Three

The EMA teacher IS the reference model. There is no third model in memory. In TRL terms, we repurpose `self.ref_model` as the EMA teacher (with `beta=0` to disable its normal KL penalty role).

With `teacher_mode="lora_ema"`, it's even better: the student and teacher share the same base model, with two separate LoRA adapter sets. No deepcopy needed.

### Shared Top-K Indices

When computing top-K KL divergence, the student's top-K vocabulary indices are computed first, then the teacher's logits are gathered at those **same** indices. Both distributions are evaluated on the same K vocabulary items. Computing top-K independently would compare different subsets, making the KL meaningless.

### Self-Distillation Mask

Not every sample has a successful peer or feedback. The `self_distillation_mask` zeros out samples without teacher signal. A batch where nothing succeeds produces loss = 0 and no gradient update. This is correct behavior per the paper.

## How SDPOTrainer Works

`SDPOTrainer` subclasses TRL's `GRPOTrainer` and overrides two methods:

### `_generate_and_score_completions(inputs)`

Calls `super()` to generate completions and score them with reward functions, then adds:

1. **Capture raw prompt messages** before they're consumed by the parent
2. **Extract feedback** from reward functions that have a `last_feedback` attribute
3. **Select peer demonstrations** -- for each prompt, find the best successful completion from a different rollout
4. **Build teacher prompts** -- assemble reprompted messages with system messages preserved, peer demos, and error feedback
5. **Tokenize teacher prompts** via `apply_chat_template`
6. **Compute self-distillation mask** -- which samples have teacher signal

### `compute_loss(model, inputs, ...)`

Does NOT call `super()`. Instead:

1. **Student forward pass** -- get logits for the original prompt + completion
2. **Extract student top-K** -- top-K log-probabilities over the vocabulary at each token position
3. **Teacher forward pass** -- get logits for the reprompted prompt + completion (under `torch.no_grad()`)
4. **Gather teacher at student indices** -- evaluate teacher at the student's top-K vocabulary items
5. **Compute KL divergence** -- top-K KL with optional tail bucket, parameterized by alpha
6. **Apply importance sampling** -- correct for off-policy data if applicable
7. **Aggregate** -- token-mean loss, masked by self-distillation mask

## EMA Teacher

The teacher model is an exponential moving average of the student. After each optimizer step:

```
teacher = (1 - rate) * teacher + rate * student
```

With the default `teacher_update_rate=0.05`, the teacher slowly tracks the student. This provides a stable distillation target that improves over time without the instability of using current weights directly (which the paper shows diverges).

### LoRA EMA

For PEFT models, `teacher_mode="lora_ema"` maintains the teacher as a second LoRA adapter on the same base model:

- **"default" adapter** (student): trainable, receives gradient updates
- **"sdpo_teacher" adapter** (teacher): frozen, receives EMA updates

The EMA update only touches adapter weights (~100-200MB), not the base model (~4GB for 7B). Same quality, much less memory.

## Reprompting

The teacher sees an enriched version of the prompt that includes:

1. **System messages** from the original prompt (preserved, not discarded)
2. **The original task** (user message)
3. **A successful peer solution** (if any rollout in the group scored above `success_reward_threshold`)
4. **Error feedback** from the student's failed attempt (if no peer solution exists, or always if configured)

This "context asymmetry" is the core mechanism: the teacher knows what a correct answer looks like, so its distribution is better-calibrated. The student learns to match that distribution without seeing the answer.

## Numerical Details

### Top-K with Tail Bucket

Computing top-K KL over a subset of the vocabulary loses information about the remaining tokens. The tail bucket appends a (K+1)th entry representing the total probability mass of all non-top-K tokens:

```python
tail_log_prob = log(1 - sum(exp(top_k_log_probs)))
```

This is numerically tricky. The implementation uses `logsumexp` + `expm1` with clamping to avoid NaN:

```python
log_s = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
log_s = torch.clamp(log_s, max=-1e-7)  # force sum < 1
tail_log = torch.log(-torch.expm1(log_s))
```

### Importance Sampling

When using off-policy data (the student has been updated since generation), importance sampling corrects the loss:

```
ratio = exp(student_log_prob - old_log_prob)
corrected_loss = clamp(ratio, max=is_clip) * loss
```

The log ratio is clamped to `[-20, 20]` before exponentiation for numerical stability, and the ratio itself is clamped to `max=is_clip` (default 2.0).

## Further Reading

- [SDPO Paper (arxiv:2601.20802)](https://arxiv.org/abs/2601.20802) -- Hubotter et al., 2026
- [Reference Implementation (lasgroup/SDPO)](https://github.com/lasgroup/SDPO) -- verl fork
- [Configuration Reference](configuration.md) -- Full SDPOConfig documentation
- [Verification Checklist](verification.md) -- Line-by-line verification against reference
- [Deviations from Reference](deviations.md) -- What differs and why
