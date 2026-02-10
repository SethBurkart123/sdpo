# SDPO-RL: Agent Handover Document

## What This Project Is

A standalone pip-installable library (`sdpo-rl`) that brings **Self-Distilled
Policy Optimization** (SDPO) to Hugging Face TRL's `GRPOTrainer`. It is a faithful
reimplementation of [arxiv:2601.20802](https://arxiv.org/abs/2601.20802), matching the
reference code at [lasgroup/SDPO](https://github.com/lasgroup/SDPO) (a verl fork).

The goal: let anyone do SDPO training with 3 lines of config on top of their existing
TRL/Unsloth GRPO setup.

---

## What Has Been Done

### Research (complete)

We did deep-dive research into:

1. **The verl SDPO implementation** — read every relevant source file from the
   lasgroup/SDPO repository. We know exactly how every function works.
2. **TRL GRPOTrainer internals** — mapped every method signature, the buffering
   system (`_prepare_inputs`, `_buffered_inputs`, `steps_per_generation`), the
   loss computation (`compute_loss` → `_compute_loss`), and the counter semantics
   (`_step` vs `global_step` vs `num_iterations`).
3. **Unsloth's patching mechanism** — how `PatchFastRL` replaces `GRPOTrainer` in
   the `trl` module namespace, and what that means for subclass method resolution.
4. **9 critical gotchas** that would have broken the implementation (see below).

### Implementation (complete)

All code uses **test-driven development**. **115 tests across 7 test files, all passing.**

| Module | What It Does | Tests |
|--------|-------------|-------|
| `distillation.py` | Top-K KL divergence, tail bucket, JSD, IS clipping, loss aggregation | 27 |
| `reprompting.py` | Teacher prompt construction, demo selection, thinking tag removal, SD mask | 19 |
| `teacher.py` | EMA update formula, callback with TRL timing | 7 |
| `config.py` | `SDPOConfig` dataclass with paper experiment defaults | 16 |
| `trainer.py` | `SDPOTrainer(GRPOTrainer)` — full integration layer | 10 |
| `utils.py` | Unsloth import order detection | 0 (runtime utility) |
| *reference match* | Verify outputs match verl reference code | 34 |
| *unsloth integration* | End-to-end with Unsloth + 4-bit LoRA | 2 |

Run tests: `uv run pytest`

### Deep verification against reference

The reference repository (lasgroup/SDPO) was cloned and every core function was
compared line-by-line against our implementation. 6 bugs were found and fixed:

1. **Rewards vs advantages** for demo selection — now uses raw rewards
2. **EMA timing** — moved from `compute_loss` to `on_step_end` callback
3. **Feedback capture** — now reads from reward function's `last_feedback` attribute
4. **Teacher tokenization** — now uses `apply_chat_template` with chat formatting
5. **Self-distillation mask** — accounts for `feedback_only_without_solution` flag
6. **EMA with quantized models** — skips non-floating-point params (Unsloth/4-bit)

---

## Architecture Decisions

### 1. SDPO Replaces GRPO Loss Entirely

**This is the single most important thing to understand.**

In the verl reference, when `loss_mode="sdpo"`, the GRPO clip loss (`-ratio * advantage`)
is **completely replaced** by the self-distillation KL loss. The GRPO advantages are
computed (for compatibility with the verl pipeline) but **never used in the loss**.

There is **no lambda blending**. It's not `lambda * grpo + (1-lambda) * sdpo`. It's
purely the distillation loss.

This means `compute_loss` in SDPOTrainer must NOT call `super().compute_loss()`. It
must do its own forward pass and compute only the distillation loss. Unsloth's
optimized `compute_loss` does NOT apply in SDPO mode (but Unsloth's generation
optimizations still apply via `super()._generate_and_score_completions()`).

### 2. Ref Model = EMA Teacher (Two Models, Not Three)

In verl, the EMA teacher IS the reference model worker. There is no third model.
In TRL, we set `beta=0` (disabling the KL penalty so `ref_model` isn't used for
that purpose) and repurpose `self.ref_model` as the EMA teacher.

Memory: policy (14GB) + ref/teacher (14GB) + optimizer (56GB) = ~84GB for a 7B model
in bf16. Fits on one H100. Adding a third model would exceed 80GB.

With LoRA/PEFT, the ref model is free (disable adapter), and only adapter weights
need EMA tracking.

### 3. Student Top-K Indices Shared With Teacher

When computing top-K KL divergence, the student's top-K vocabulary indices are
computed first, then the teacher's logits are gathered at those SAME indices. Both
distributions are evaluated on the same K vocabulary items.

If you compute top-K independently for student and teacher, you're comparing
different subsets and the KL is meaningless.

### 4. EMA Update Timing

verl: EMA update fires once after ALL PPO epochs on a batch complete.
TRL: fires when `global_step % num_iterations == 0` (boundary before next generation).

With `num_iterations=1` (default), this is every optimizer step.

### 5. Import Order for Unsloth

Unsloth's `PatchFastRL("GRPO", ...)` must be called BEFORE `from sdpo_rl import
SDPOTrainer`. Python's MRO resolves `super()` calls at call time, so if Unsloth
patches `GRPOTrainer` methods before our subclass is imported, `super()` calls in
`_generate_and_score_completions` will go to the patched (optimized) version.

### 6. Config Defaults: Paper vs YAML

The verl `actor.yaml` has different defaults than what the paper's experiments use.
The experiment scripts override via command-line args. Our `SDPOConfig` defaults
match the **experiment scripts** (which produce the paper's results):

| Parameter | Paper/Scripts | verl YAML Default |
|-----------|--------------|-------------------|
| `alpha` | 0.5 (JSD) | 0.0 (forward KL) |
| `dont_reprompt_on_self_success` | True | False |
| `remove_thinking_from_demonstration` | True | False |
| `include_environment_feedback` | True | False |

---

## The 9 Gotchas

These were discovered during deep research and are critical for a correct implementation.

### Gotcha 1: No Lambda Blending
See Architecture Decision #1 above. SDPO replaces GRPO loss entirely.

### Gotcha 2: Shared Top-K Indices
See Architecture Decision #3 above.

### Gotcha 3: Tail Bucket Numerical Stability
Computing `log(1 - sum(exp(top_k_log_probs)))` for the tail bucket can produce NaN/Inf.
The fix (matching verl):
```python
log_s = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
log_s = torch.clamp(log_s, max=-1e-7)  # force sum < 1
tail_log = torch.log(-torch.expm1(log_s))
```
This is implemented and tested (including bfloat16 and near-one-hot edge cases).

### Gotcha 4: EMA Update Timing
See Architecture Decision #4 above.

### Gotcha 5: Self-Distillation Mask
Not every sample has a successful peer or feedback. The `self_distillation_mask` is
`(batch_size,)` and zeros out samples without teacher signal. These contribute zero
loss. A batch where nothing succeeds → loss = 0 → no gradient update. This is correct
behavior per the paper.

### Gotcha 6: Three Models = OOM
See Architecture Decision #2 above.

### Gotcha 7: compute_loss Needs Full Logits
TRL's `_get_per_token_logps_and_entropies` does a forward pass, extracts sampled-token
log probs via `selective_log_softmax`, and DISCARDS the full logits. For SDPO we need
top-K logits from the same forward pass.

Since we're overriding `compute_loss` entirely (Gotcha 1), we do our own forward pass
and extract top-K logits before discarding the full tensor. No redundant forward pass.

### Gotcha 8: Reward Function Feedback Format
TRL reward functions return `list[float]`. SDPO needs `{"score": float, "feedback": str}`.
The trainer must detect which format is returned and extract feedback accordingly.
Standard `list[float]` functions still work — they just provide no feedback.

### Gotcha 9: Config Defaults Mismatch
See Architecture Decision #6 above.

---

## File-by-File Reference

### `src/sdpo_rl/distillation.py`

The mathematical core. Contains:

- `add_tail_bucket(topk_log_probs)` — appends residual probability mass as (K+1)th entry
- `top_k_kl_divergence(student_topk, teacher_topk, alpha, add_tail)` — KL with JSD support
- `apply_importance_sampling_correction(loss, student_lp, old_lp, is_clip)` — off-policy correction
- `aggregate_loss(loss_mat, response_mask, sd_mask)` — token-mean with self-distillation mask
- `compute_self_distillation_loss(...)` — full SDPO loss (the main entry point)

All match `verl/trainer/ppo/core_algos.py::compute_self_distillation_loss` exactly.

### `src/sdpo_rl/reprompting.py`

Teacher prompt construction. Contains:

- `remove_thinking_tags(text)` — strips `<think>...</think>` blocks
- `select_demonstration(idx, uids, rewards, completions, ...)` — finds successful peer
- `build_teacher_prompts(prompts, solutions, feedback, ...)` — assembles templates
- `compute_self_distillation_mask(solutions, feedback)` — which samples have teacher signal

Default templates match `verl/workers/config/actor.py::SelfDistillationConfig`.

### `src/sdpo_rl/teacher.py`

EMA teacher management. Contains:

- `ema_update(teacher, student, rate)` — `θ_t = (1-α)θ_t + αθ_s`
- `EMATeacherCallback(teacher_model, student_model, update_rate, num_iterations)` — timing

Matches `verl/workers/actor/dp_actor.py::DataParallelPPOActor._update_teacher`.

### `src/sdpo_rl/config.py`

`SDPOConfig` dataclass. Defaults match paper experiment scripts. Validates alpha range,
teacher mode, truncation side, topk.

### `src/sdpo_rl/utils.py`

`check_unsloth_import_order()` — runtime detection that warns if Unsloth is installed
but PatchFastRL wasn't called before importing sdpo_rl.

---

## How SDPOTrainer Works

`SDPOTrainer` subclasses TRL's `GRPOTrainer`. It overrides two methods:

1. **`_generate_and_score_completions(inputs)`** — calls `super()`, then adds
   teacher prompt construction, demo selection, tokenization via `apply_chat_template`,
   and self-distillation mask computation.

2. **`compute_loss(model, inputs, ...)`** — does NOT call `super()`. Runs student
   and teacher forward passes, extracts top-K logits, and computes the distillation loss.

UID grouping uses TRL's `RepeatSampler` structure: indices `[i*G : (i+1)*G]` share
a prompt, which matches how TRL groups rollouts for advantage computation.

`logits_to_keep` is set to `completion_length` for both student and teacher — the
teacher's extra prompt tokens only affect attention context, not extracted positions.

---

## Environment

- Python managed by `uv` (v0.9.29)
- Virtual env at `.venv/`
- Run tests: `uv run pytest`
- Add dependency: `uv add <package>`
- Build backend: `hatchling`
- Package source layout: `src/sdpo_rl/`
- Platform: Linux, tested on NVIDIA RTX 3080 (10GB)

---

## Key External References

- **Paper**: https://arxiv.org/abs/2601.20802
- **verl reference code**: https://github.com/lasgroup/SDPO
- **Key verl files**:
  - `verl/trainer/ppo/core_algos.py` — `compute_self_distillation_loss` (lines 1085-1188)
  - `verl/workers/actor/dp_actor.py` — `update_policy`, `_update_teacher`, `TrustRegionTeacher`
  - `verl/trainer/ppo/ray_trainer.py` — `_maybe_build_self_distillation_batch`, `_build_teacher_message`
  - `verl/workers/config/actor.py` — `SelfDistillationConfig` defaults
  - `verl/trainer/config/sdpo.yaml` — experiment config
- **TRL GRPOTrainer**: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
- **TRL version pinned**: `>=0.17.0` (tested with 0.24.0)
