# SDPO-Trainer: Agent Handover Document

## What This Project Is

A standalone pip-installable library (`sdpo-trainer`) that brings **Self-Distilled
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

### Implementation (Phases 1-4 complete)

All code uses **test-driven development** — tests were written first, then
implementations made them pass. **69 tests, all passing.**

| Module | What It Does | Tests |
|--------|-------------|-------|
| `distillation.py` | Top-K KL divergence, tail bucket, JSD, IS clipping, loss aggregation | 27 |
| `reprompting.py` | Teacher prompt construction, demo selection, thinking tag removal, SD mask | 19 |
| `teacher.py` | EMA update formula, callback with TRL timing | 7 |
| `config.py` | `SDPOConfig` dataclass with paper experiment defaults | 16 |
| `utils.py` | Unsloth import order detection | 0 (runtime utility) |

Run tests: `uv run pytest`

### What Has NOT Been Done

**Phase 5: `SDPOTrainer(GRPOTrainer)` — the integration layer.** This is the main
remaining work. See `TODO.md` for the full breakdown.

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

Unsloth's `PatchFastRL("GRPO", ...)` must be called BEFORE `from sdpo_trainer import
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

### `src/sdpo_trainer/distillation.py`

The mathematical core. Contains:

- `add_tail_bucket(topk_log_probs)` — appends residual probability mass as (K+1)th entry
- `top_k_kl_divergence(student_topk, teacher_topk, alpha, add_tail)` — KL with JSD support
- `apply_importance_sampling_correction(loss, student_lp, old_lp, is_clip)` — off-policy correction
- `aggregate_loss(loss_mat, response_mask, sd_mask)` — token-mean with self-distillation mask
- `compute_self_distillation_loss(...)` — full SDPO loss (the main entry point)

All match `verl/trainer/ppo/core_algos.py::compute_self_distillation_loss` exactly.

### `src/sdpo_trainer/reprompting.py`

Teacher prompt construction. Contains:

- `remove_thinking_tags(text)` — strips `<think>...</think>` blocks
- `select_demonstration(idx, uids, rewards, completions, ...)` — finds successful peer
- `build_teacher_prompts(prompts, solutions, feedback, ...)` — assembles templates
- `compute_self_distillation_mask(solutions, feedback)` — which samples have teacher signal

Default templates match `verl/workers/config/actor.py::SelfDistillationConfig`.

### `src/sdpo_trainer/teacher.py`

EMA teacher management. Contains:

- `ema_update(teacher, student, rate)` — `θ_t = (1-α)θ_t + αθ_s`
- `EMATeacherCallback(teacher_model, student_model, update_rate, num_iterations)` — timing

Matches `verl/workers/actor/dp_actor.py::DataParallelPPOActor._update_teacher`.

### `src/sdpo_trainer/config.py`

`SDPOConfig` dataclass. Defaults match paper experiment scripts. Validates alpha range,
teacher mode, truncation side, topk.

### `src/sdpo_trainer/utils.py`

`check_unsloth_import_order()` — runtime detection that warns if Unsloth is installed
but PatchFastRL wasn't called before importing sdpo_trainer.

---

## How to Continue: Building SDPOTrainer

### The approach

`SDPOTrainer` subclasses TRL's `GRPOTrainer`. It overrides two methods:

1. **`_generate_and_score_completions(inputs)`**:
   - Call `super()` to get all GRPO outputs (generation + scoring + advantages)
   - Decode completions and prompts
   - For each sample, call `select_demonstration()` and collect feedback
   - Call `build_teacher_prompts()` to create reprompted text
   - Tokenize via `apply_chat_template` (with truncation at `max_reprompt_length`)
   - Concatenate `[teacher_prompt_tokens | response_tokens]` into `teacher_input_ids`
   - Compute `teacher_position_ids` from `teacher_attention_mask`
   - Store in output dict: `teacher_input_ids`, `teacher_attention_mask`,
     `teacher_position_ids`, `self_distillation_mask`

2. **`compute_loss(model, inputs, ...)`**:
   - Do NOT call `super()` — SDPO replaces GRPO loss entirely
   - Student forward pass: run model on original `input_ids`, extract top-K logits
     from the completion positions, compute per-token log probs
   - Teacher forward pass: with `torch.no_grad()`, run teacher model (self.ref_model)
     on `teacher_input_ids`, gather teacher logits at student's top-K indices
   - Call `compute_self_distillation_loss()` from distillation.py
   - Return scalar loss

### UID grouping problem

verl has explicit `uid` fields in its data. TRL does not. Options:
- Use the prompt text itself as the UID (hash it)
- Use the dataset index (TRL's `RepeatSampler` repeats prompts `num_generations` times
  in consecutive positions, so indices `[i*G .. (i+1)*G-1]` share a prompt)
- Add a `uid_column` config option

The consecutive-index approach is simplest and matches how TRL already groups rollouts
for advantage computation.

### logits_to_keep alignment

Student and teacher have different prompt lengths (teacher prompt is longer due to
reprompting). But they share the SAME response tokens. The `logits_to_keep` parameter
controls how many positions from the end of the sequence to return logits for.

For both student and teacher, `logits_to_keep = completion_length`. The teacher's
extra prompt tokens only affect the attention context, not the positions we extract
logits from. This alignment is natural — just use the same `completion_length` for both.

### Testing approach

Phase 5 integration tests need a model. Options:
- Use a tiny model on CPU (e.g., `sshleifer/tiny-gpt2` or create a random
  `GPT2LMHeadModel` with 2 layers). This is fast but won't test chat templates.
- Use `Qwen/Qwen2.5-0.5B-Instruct` on GPU. Realistic but requires GPU hardware.
- Mock the model's forward pass entirely. Fast and deterministic but tests less.

Recommend: tiny random model on CPU for CI, real model on GPU for validation.

---

## Environment

- Python managed by `uv` (v0.9.29)
- Virtual env at `.venv/`
- Run tests: `uv run pytest`
- Add dependency: `uv add <package>`
- Build backend: `hatchling`
- Package source layout: `src/sdpo_trainer/`
- Platform: macOS (darwin), no GPU locally

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
