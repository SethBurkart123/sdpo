# SDPO Implementation Audit & Change Plan

This document captures the results of a comprehensive audit comparing our TRL-based SDPO implementation against the reference implementation at `SDPO_reference/` (lasgroup/SDPO, arxiv:2601.20802).

It serves as both a record of findings and a living tracker for fixes.

---

## Table of Contents

1. [Correctness Bugs](#1-correctness-bugs)
2. [Verified Correct](#2-verified-correct)
3. [Intentional Differences](#3-intentional-differences)
4. [Phase 1: Bug Fixes](#4-phase-1-bug-fixes)
5. [Phase 2: lora_ema Teacher Mode](#5-phase-2-lora_ema-teacher-mode)
6. [Research Context](#6-research-context)

---

## 1. Correctness Bugs

### Bug 1: Teacher prompt drops system messages (CRITICAL)

**Status:** `[x] FIXED`

**Location:** `src/sdpo_rl/trainer.py:220`

**Current (wrong):**
```python
teacher_messages = [[{"role": "user", "content": tp}] for tp in teacher_prompts_list]
```

**Reference (correct):** `SDPO_reference/verl/trainer/ppo/ray_trainer.py:710-744`
```python
def _build_teacher_message(i: int) -> list[dict]:
    system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]  # preserve system msgs
    # ... build reprompt_text ...
    return system_messages + [{"role": "user", "content": reprompt_text}]
```

**Problem:** The current code wraps the entire teacher prompt in a single `{"role": "user"}` message, discarding any system messages from the original prompt. For datasets with system prompts (e.g., "You are a helpful assistant"), this changes the teacher's context. The reference preserves all messages except the last user message, replacing only the user turn with the reprompted version.

**Impact:** Different teacher logits → different distillation signal → different training outcome.

**Root cause:** TRL's `_generate_and_score_completions` does not store the raw prompt messages in its output dict. Our override decodes `prompt_ids` back to text (line 181), losing the structured message format. We need to capture the raw messages.

**Fix:** Stash `inputs[i]["prompt"]` (the raw message list) during `_generate_and_score_completions`, then use the system messages from that list when building teacher messages.

---

### Bug 2: Missing `enable_thinking` in apply_chat_template (MODERATE)

**Status:** `[x] FIXED`

**Location:** `src/sdpo_rl/trainer.py:224-233`

**Current (missing parameter):**
```python
tokenized = self.processing_class.apply_chat_template(
    teacher_messages,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
    padding=True,
    truncation=True,
    max_length=self.sdpo_config.max_reprompt_length,
)
```

**Reference:** `SDPO_reference/verl/trainer/ppo/ray_trainer.py:749-761`
```python
enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True)
teacher_prompt = self.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    continue_final_message=False,
    add_generation_prompt=True,
    enable_thinking=enable_thinking,
    max_length=self_distillation_cfg.max_reprompt_len,
    padding=True,
    truncation=True,
)
```

**Problem:** For thinking-enabled models (Qwen3, QwQ, etc.), `enable_thinking` controls whether `<think>` tokens are added to the chat template output. Omitting it relies on the tokenizer's default, which may differ from what the user configured for generation. Also missing: `continue_final_message=False`.

**Fix:** Add `apply_chat_template_kwargs` field to `SDPOConfig` (dict, default `{}`). Pass these kwargs through to `apply_chat_template`, plus always pass `continue_final_message=False`.

---

### Bug 3: teacher_log_probs always passed as None (MINOR)

**Status:** `[x] FIXED`

**Location:** `src/sdpo_rl/trainer.py:345`

**Current:**
```python
loss, metrics = compute_self_distillation_loss(
    ...
    teacher_log_probs=None,  # ← Always None
    ...
)
```

**Reference:** `SDPO_reference/verl/workers/actor/dp_actor.py:834-835`
```python
pg_loss, pg_metrics = compute_self_distillation_loss(
    student_log_probs=log_prob,
    teacher_log_probs=teacher_log_prob,  # ← Actual teacher per-token log-probs
    ...
)
```

**Problem:** The teacher's per-token log-probabilities (log prob of the actual completion token under the teacher) are always `None`. This is used in the non-full-logit fallback path (REINFORCE-style reverse KL at `distillation.py:234`). Since `full_logit_distillation=True` by default, this code path isn't normally hit, but it would crash if someone set `distillation_topk=None`.

**Fix:** After the teacher forward pass, gather teacher log-probs at completion token indices (same as we do for the student at line 319-321), and pass them through.

---

## 2. Verified Correct

### Core Distillation Math ✅

The following were verified identical to the reference (max error 1.4e-08 across 200 steps per `test_reference_match.py`):

| Component | Our file | Ref file | Status |
|-----------|----------|----------|--------|
| Top-K tail bucket (`add_tail`) | `distillation.py:24-49` | `core_algos.py:1112-1116` | **Identical** |
| Top-K renormalization | `distillation.py:86-87` | `core_algos.py:1120-1122` | **Identical** |
| Forward KL (α=0) | `distillation.py:89-91` | `core_algos.py:1139-1142` | **Identical** |
| Reverse KL (α=1) | `distillation.py:92-94` | `core_algos.py:1143-1146` | **Identical** |
| Generalized JSD (0<α<1) | `distillation.py:96-112` | `core_algos.py:1148-1161` | **Identical** |
| KL sum over vocab dim | `distillation.py:115` | `core_algos.py:1163` | **Identical** |
| IS correction formula | `distillation.py:147-149` | `core_algos.py:1173-1175` | **Identical** |
| IS correction clamp values | `[-20, 20]`, `max=is_clip` | `[-20, 20]`, `max=is_clip` | **Identical** |
| Mask application | `distillation.py:174-176` | `core_algos.py:1102-1104` | **Identical** |
| Token-mean aggregation | `distillation.py:178-179` | `core_algos.py:1057-1060` | **Equivalent** (dp_size=1) |

### Reprompting Logic ✅

| Component | Status |
|-----------|--------|
| Template strings (all 3) | **Character-for-character match** |
| `select_demonstration` (first successful peer) | **Identical** |
| Self-exclusion logic | **Identical** |
| Thinking tag removal regex | **Identical** |
| `feedback_only_without_solution` logic | **Identical** |
| `self_distillation_mask` formula | **Identical** |

### Teacher Management ✅

| Component | Status |
|-----------|--------|
| EMA formula: `teacher = (1-rate)*teacher + rate*student` | **Identical** |
| EMA fires once per optimizer step (not per micro-batch) | **Correct** via `_EMAStepCallback.on_step_end` |
| `torch.no_grad()` for EMA update | **Correct** |
| `is_floating_point()` filter for quantized params | **Correct extension** (ref doesn't handle LoRA) |

### Forward Pass Alignment ✅

| Component | Status |
|-----------|--------|
| Logit slicing: `[:, -response_length-1:-1, :]` | **Identical** to reference's `_forward_micro_batch` |
| Teacher uses student's top-K indices | **Correct** (`trainer.py:338`) |
| `logits_to_keep=logits_to_keep + 1` | **Correct** |
| Teacher forward under `torch.no_grad()` | **Correct** |

---

## 3. Intentional Differences

These differ from the reference's *source code defaults* but match the *paper's experiment scripts*. This is documented in `config.py:29-30`.

| Parameter | Ref default (actor.yaml) | Our default | Paper experiment |
|-----------|--------------------------|-------------|------------------|
| `alpha` | 0.0 (forward KL) | **0.5** (JSD) | 0.5 |
| `is_clip` | None (disabled) | **2.0** | 2.0 |
| `dont_reprompt_on_self_success` | False | **True** | True |
| `remove_thinking_from_demonstration` | False | **True** | True |
| `include_environment_feedback` | False | **True** | True |
| `environment_feedback_only_without_solution` | False | **True** | True |

### Other deliberate simplifications:

| Difference | Justification |
|-----------|---------------|
| Only `token-mean` aggregation supported | Only mode used in paper experiments |
| IS correction silently skips when `old_log_probs=None` | Safer for on-policy use (ref raises ValueError) |
| No NaN guard in aggregation (`torch.where`) | Unlikely to trigger; adds unnecessary overhead |
| `dp_size` multiplier omitted | TRL/HF Trainer handles gradient averaging externally |
| Added diagnostic metrics (`sdpo/kl_mean`, etc.) | Enhancement — ref returns empty metrics dict |
| Prompt text decoded from `prompt_ids` | TRL doesn't pass raw messages through to outputs |

---

## 4. Phase 1: Bug Fixes

### Fix 1: Preserve system messages in teacher prompt

**Files changed:**
- `src/sdpo_rl/trainer.py` — `_generate_and_score_completions`
- `src/sdpo_rl/reprompting.py` — `build_teacher_messages` (new function)
- `tests/test_reprompting.py` — new tests for structured message building

**Approach:**
1. In `_generate_and_score_completions`, capture raw prompt messages from `inputs`:
   ```python
   prompts_raw = [x["prompt"] for x in inputs]  # list of message lists
   ```
   This must happen BEFORE calling `super()._generate_and_score_completions(inputs)` since `inputs` is consumed by the parent.

2. Extract system messages per sample: `system_msgs = raw_messages[:-1]`

3. New function `build_teacher_messages()` returns `list[list[dict]]` — proper chat message structures with system messages preserved:
   ```python
   [system_msgs + [{"role": "user", "content": reprompt_text}], ...]
   ```

4. Replace the current single-user-message construction at line 220.

**Test (TDD):** Test that `build_teacher_messages` preserves system messages from the original prompt.

**Status:** `[x] DONE` — `build_teacher_messages()` added to `reprompting.py`, wired into `trainer.py:_generate_and_score_completions`. 5 tests in `test_reprompting.py::TestBuildTeacherMessages` all pass.

---

### Fix 2: Pass `enable_thinking` and `continue_final_message` to apply_chat_template

**Files changed:**
- `src/sdpo_rl/config.py` — add `apply_chat_template_kwargs` field
- `src/sdpo_rl/trainer.py` — pass kwargs to `apply_chat_template`

**Approach:**
1. Add to `SDPOConfig`:
   ```python
   apply_chat_template_kwargs: dict = field(default_factory=dict)
   ```

2. In the `apply_chat_template` call, merge these kwargs:
   ```python
   chat_kwargs = {
       "tokenize": True,
       "return_tensors": "pt",
       "return_dict": True,
       "continue_final_message": False,
       "add_generation_prompt": True,
       "padding": True,
       "truncation": True,
       "max_length": self.sdpo_config.max_reprompt_length,
       **self.sdpo_config.apply_chat_template_kwargs,
   }
   tokenized = self.processing_class.apply_chat_template(teacher_messages, **chat_kwargs)
   ```

**Test (TDD):** Test that `apply_chat_template_kwargs` is passed through correctly.

**Status:** `[x] DONE` — `apply_chat_template_kwargs: dict` added to `SDPOConfig`, `"lora_ema"` added to valid `teacher_mode` values. 4 tests in `test_config.py::TestSDPOConfigChatTemplateKwargs` all pass.

---

### Fix 3: Compute and pass teacher per-token log-probs

**Files changed:**
- `src/sdpo_rl/trainer.py` — `compute_loss`

**Approach:** After the teacher forward pass (line 332-335), add:
```python
teacher_per_token_logps = torch.gather(
    teacher_log_probs_full, dim=-1, index=completion_ids.unsqueeze(-1)
).squeeze(-1)
```
Then pass to `compute_self_distillation_loss` as `teacher_log_probs=teacher_per_token_logps`.

**Test (TDD):** Test that the fallback path (no topk) works when `teacher_log_probs` is provided.

**Status:** `[x] DONE` — `teacher_per_token_logps` computed via `torch.gather` in `compute_loss` and passed through. 2 tests in `test_distillation.py` for fallback path pass.

---

## 5. Phase 2: lora_ema Teacher Mode

### Background

The current implementation creates the teacher via `copy.deepcopy(self.model)` (trainer.py:104). For QLoRA models, this duplicates the entire 4-bit quantized base (~3.5-4GB for 7B) even though the EMA only updates the LoRA adapter weights (~100-200MB). The base weights are identical between student and teacher and never change.

### Design: `teacher_mode="lora_ema"`

Uses PEFT's multi-adapter API to maintain two LoRA adapter sets on a shared base model:

- **"default" adapter** (student): trainable, receives gradient updates
- **"sdpo_teacher" adapter** (teacher): frozen, receives EMA updates from student

Memory savings: ~3.5-4GB for QLoRA 7B models (eliminates base weight duplication).

Quality: **Identical to current EMA mode** — the EMA update targets the same adapter parameters.

### Implementation Plan

**Config change:** Add `"lora_ema"` to valid `teacher_mode` values.

**Init (trainer.py `__init__`):**
```python
if self.sdpo_config.teacher_mode == "lora_ema":
    if not hasattr(model, 'peft_config'):
        raise ValueError("teacher_mode='lora_ema' requires a PEFT model with LoRA adapters")
    # Get the current adapter config
    active_adapter = model.active_adapter
    adapter_config = model.peft_config[active_adapter]
    # Add a second "teacher" adapter with identical config
    model.add_adapter("sdpo_teacher", adapter_config)
    # Copy student weights to teacher
    _copy_adapter_weights(model, src=active_adapter, dst="sdpo_teacher")
    # Switch back to student
    model.set_adapter(active_adapter)
    # ref_model IS the same model (different adapter)
    self.ref_model = self.model
    self.ema_callback = LoraEMATeacherCallback(...)
```

**Forward pass (trainer.py `compute_loss`):**
```python
if self.sdpo_config.teacher_mode == "lora_ema":
    self.model.set_adapter("sdpo_teacher")
    with torch.no_grad():
        teacher_outputs = self.model(teacher_input_ids, ...)
    self.model.set_adapter("default")  # back to student
else:
    with torch.no_grad():
        teacher_outputs = self.ref_model(teacher_input_ids, ...)
```

**EMA update (teacher.py):**
```python
def ema_update_adapters(model, student_adapter, teacher_adapter, rate):
    """EMA update only on adapter weights — no base model touching."""
    student_sd = model.get_adapter_state_dict(student_adapter)
    teacher_sd = model.get_adapter_state_dict(teacher_adapter)
    with torch.no_grad():
        for key in teacher_sd:
            teacher_sd[key].mul_(1.0 - rate).add_(student_sd[key], alpha=rate)
```

**Dependency:** `peft` becomes an optional dependency (guarded by try/except).

**Status:** `[x] DONE`

**Files changed:**
- `src/sdpo_rl/teacher.py` — Added `collect_lora_adapter_pairs`, `ema_update_lora_adapters`, `init_lora_ema_teacher`, `LoraEMATeacherCallback`, `LORA_EMA_TEACHER_ADAPTER` constant.
- `src/sdpo_rl/trainer.py` — `__init__` handles `lora_ema` mode (skips deepcopy, calls `init_lora_ema_teacher`). `compute_loss` switches adapters for teacher forward pass.
- `src/sdpo_rl/__init__.py` — Exports new lora_ema symbols.
- `pyproject.toml` — Added `lora` optional dependency (`peft>=0.14.0`).
- `tests/test_teacher.py` — 17 new tests covering adapter pair collection, init, EMA update, callback, base weight immutability, convergence, and adapter-switching output divergence.

**Implementation details:**
- Uses PEFT's multi-adapter API: `model.add_adapter("sdpo_teacher", config)` creates a second set of LoRA weights on the shared base model.
- `init_lora_ema_teacher` copies student weights into teacher adapter, freezes teacher params, and leaves student ("default") active.
- `ema_update_lora_adapters` iterates `module.lora_A[adapter]` / `module.lora_B[adapter]` directly — no state_dict overhead, in-place update with dtype handling.
- `compute_loss` switches to teacher adapter with `model.set_adapter(LORA_EMA_TEACHER_ADAPTER)`, runs forward pass, then switches back to "default".
- Teacher adapter weights are bf16 when created by PEFT (matching base model dtype). EMA update handles float32→bf16 conversion via `.to(dtype=teacher_p.dtype)`.
- All 133 tests pass (24 teacher tests, 109 other non-GPU tests).

---

## 6. Research Context

### Why EMA Matters (SDPO Paper Table 4)

| Teacher variant | Best Accuracy | Avg Accuracy |
|---|---|---|
| Current weights (no regularization) | 36.1% ± 1.6 | 29.8% ± 1.3 |
| Frozen (initial checkpoint) | 48.8% ± 0.7 | 44.4% ± 0.2 |
| Trust-region (α=0.01) | **50.6% ± 0.9** | **45.6% ± 0.2** |
| EMA (α=0.01) | 49.3% ± 0.3 | 45.3% ± 0.2 |

The "current weights (no regularization)" baseline **diverges** even with context asymmetry present. EMA provides the stability needed for iterative self-improvement.

### Why lora_ema Works

For QLoRA models:
- Base weights are quantized (uint8) and frozen — identical between student and teacher
- Only LoRA adapter weights (float16/32) differ and receive EMA updates
- `copy.deepcopy` wastefully duplicates the frozen base (~4GB for 7B)
- Multi-adapter API shares the base and only duplicates adapter weights (~100-200MB)
- **Same EMA quality, ~20× less extra memory**

### Key References

- SDPO paper: arxiv:2601.20802 (Hübotter et al., 2026)
- "How to Scale Your EMA" (Busbridge et al., NeurIPS 2023): `ρ̂ = ρ^κ`
- PEFT multi-adapter API: `model.add_adapter()`, `model.set_adapter()`
- Context distillation: Snell et al. (2024), Anthropic
