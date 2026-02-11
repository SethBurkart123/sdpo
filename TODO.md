# TODO: README, Examples & Docs Refresh

This file tracks the full plan for updating documentation, examples, and
onboarding materials after the Phase 1 bug fixes and Phase 2 lora_ema feature.

**Re-read this file whenever context is compressed.**

---

## Context: What Changed Recently

### Phase 1 — Bug Fixes (committed as `1d101cc`)
1. **Bug 1 (CRITICAL):** Teacher prompts now preserve system messages via
   `build_teacher_messages()` in `reprompting.py`. Wired into
   `trainer.py:_generate_and_score_completions` capturing `_last_raw_prompts`.
2. **Bug 2 (MODERATE):** `apply_chat_template_kwargs: dict` added to `SDPOConfig`.
   Forwarded to `apply_chat_template` with `continue_final_message=False`.
   Also added `"lora_ema"` to valid `teacher_mode` values in config validation.
3. **Bug 3 (MINOR):** `teacher_per_token_logps` now computed via `torch.gather`
   in `compute_loss` and passed to `compute_self_distillation_loss`.

### Phase 2 — lora_ema Teacher Mode (not yet committed)
New `teacher_mode="lora_ema"` for PEFT/LoRA models. Uses PEFT multi-adapter API
to maintain two LoRA adapters ("default" = student, "sdpo_teacher" = teacher)
on a shared base model. Saves ~4GB for 7B QLoRA.

New functions in `teacher.py`:
- `init_lora_ema_teacher(model)` — creates teacher adapter, copies weights, freezes
- `collect_lora_adapter_pairs(model, student, teacher)` — finds param pairs
- `ema_update_lora_adapters(model, rate)` — in-place EMA on adapter params only
- `LoraEMATeacherCallback` — step-gated callback wrapping adapter EMA
- `LORA_EMA_TEACHER_ADAPTER = "sdpo_teacher"` — constant

Trainer changes:
- `__init__`: lora_ema branch skips deepcopy, calls `init_lora_ema_teacher`
- `compute_loss`: switches adapters for teacher forward pass

New exports in `__init__.py`. 17 new tests in `test_teacher.py`.
`peft>=0.14.0` added as optional dep in `pyproject.toml` under `[lora]`.

### Current Test Status
- 133 non-GPU tests passing, 0 failures
- 10 tests deselected (GPU-marked)
- Environment: RTX 3080 10GB, Python 3.11, peft 0.18.1, Unsloth 2026.1.4

---

## CRITICAL: Rules

1. **Never change base SDPO algorithm** — our implementation must match
   `SDPO_reference/` exactly. Only change integration/plumbing code.
2. **Use Perplexity** to diagnose GPU issues, Unsloth problems, etc.
3. **TDD**: Run examples before and after changes. Fix issues found.
4. **peft must be a DEFAULT dependency** — `pip install sdpo-rl` should install
   it. Currently it's optional under `[lora]`. This needs fixing.

---

## Task List

### 0. Fix peft as default dependency
- [ ] Move `peft>=0.14.0` from `[project.optional-dependencies].lora` to
  `[project].dependencies` in `pyproject.toml`
- [ ] Remove the `lora = [...]` optional group (or keep for backwards compat)
- [ ] Update any docs/README that mention `pip install sdpo-rl[lora]`

### 1. Fix and verify existing examples

#### 1a. Run `examples/basic_sdpo.py`
- [ ] Run the script on GPU: `.venv/bin/python examples/basic_sdpo.py`
- [ ] Fix any errors caused by recent changes
- [ ] Verify it completes training without crash
- [ ] Note output for README (runtime, any metrics)

#### 1b. Run `examples/sdpo_rich_feedback.py`
- [ ] Run the script on GPU
- [ ] Fix any errors
- [ ] Verify completion

#### 1c. Run `examples/sdpo_with_unsloth.py`
- [ ] Run the script on GPU
- [ ] Fix any Unsloth-specific errors (import order, patches)
- [ ] Verify completion

### 2. Write new lora_ema example

#### 2a. Create `examples/sdpo_lora_ema.py`
- [ ] Write a new example demonstrating `teacher_mode="lora_ema"` with PEFT
- [ ] Use plain HF PEFT (not Unsloth) — `peft.get_peft_model()`
- [ ] Math or reasoning task (same pattern as basic_sdpo.py)
- [ ] Comments highlighting: no deepcopy, shared base, ~4GB savings
- [ ] Show `SDPOConfig(teacher_mode="lora_ema")` prominently

#### 2b. Run `examples/sdpo_lora_ema.py`
- [ ] Run the script on GPU
- [ ] Fix any errors
- [ ] Verify completion

### 3. Add example smoke tests

#### 3a. Create `tests/test_examples.py`
- [ ] Write GPU-marked pytest tests that run each example
- [ ] Each test calls the example's `main()` or runs via subprocess
- [ ] Assert no exceptions
- [ ] Mark with `@pytest.mark.gpu`

#### 3b. Run example tests
- [ ] Run: `.venv/bin/python -m pytest tests/test_examples.py -m gpu -v`
- [ ] All 4 examples should pass

### 4. Update README.md

Changes needed:
- [ ] Remove the "218 passing" test badge (badges go stale)
- [ ] Keep warning banner as-is
- [ ] Add lora_ema section under "Training Modes" showing:
  ```python
  SDPOConfig(teacher_mode="lora_ema")
  ```
  With explanation of memory savings for QLoRA users
- [ ] Update SDPOConfig Reference to include:
  - `apply_chat_template_kwargs` field
  - `teacher_mode` updated to show "lora_ema" option
- [ ] Update Examples table to include `sdpo_lora_ema.py`
- [ ] Update GPU Requirements table with lora_ema column
- [ ] Keep Quick Start, Unsloth, and other sections as-is
- [ ] No mention of `pip install sdpo-rl[lora]` (peft is now default dep)

### 5. Update stale docs

#### 5a. VERIFICATION.md
- [ ] Update test counts (115 → 133+)
- [ ] Add lora_ema functions to teacher verification table
- [ ] Add `apply_chat_template_kwargs` to config verification
- [ ] Add `"lora_ema"` to teacher_mode values
- [ ] Note the 3 bugs found and fixed

#### 5b. DEVIATIONS.md
- [ ] Update test count
- [ ] Add `teacher_mode="lora_ema"` as novel extension (not in verl reference)
- [ ] Add lora_ema as memory optimization strategy
- [ ] Note `apply_chat_template_kwargs` as TRL-specific addition

#### 5c. HANDOVER.md
- [ ] Update test counts (total and per-module)
- [ ] Add 3 new bugs to bug history
- [ ] Add new functions to file reference (teacher.py, reprompting.py)
- [ ] Update SDPOConfig docs with new fields
- [ ] Add lora_ema architecture decision
- [ ] Mention peft dependency

#### 5d. UNSLOTH_INTEGRATION.md
- [ ] Add lora_ema + Unsloth interaction section
- [ ] Update memory table with lora_ema option
- [ ] Show `teacher_mode` in config examples

### 6. Update `examples/README.md`
- [ ] Add `sdpo_lora_ema.py` entry
- [ ] Update GPU requirements table
- [ ] Add lora_ema config pattern to Configuration Patterns section

### 7. Final validation
- [ ] Run full non-GPU test suite: all pass
- [ ] Run GPU example tests: all pass
- [ ] Git commit everything

---

## File Reference (for context after compression)

### Source files modified in Phase 1+2:
- `src/sdpo_rl/config.py` — SDPOConfig with `apply_chat_template_kwargs`, `"lora_ema"` valid
- `src/sdpo_rl/reprompting.py` — Added `build_teacher_messages()`
- `src/sdpo_rl/trainer.py` — Bug fixes + lora_ema init/compute_loss
- `src/sdpo_rl/teacher.py` — lora_ema functions + constants
- `src/sdpo_rl/__init__.py` — New exports
- `pyproject.toml` — peft optional dep (needs → default dep)

### Test files:
- `tests/test_config.py` — 20 tests
- `tests/test_distillation.py` — 29 tests
- `tests/test_reprompting.py` — 48+ tests
- `tests/test_teacher.py` — 24 tests (7 original + 17 lora_ema)
- `tests/test_unsloth_integration.py` — 2 tests (GPU-marked)

### Example files:
- `examples/basic_sdpo.py` — Math addition, core SDPO loop
- `examples/sdpo_with_unsloth.py` — Unsloth + QLoRA reasoning
- `examples/sdpo_rich_feedback.py` — Code gen with test execution feedback
- `examples/sdpo_lora_ema.py` — TO BE CREATED

### Docs to update:
- `README.md` — Main onboarding doc
- `VERIFICATION.md` — Line-by-line verification checklist
- `DEVIATIONS.md` — Differences from verl reference
- `HANDOVER.md` — Architecture decisions, file reference, gotchas
- `UNSLOTH_INTEGRATION.md` — Unsloth compatibility guide
- `SDPO_AUDIT.md` — Comprehensive audit (already up to date)
- `examples/README.md` — Example walkthrough
