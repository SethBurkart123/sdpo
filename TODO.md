# SDPO-Trainer: Remaining Work

## Phase 5: SDPOTrainer Integration (HIGH PRIORITY — the main deliverable)

This is the glue that connects Phases 1-4 into a working TRL trainer.

### 5a. SDPOTrainer class (`src/sdpo_trainer/trainer.py`)

- [ ] Create `SDPOTrainer(GRPOTrainer)` subclass
- [ ] Override `__init__` to:
  - Accept `SDPOConfig` as a parameter
  - Set `beta=0` (disable TRL's KL penalty — SDPO doesn't use it)
  - Repurpose `self.ref_model` as the EMA teacher (no third model)
  - Register `EMATeacherCallback` with correct `num_iterations`
  - Set tokenizer `truncation_side` for reprompting
- [ ] Override `_generate_and_score_completions` to:
  - Call `super()` (inherits Unsloth optimizations for generation)
  - Decode completions and prompts from output tensors
  - Extract UIDs from the dataset to group rollouts by prompt
  - Call `select_demonstration()` for each sample
  - Collect environment feedback from reward function returns
  - Call `build_teacher_prompts()` to construct reprompted inputs
  - Tokenize teacher prompts with `apply_chat_template`
  - Concatenate teacher prompt tokens with original response tokens
  - Build `self_distillation_mask` tensor
  - Store `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`,
    `self_distillation_mask` in the output dict
- [ ] Override `compute_loss` to:
  - If `sdpo_config.enabled is False`, fall through to `super().compute_loss()`
  - Do student forward pass extracting top-K logits during the same pass
    (override `_get_per_token_logps_and_entropies` or do own forward)
  - Do teacher forward pass with `torch.no_grad()` on teacher model using
    `teacher_input_ids` — gather teacher logits at student's top-K indices
  - Call `compute_self_distillation_loss()` from distillation.py
  - Return scalar loss
- [ ] Handle reward function feedback format:
  - Detect if reward functions return `list[dict]` vs `list[float]`
  - Extract `{"score": float, "feedback": str}` when dict format is used
  - Store feedback strings for reprompting

### 5b. Test file (`tests/test_trainer_e2e.py`)

- [ ] `test_sdpo_mode_replaces_grpo_loss_entirely` — verify GRPO advantages are NOT used
- [ ] `test_teacher_prompt_constructed_correctly` — decode teacher_input_ids and verify template
- [ ] `test_student_topk_passed_to_teacher` — assert indices match
- [ ] `test_ema_updates_ref_model_in_place` — verify ref_model weights change after training
- [ ] `test_no_third_model_created` — verify only 2 model copies exist
- [ ] `test_zero_teacher_coverage_zero_loss` — batch where no rollouts succeed
- [ ] `test_reward_func_dict_format_extracts_feedback` — structured reward returns
- [ ] `test_reward_func_float_format_no_feedback` — standard TRL reward returns
- [ ] `test_training_loop_completes_10_steps` — smoke test with tiny model (Qwen/Qwen2.5-0.5B or similar)
- [ ] `test_loss_decreases_over_steps` — loss should trend down on a trivial task

### 5c. Critical implementation details (GOTCHAS — read HANDOVER.md)

- `compute_loss` is a FULL override in SDPO mode — do NOT call `super().compute_loss()`
- Top-K indices from the student must be reused for teacher (not computed independently)
- Teacher forward pass uses different input_ids (reprompted) but SAME response tokens
- The `_prepare_inputs` buffering means teacher data must flow through the buffer correctly
- Need to handle `logits_to_keep` alignment between teacher and student (different prompt lengths)

---

## Phase 6: Verification Against verl Reference

- [ ] `tests/test_reference_match.py`:
  - Feed identical synthetic inputs to verl's `compute_self_distillation_loss` and ours
  - Assert outputs match within floating-point tolerance
  - This requires either:
    (a) extracting the reference function from verl as a test dependency, or
    (b) hardcoding expected outputs from a manual run of the verl code

---

## Phase 7: Unsloth Compatibility Testing

- [ ] `tests/test_unsloth_compat.py`:
  - `test_import_order_correct` — PatchFastRL before import
  - `test_import_order_wrong_raises_warning` — import before PatchFastRL
  - `test_generation_uses_unsloth_optimization` — verify MRO shows patched class
- [ ] Requires `unsloth` as a test dependency (GPU needed)

---

## Phase 8: Examples & Documentation

- [ ] `examples/basic_sdpo.py` — minimal SDPO training on a math dataset
- [ ] `examples/sdpo_with_unsloth.py` — Unsloth + SDPO on free Colab GPU
- [ ] `examples/sdpo_rich_feedback.py` — code task with test case feedback
- [ ] `examples/sdpo_vs_grpo.py` — side-by-side comparison script
- [ ] `notebooks/sdpo_quickstart.ipynb` — Colab notebook (free T4)
- [ ] Expand README.md with:
  - Installation instructions
  - Quick start code
  - Architecture diagram
  - Configuration reference
  - Unsloth usage guide

---

## Phase 9: Test-Time Self-Distillation (OPTIONAL)

- [ ] `src/sdpo_trainer/test_time.py`:
  - Standalone inference-time iterative refinement
  - Generate N candidates, score with reward function, use best as demo for next round
  - Does not require training — works with any model
- [ ] `tests/test_test_time.py`
- [ ] `examples/test_time_distillation.py`

---

## Phase 10: CI & Packaging

- [ ] GitHub Actions workflow:
  - Run `uv run pytest` on push
  - CPU-only tests (Phases 1-4 tests don't need GPU)
  - Optional GPU tests (Phase 5 e2e) via self-hosted runner or skipped in CI
- [ ] PyPI publishing workflow
- [ ] License file (Apache-2.0)

---

## Known Technical Debt

1. **`_generate_and_score_completions` override is complex** — it needs to:
   - Hook into TRL's reward computation to intercept dict-format returns
   - Manage UID grouping (TRL doesn't have a concept of UIDs natively — need to
     derive from the dataset or use prompt text as the UID)
   - Handle the `steps_per_generation` buffering (teacher data must be split/buffered
     alongside the standard GRPO data)

2. **Memory pressure from teacher forward pass** — the teacher sees a longer sequence
   (reprompted prompt + response) than the student. On consumer GPUs this may OOM.
   Need `distillation_micro_batch_size` config option for chunking the teacher pass.

3. **Multi-GPU / FSDP** — the EMA teacher (repurposed ref_model) is already FSDP-wrapped
   by TRL's `prepare_fsdp`. The `ema_update` function needs to handle sharded parameters
   correctly. May need `torch.distributed` coordination.

4. **vLLM weight sync** — when using vLLM for generation, TRL syncs policy weights to
   vLLM. The EMA teacher weights are separate and don't need syncing (teacher is only
   used in `compute_loss`, not generation). But verify this doesn't break.
