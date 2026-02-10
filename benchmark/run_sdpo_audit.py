#!/usr/bin/env python3
"""
Run 3: SDPO with reference loss verification.

Same as run_sdpo.py, but subclasses SDPOTrainer to also compute the verl
reference distillation loss on every micro-batch. Logs |our_loss - ref_loss|
to prove correctness under real training dynamics (EMA drift, gradient
accumulation, mixed precision, etc.).

Usage:
    uv run python benchmark/run_sdpo_audit.py [--max-steps N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

# Unsloth must be patched before importing trainers
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import GRPOConfig
from transformers import TrainerCallback

# benchmark/ modules
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    BATCH_SIZE,
    EVAL_EVERY,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_COMPLETION_LENGTH,
    MAX_STEPS_DEFAULT,
    NUM_GENERATIONS,
    RESULTS_DIR,
    SEED,
    TEMPERATURE,
    EvalCallback,
    ForceExitCallback,
    cleanup,
    load_datasets,
    load_model_and_tokenizer,
)
from csv_logger import CSVLogger
from reward_mbpp import FormatRewardFunction, MBPPRewardFunction

# Import our library AFTER PatchFastRL
from sdpo_rl import SDPOConfig, SDPOTrainer

# ---------------------------------------------------------------------------
# Reference distillation loss — extracted verbatim from verl core_algos.py
# (same functions used in tests/test_reference_match.py)
# ---------------------------------------------------------------------------


def _ref_add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _ref_renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
    return logp - logZ


def _ref_compute_self_distillation_loss(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    response_mask: torch.Tensor,
    self_distillation_mask: Optional[torch.Tensor],
    alpha: float,
    is_clip: Optional[float],
    add_tail: bool,
) -> torch.Tensor:
    """
    Reference implementation matching verl's compute_self_distillation_loss.

    Full-logit top-K path only (that's what we use in training).
    Returns the scalar loss.
    """
    # Effective mask
    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    # Prepare top-K distributions
    if add_tail:
        s_lp = _ref_add_tail(student_topk_log_probs)
        t_lp = _ref_add_tail(teacher_topk_log_probs)
    else:
        s_lp = _ref_renorm_topk_log_probs(student_topk_log_probs)
        t_lp = _ref_renorm_topk_log_probs(teacher_topk_log_probs)

    # KL divergence
    if alpha == 0.0:
        kl = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
    else:
        alpha_t = torch.tensor(alpha, dtype=s_lp.dtype, device=s_lp.device)
        mixture = torch.logsumexp(
            torch.stack([s_lp + torch.log(1 - alpha_t), t_lp + torch.log(alpha_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture, t_lp, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture, s_lp, reduction="none", log_target=True)
        kl = torch.lerp(kl_student, kl_teacher, alpha)

    per_token_loss = kl.sum(-1)

    # IS correction
    if is_clip is not None and old_log_probs is not None:
        neg_approx_kl = (student_log_probs - old_log_probs).detach()
        neg_approx_kl = torch.clamp(neg_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(neg_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    # Token-mean aggregation
    numerator = (per_token_loss * loss_mask).sum()
    denominator = loss_mask.sum().clamp(min=1.0)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Auditing trainer: wraps compute_loss to also run the reference
# ---------------------------------------------------------------------------


class AuditSDPOTrainer(SDPOTrainer):
    """
    SDPOTrainer that also computes the verl reference loss on each micro-batch.

    Instead of duplicating forward passes (which doubles VRAM), we monkey-patch
    compute_self_distillation_loss to intercept the intermediate tensors and
    run the reference computation on them. Zero extra GPU cost.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audit_delta: float = 0.0
        self._audit_our_loss: float = 0.0
        self._audit_ref_loss: float = 0.0
        self._install_audit_hook()

    def _install_audit_hook(self):
        """Patch the distillation module so every call also runs the reference."""
        import sdpo_rl.distillation as dist_mod
        import sdpo_rl.trainer as trainer_mod

        original_fn = dist_mod.compute_self_distillation_loss
        audit_trainer = self  # capture for closure

        def audited_compute_self_distillation_loss(**kwargs):
            # Run our implementation normally
            loss, metrics = original_fn(**kwargs)

            # Run reference on the same tensors (detached, float64, no grad)
            with torch.no_grad():
                stk = kwargs.get("student_topk_log_probs")
                ttk = kwargs.get("teacher_topk_log_probs")
                slp = kwargs.get("student_log_probs")
                olp = kwargs.get("old_log_probs")
                rmask = kwargs.get("response_mask")
                sdmask = kwargs.get("self_distillation_mask")

                # Only audit when we have full-logit tensors
                if stk is not None and ttk is not None:
                    ref_loss = _ref_compute_self_distillation_loss(
                        student_topk_log_probs=stk.detach().to(torch.float64),
                        teacher_topk_log_probs=ttk.detach().to(torch.float64),
                        student_log_probs=slp.detach().to(torch.float64),
                        old_log_probs=olp.detach().to(torch.float64) if olp is not None else None,
                        response_mask=rmask.to(torch.float64),
                        self_distillation_mask=sdmask.to(torch.float64) if sdmask is not None else None,
                        alpha=kwargs.get("alpha", 0.5),
                        is_clip=kwargs.get("is_clip", 2.0),
                        add_tail=kwargs.get("add_tail", True),
                    )
                    our_f64 = loss.detach().to(torch.float64)
                    delta = (our_f64 - ref_loss).abs().item()
                    audit_trainer._audit_delta = delta
                    audit_trainer._audit_our_loss = loss.item()
                    audit_trainer._audit_ref_loss = ref_loss.item()
                    metrics["sdpo/audit_delta"] = delta

            return loss, metrics

        # Patch both the module and the trainer module's imported reference
        dist_mod.compute_self_distillation_loss = audited_compute_self_distillation_loss
        trainer_mod.compute_self_distillation_loss = audited_compute_self_distillation_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SDPO audit benchmark")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT, help="Training steps")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip final 257-problem eval")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== SDPO Audit | max_steps={args.max_steps} ===")

    # Load model + data (same seed as other runs)
    model, tokenizer = load_model_and_tokenizer()
    train_ds, eval_ds = load_datasets()
    reward_fn = MBPPRewardFunction()
    format_fn = FormatRewardFunction()

    # GRPO base config
    grpo_config = GRPOConfig(
        output_dir=str(RESULTS_DIR / "sdpo_audit_checkpoints"),
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=100,
        max_steps=args.max_steps,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_steps=999999,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        seed=SEED,
        temperature=TEMPERATURE,
        reward_weights=[1.0, 0.3],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )

    # SDPO config (same as run_sdpo.py)
    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        distillation_add_tail=True,
        is_clip=2.0,
        teacher_mode="ema",
        teacher_update_rate=0.05,
        success_reward_threshold=1.0,
        dont_reprompt_on_self_success=True,
        remove_thinking_from_demonstration=True,
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,  # paper default: feedback only when no solution
    )

    # CSV loggers
    train_csv = CSVLogger(
        str(RESULTS_DIR / "sdpo_audit_train.csv"),
        fieldnames=[
            "step",
            "loss",
            "reward_mean",
            "sdpo_loss",
            "sdpo_kl_mean",
            "sdpo_teacher_coverage",
            "audit_delta",
            "audit_our_loss",
            "audit_ref_loss",
            "timestamp",
        ],
    )
    eval_csv = CSVLogger(
        str(RESULTS_DIR / "sdpo_audit_eval.csv"),
        fieldnames=["step", "pass_at_1", "timestamp"],
    )

    # Training metrics callback — captures the audit delta
    class TrainMetricsCallback(TrainerCallback):
        def __init__(self, trainer_ref):
            self.trainer_ref = trainer_ref

        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs and state.global_step > 0:
                train_csv.log(
                    {
                        "step": state.global_step,
                        "loss": logs.get("loss", ""),
                        "reward_mean": logs.get("reward", ""),
                        "sdpo_loss": logs.get("sdpo/loss", ""),
                        "sdpo_kl_mean": logs.get("sdpo/kl_mean", ""),
                        "sdpo_teacher_coverage": logs.get("sdpo/teacher_coverage", ""),
                        "audit_delta": self.trainer_ref._audit_delta,
                        "audit_our_loss": self.trainer_ref._audit_our_loss,
                        "audit_ref_loss": self.trainer_ref._audit_ref_loss,
                        "timestamp": time.time(),
                    }
                )

    # Create auditing trainer
    trainer = AuditSDPOTrainer(
        model=model,
        args=grpo_config,
        sdpo_config=sdpo_config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn, format_fn],
        train_dataset=train_ds,
    )

    # Add callbacks
    trainer.add_callback(TrainMetricsCallback(trainer))
    trainer.add_callback(
        EvalCallback(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_ds,
            eval_csv=eval_csv,
            eval_every=args.eval_every,
        )
    )

    def save_fn():
        adapter_dir = str(RESULTS_DIR / "sdpo_audit_adapter")
        print(f"Saving adapter to {adapter_dir}...", flush=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    def post_train(state):
        import csv as csv_mod

        elapsed = time.time() - start_time
        print(f"\n=== SDPO Audit complete | {args.max_steps} steps in {elapsed / 60:.1f} min ===")

        # Report max audit delta
        max_delta = 0.0
        csv_path = RESULTS_DIR / "sdpo_audit_train.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv_mod.DictReader(f):
                    d = row.get("audit_delta", "0")
                    if d:
                        max_delta = max(max_delta, float(d))
        print(f"Max |our_loss - ref_loss| across all steps: {max_delta:.2e}")
        if max_delta < 1e-4:
            print("PASS: Implementation matches reference within 1e-4.")
        else:
            print("WARNING: Delta exceeds 1e-4. Investigate bf16 rounding or a real bug.")

        train_csv.close()
        eval_csv.close()
        cleanup()

    trainer.add_callback(ForceExitCallback(max_steps=args.max_steps, save_fn=save_fn, post_train_fn=post_train))

    # Train (may not return due to TRL hang — ForceExitCallback handles exit)
    start_time = time.time()
    trainer.train()


if __name__ == "__main__":
    main()
