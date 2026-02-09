#!/usr/bin/env python3
"""
Run 2: SDPO on MBPP with rich feedback.

SDPOTrainer with Unsloth + 4-bit QLoRA.
Reward function provides scalar scores AND rich feedback (tracebacks,
assertion errors) that SDPO bakes into teacher prompts.

Usage:
    uv run python benchmark/run_sdpo.py [--max-steps N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Unsloth must be patched before importing trainers
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

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
from sdpo_trainer import SDPOConfig, SDPOTrainer


def main():
    parser = argparse.ArgumentParser(description="SDPO benchmark")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT, help="Training steps")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip final 257-problem eval")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== SDPO | max_steps={args.max_steps} ===")

    # Load model + data
    model, tokenizer = load_model_and_tokenizer()
    train_ds, eval_ds = load_datasets()
    reward_fn = MBPPRewardFunction()
    format_fn = FormatRewardFunction()

    # GRPO base config (SDPO overrides beta to 0)
    grpo_config = GRPOConfig(
        output_dir=str(RESULTS_DIR / "sdpo_checkpoints"),
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

    # SDPO config
    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,  # Generalized JSD
        distillation_topk=100,
        distillation_add_tail=True,
        is_clip=2.0,
        teacher_mode="ema",
        teacher_update_rate=0.05,
        success_reward_threshold=1.0,  # Only fully correct solutions as demos
        dont_reprompt_on_self_success=True,
        remove_thinking_from_demonstration=True,
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,  # paper default: feedback only when no solution
    )

    # CSV loggers
    train_csv = CSVLogger(
        str(RESULTS_DIR / "sdpo_train.csv"),
        fieldnames=[
            "step",
            "loss",
            "reward_mean",
            "sdpo_loss",
            "sdpo_kl_mean",
            "sdpo_teacher_coverage",
            "timestamp",
        ],
    )
    eval_csv = CSVLogger(
        str(RESULTS_DIR / "sdpo_eval.csv"),
        fieldnames=["step", "pass_at_1", "timestamp"],
    )

    # Training metrics callback
    class TrainMetricsCallback(TrainerCallback):
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
                        "timestamp": time.time(),
                    }
                )

    # Create SDPO trainer
    trainer = SDPOTrainer(
        model=model,
        args=grpo_config,
        sdpo_config=sdpo_config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn, format_fn],
        train_dataset=train_ds,
    )

    # Add callbacks
    trainer.add_callback(TrainMetricsCallback())
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
        adapter_dir = str(RESULTS_DIR / "sdpo_adapter")
        print(f"Saving adapter to {adapter_dir}...", flush=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    def post_train(state):
        elapsed = time.time() - start_time
        print(f"\n=== SDPO complete | {args.max_steps} steps in {elapsed / 60:.1f} min ===")
        train_csv.close()
        eval_csv.close()
        cleanup()

    trainer.add_callback(ForceExitCallback(max_steps=args.max_steps, save_fn=save_fn, post_train_fn=post_train))

    # Train (may not return due to TRL hang — ForceExitCallback handles exit)
    start_time = time.time()
    trainer.train()


if __name__ == "__main__":
    main()
