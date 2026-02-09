#!/usr/bin/env python3
"""
Run 1: GRPO Baseline on MBPP.

Vanilla TRL GRPOTrainer with Unsloth + 4-bit QLoRA.
Reward function provides scalar scores only (no feedback).
Periodic evaluation on 257-problem test set.

Usage:
    uv run python benchmark/run_grpo.py [--max-steps N]
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

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# benchmark/ modules (resolved via sys.path)
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


def main():
    parser = argparse.ArgumentParser(description="GRPO baseline benchmark")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT, help="Training steps")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip final 257-problem eval")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== GRPO Baseline | max_steps={args.max_steps} ===")

    # Load model + data
    model, tokenizer = load_model_and_tokenizer()
    train_ds, eval_ds = load_datasets()
    reward_fn = MBPPRewardFunction()
    format_fn = FormatRewardFunction()

    # Training config
    grpo_config = GRPOConfig(
        output_dir=str(RESULTS_DIR / "grpo_checkpoints"),
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=100,  # effectively infinite — we stop at max_steps
        max_steps=args.max_steps,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_steps=999999,
        save_strategy="no",  # avoid post-training hang in Unsloth/TRL
        bf16=True,
        remove_unused_columns=False,
        seed=SEED,
        temperature=TEMPERATURE,
        reward_weights=[1.0, 0.3],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )

    # CSV loggers
    # Note: GRPO loss=0.0 is expected with beta=0 (the default). Monitor reward
    # and reward_std instead. Gradients still flow correctly despite zero loss.
    train_csv = CSVLogger(
        str(RESULTS_DIR / "grpo_train.csv"),
        fieldnames=["step", "loss", "reward_mean", "reward_std", "timestamp"],
    )
    eval_csv = CSVLogger(
        str(RESULTS_DIR / "grpo_eval.csv"),
        fieldnames=["step", "pass_at_1", "timestamp"],
    )

    # Training reward logger callback
    class TrainMetricsCallback(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs and state.global_step > 0:
                train_csv.log(
                    {
                        "step": state.global_step,
                        "loss": logs.get("loss", ""),
                        "reward_mean": logs.get("reward", ""),
                        "reward_std": logs.get("reward_std", ""),
                        "timestamp": time.time(),
                    }
                )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
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

    # save_fn: called one step early to guarantee adapter is saved before
    # Unsloth can silently os._exit on the final step.
    def save_fn():
        adapter_dir = str(RESULTS_DIR / "grpo_adapter")
        print(f"Saving adapter to {adapter_dir}...", flush=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    # post_train_fn: called at max_steps or on_train_end to close CSVs.
    def post_train(state):
        elapsed = time.time() - start_time
        print(f"\n=== GRPO complete | {args.max_steps} steps in {elapsed / 60:.1f} min ===")
        train_csv.close()
        eval_csv.close()
        cleanup()

    trainer.add_callback(ForceExitCallback(max_steps=args.max_steps, save_fn=save_fn, post_train_fn=post_train))

    # Train (may not return due to TRL hang — ForceExitCallback handles exit)
    start_time = time.time()
    trainer.train()


if __name__ == "__main__":
    main()
