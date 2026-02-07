#!/usr/bin/env python3
"""
Basic SDPO Training Example

This script demonstrates minimal SDPO training on a simple math task.
It uses a small model (Qwen-0.5B) that can run on consumer GPUs.

Usage:
    python examples/basic_sdpo.py

Expected runtime: ~5 minutes on RTX 3080 (10GB)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig

from sdpo_trainer import SDPOTrainer, SDPOConfig


def create_math_dataset(num_samples: int = 100) -> Dataset:
    """Create a simple math problem dataset for SDPO training."""
    problems = []
    for i in range(num_samples):
        a, b = i % 10 + 1, (i * 3) % 10 + 1
        problems.append(
            {
                "uid": f"math_{i}",  # Unique identifier for grouping rollouts
                "prompt": f"What is {a} + {b}? Answer with just the number.",
                "correct_answer": a + b,
            }
        )
    return Dataset.from_list(problems)


def math_reward_function(prompts: list[str], completions: list[str], **kwargs) -> list[dict]:
    """
    Reward function that provides both scores and feedback.

    Returns:
        list[dict]: Each dict has {"score": float, "feedback": str}
                   - score: 1.0 for correct, 0.0 for wrong
                   - feedback: explanation string for the teacher
    """
    # Extract dataset to get correct answers
    dataset = kwargs.get("dataset", [])

    results = []
    for prompt, completion, sample in zip(prompts, completions, dataset):
        try:
            # Extract the number from completion
            answer = int("".join(filter(str.isdigit, completion.split()[0])))
            correct = sample["correct_answer"]

            if answer == correct:
                score = 1.0
                feedback = f"Correct! {answer} is the right answer."
            else:
                score = 0.0
                feedback = f"Wrong. The answer is {correct}, not {answer}."
        except (ValueError, IndexError):
            # Couldn't parse an answer
            score = 0.0
            feedback = f"Invalid answer format. Expected a number, got: {completion[:50]}"

        results.append({"score": score, "feedback": feedback})

    return results


def main():
    print("=" * 80)
    print("SDPO Training - Basic Math Example")
    print("=" * 80)

    # 1. Load a small model (fits on most GPUs)
    print("\n1. Loading Qwen-0.5B model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✓ Model loaded: {model_name}")
    print(f"   ✓ Parameters: {model.num_parameters() / 1e6:.1f}M")

    # 2. Create dataset
    print("\n2. Creating math problem dataset...")
    dataset = create_math_dataset(num_samples=50)
    print(f"   ✓ Dataset: {len(dataset)} problems")
    print(f"   ✓ Example: {dataset[0]['prompt']}")

    # 3. Configure GRPO (base trainer settings)
    print("\n3. Configuring training...")
    grpo_config = GRPOConfig(
        output_dir="./output/basic_sdpo",
        # Generation settings
        num_generations=4,  # 4 completions per prompt
        max_completion_length=32,  # Short completions for math
        temperature=0.7,
        # Training settings
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 4 prompts per batch
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        # Logging
        logging_steps=1,
        save_steps=100,
        # Performance
        bf16=True,
        remove_unused_columns=False,
    )

    # 4. Configure SDPO (self-distillation settings)
    sdpo_config = SDPOConfig(
        enabled=True,  # Enable SDPO (replaces GRPO loss)
        # Core SDPO parameters
        alpha=0.5,  # Generalized JSD balance (0.5 = symmetric)
        distillation_topk=100,  # Top-100 logits for efficiency
        teacher_mode="ema",  # EMA teacher (no third model)
        ema_tau=0.05,  # EMA update rate: θ_t = 0.95*θ_t + 0.05*θ_s
        ema_update_every=1,  # Update teacher every batch
        # Importance sampling
        is_coef=0.0,  # Disable IS correction for on-policy setting
        is_clip=2.0,  # Clip IS weights (not used when is_coef=0)
        # Reprompting (uses successful peer rollouts as teacher demos)
        reprompt_teacher=True,
        thinking_tag="<think>",  # Strip <think>...</think> from demos
    )

    print(f"   ✓ GRPO config: {grpo_config.num_generations} gens, batch={grpo_config.per_device_train_batch_size}")
    print(
        f"   ✓ SDPO config: alpha={sdpo_config.alpha}, topk={sdpo_config.distillation_topk}, EMA_tau={sdpo_config.ema_tau}"
    )

    # 5. Create trainer
    print("\n4. Initializing SDPOTrainer...")
    trainer = SDPOTrainer(
        model=model,
        args=grpo_config,
        sdpo_config=sdpo_config,
        processing_class=tokenizer,
        reward_funcs=[math_reward_function],
        train_dataset=dataset,
    )

    print("   ✓ SDPOTrainer initialized")
    print("   ✓ EMA teacher: repurposed ref_model (2 models total)")

    # 6. Train!
    print("\n5. Starting training...")
    print("   Note: This trains for 1 epoch (~50 problems)")
    print("   Expected: Loss should decrease as model learns math")
    print()

    trainer.train()

    print("\n" + "=" * 80)
    print("Training complete! ✓")
    print("=" * 80)
    print("\nModel saved to:", grpo_config.output_dir)
    print("\nWhat happened during training:")
    print("  1. Generated 4 completions per math problem")
    print("  2. Rewarded correct answers (1.0), penalized wrong answers (0.0)")
    print("  3. Used successful peer answers as teacher demonstrations")
    print("  4. Computed self-distillation loss (SDPO replaces GRPO loss)")
    print("  5. Updated EMA teacher every batch (tau=0.05)")
    print("\nNext steps:")
    print("  - Try sdpo_with_unsloth.py for faster training with 4-bit quantization")
    print("  - Try sdpo_rich_feedback.py for tasks with detailed error messages")


if __name__ == "__main__":
    main()
