#!/usr/bin/env python3
"""
Basic SDPO Training Example

Trains a small model on simple math problems using SDPO self-distillation.
Demonstrates the core loop: generate completions, reward, select peer demos,
reprompt teacher, distill.

Usage:
    python examples/basic_sdpo.py

Expected runtime: ~5 minutes on RTX 3080 (10GB)
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from sdpo_trainer import SDPOConfig, SDPOTrainer


def create_math_dataset(num_samples: int = 100) -> Dataset:
    """Simple addition problems."""
    problems = []
    for i in range(num_samples):
        a, b = i % 10 + 1, (i * 3) % 10 + 1
        problems.append(
            {
                "prompt": [{"role": "user", "content": f"What is {a} + {b}? Answer with just the number."}],
                "correct_answer": a + b,
            }
        )
    return Dataset.from_list(problems)


class MathReward:
    """
    Reward function that scores math answers and provides feedback for SDPO.

    TRL requires list[float] return type. To provide feedback strings for
    SDPO's teacher prompts, store them on self.last_feedback -- the trainer
    checks for this attribute after each reward call.
    """

    def __init__(self):
        self.last_feedback: list[str] = []

    def __call__(self, prompts, completions, correct_answer, **kwargs) -> list[float]:
        scores = []
        self.last_feedback = []

        for completion, answer in zip(completions, correct_answer):
            try:
                parsed = int("".join(filter(str.isdigit, completion.split()[0])))
                if parsed == answer:
                    scores.append(1.0)
                    self.last_feedback.append("")
                else:
                    scores.append(0.0)
                    self.last_feedback.append(f"Wrong. The answer is {answer}, not {parsed}.")
            except (ValueError, IndexError):
                scores.append(0.0)
                self.last_feedback.append(f"Could not parse a number from: {completion[:50]}")

        return scores


def main():
    print("=== SDPO Training - Basic Math Example ===\n")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_math_dataset(num_samples=50)
    print(f"Dataset: {len(dataset)} problems")
    print(f"Example: {dataset[0]['prompt'][0]['content']}\n")

    grpo_config = GRPOConfig(
        output_dir="./output/basic_sdpo",
        num_generations=4,
        max_completion_length=32,
        temperature=0.7,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
    )

    # SDPOConfig defaults match the paper's experiment scripts.
    # alpha=0.5 gives symmetric JSD, distillation_topk=100 captures >99% of mass.
    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        teacher_mode="ema",
        teacher_update_rate=0.05,
        include_environment_feedback=True,
    )

    reward_fn = MathReward()

    trainer = SDPOTrainer(
        model=model,
        args=grpo_config,
        sdpo_config=sdpo_config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
    )

    print("Starting training...\n")
    trainer.train()
    print("\nDone.")


if __name__ == "__main__":
    main()
