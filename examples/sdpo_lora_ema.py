#!/usr/bin/env python3
"""
SDPO Training with LoRA EMA Teacher Mode

Demonstrates the memory-efficient lora_ema teacher mode, which keeps student
and teacher as two LoRA adapters on a shared base model instead of creating
a full deepcopy. For a 7B model with 4-bit QLoRA this saves ~3-4 GB of VRAM.

How it works:
  - The student adapter ("default") trains normally via GRPO.
  - A frozen teacher adapter ("sdpo_teacher") provides distillation targets.
  - After each training step the teacher adapter is EMA-updated toward the student.
  - Base model weights are never duplicated — both adapters share them.

Usage:
    python examples/sdpo_lora_ema.py

Expected runtime: ~2 minutes on RTX 3080 (10GB)
"""

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from sdpo_rl import SDPOConfig, SDPOTrainer


def create_math_dataset(num_samples: int = 50) -> Dataset:
    """Multiplication problems — slightly harder than addition."""
    problems = []
    for i in range(num_samples):
        a = (i % 5) + 2  # 2..6
        b = (i * 3 % 7) + 2  # 2..8
        problems.append(
            {
                "prompt": [{"role": "user", "content": f"What is {a} * {b}? Answer with just the number."}],
                "correct_answer": a * b,
            }
        )
    return Dataset.from_list(problems)


class MathReward:
    """Scores multiplication answers and provides feedback for SDPO reprompting."""

    __name__ = "math_reward"

    def __init__(self):
        self.last_feedback: list[str] = []

    def __call__(self, prompts, completions, correct_answer, **kwargs) -> list[float]:
        scores = []
        self.last_feedback = []

        for completion, answer in zip(completions, correct_answer):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                parsed = int("".join(filter(str.isdigit, text.split()[0])))
                if parsed == answer:
                    scores.append(1.0)
                    self.last_feedback.append("")
                else:
                    scores.append(0.0)
                    self.last_feedback.append(f"Wrong. The answer is {answer}, not {parsed}.")
            except (ValueError, IndexError):
                scores.append(0.0)
                self.last_feedback.append(f"Could not parse a number from: {text[:50]}")

        return scores


def main():
    print("=== SDPO Training - LoRA EMA Teacher Mode ===\n")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Load base model in bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap with PEFT LoRA — this is what makes lora_ema mode possible.
    # The trainer will add a second "sdpo_teacher" adapter automatically.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")

    dataset = create_math_dataset(num_samples=50)
    print(f"Dataset: {len(dataset)} multiplication problems")
    print(f"Example: {dataset[0]['prompt'][0]['content']}\n")

    grpo_config = GRPOConfig(
        output_dir="./output/sdpo_lora_ema",
        num_generations=4,
        max_completion_length=32,
        temperature=0.7,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,  # higher LR typical for LoRA
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
    )

    # The key difference: teacher_mode="lora_ema"
    #
    # Instead of deepcopying the entire model for the teacher, the trainer
    # creates a second LoRA adapter ("sdpo_teacher") on the same base model.
    # EMA updates only touch the adapter weights (~8M params for this model).
    # For larger 7B QLoRA models this saves ~3-4 GB of VRAM.
    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        teacher_mode="lora_ema",
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
