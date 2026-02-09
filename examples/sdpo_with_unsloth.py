#!/usr/bin/env python3
"""
SDPO Training with Unsloth Integration

Demonstrates SDPO with Unsloth's 4-bit QLoRA for faster training and less memory.
Import order is critical: PatchFastRL MUST be called before importing SDPOTrainer.

Usage:
    pip install unsloth
    python examples/sdpo_with_unsloth.py

Expected runtime: ~3 minutes on RTX 3080 (10GB)
"""

# Step 1: Import and patch Unsloth BEFORE SDPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

# Step 2: Now safe to import SDPOTrainer (inherits patched GRPOTrainer)
from sdpo_trainer import SDPOConfig, SDPOTrainer
from trl import GRPOConfig
from datasets import Dataset


def create_reasoning_dataset(num_samples: int = 50) -> Dataset:
    """Simple even/odd reasoning problems with thinking tags."""
    problems = []
    for i in range(num_samples):
        num = (i % 10) + 1
        problems.append(
            {
                "prompt": [
                    {"role": "user", "content": f"Is {num} an even number? Think step by step, then answer YES or NO."}
                ],
                "is_even": num % 2 == 0,
            }
        )
    return Dataset.from_list(problems)


class ReasoningReward:
    """Reward function that checks reasoning answers and provides feedback."""

    def __init__(self):
        self.last_feedback: list[str] = []

    def __call__(self, prompts, completions, is_even, **kwargs) -> list[float]:
        scores = []
        self.last_feedback = []

        for completion, expected_even in zip(completions, is_even):
            expected = "YES" if expected_even else "NO"

            # Extract answer after thinking block if present
            answer_part = completion.split("</think>")[-1].strip() if "</think>" in completion else completion.strip()
            answer = "YES" if "YES" in answer_part.upper() else "NO" if "NO" in answer_part.upper() else "INVALID"

            if answer == expected:
                scores.append(1.0)
                self.last_feedback.append("")
            else:
                scores.append(0.0)
                self.last_feedback.append(f"Wrong. The answer is {expected}, not {answer}.")

        return scores


def main():
    print("=== SDPO Training with Unsloth ===\n")

    # Step 3: Load model with Unsloth (4-bit quantization)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    # Step 4: Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_reasoning_dataset(num_samples=50)
    print(f"Dataset: {len(dataset)} reasoning problems\n")

    grpo_config = GRPOConfig(
        output_dir="./output/sdpo_unsloth",
        num_generations=4,
        max_completion_length=128,
        temperature=0.7,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # Higher LR typical for LoRA
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
    )

    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        teacher_mode="ema",
        teacher_update_rate=0.05,
        remove_thinking_from_demonstration=True,  # Strip <think>...</think> from peer demos
        include_environment_feedback=True,
    )

    reward_fn = ReasoningReward()

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
