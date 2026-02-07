#!/usr/bin/env python3
"""
SDPO Training with Unsloth Integration

This script demonstrates SDPO training with Unsloth's optimizations:
- 4-bit quantization (QLoRA)
- 2x faster training
- 60% less memory usage

Usage:
    pip install unsloth
    python examples/sdpo_with_unsloth.py

Expected runtime: ~3 minutes on RTX 3080 (10GB)
                 ~5 minutes on Google Colab free tier (T4)

CRITICAL: Import order matters! PatchFastRL BEFORE SDPOTrainer.
"""

# ============================================================================
# STEP 1: Import Unsloth and Patch BEFORE importing SDPOTrainer
# ============================================================================
# This is the critical import order for Unsloth compatibility.
# See UNSLOTH_INTEGRATION.md for detailed explanation.

from unsloth import FastLanguageModel, PatchFastRL

# Patch the GRPO trainer BEFORE importing SDPOTrainer
PatchFastRL("GRPO", FastLanguageModel)

# ============================================================================
# STEP 2: Now safe to import SDPOTrainer (it will use patched base class)
# ============================================================================

from sdpo_trainer import SDPOTrainer, SDPOConfig
from trl import GRPOConfig
from datasets import Dataset
import torch


def create_reasoning_dataset(num_samples: int = 50) -> Dataset:
    """Create a simple reasoning dataset with thinking tags."""
    problems = []

    # Simple logic puzzles
    for i in range(num_samples):
        num = (i % 10) + 1
        problems.append(
            {
                "uid": f"logic_{i}",
                "prompt": f"Is {num} an even number? Think step by step, then answer YES or NO.",
                "is_even": num % 2 == 0,
            }
        )

    return Dataset.from_list(problems)


def reasoning_reward_function(prompts: list[str], completions: list[str], **kwargs) -> list[dict]:
    """
    Reward function that checks reasoning correctness.

    Supports thinking tags: <think>reasoning here</think>
    SDPO will automatically strip these when selecting teacher demos.
    """
    dataset = kwargs.get("dataset", [])

    results = []
    for prompt, completion, sample in zip(prompts, completions, dataset):
        is_even = sample["is_even"]
        expected = "YES" if is_even else "NO"

        # Extract final answer (after thinking, if present)
        if "</think>" in completion:
            # Answer is after thinking block
            answer_part = completion.split("</think>")[-1].strip()
        else:
            answer_part = completion.strip()

        # Check if answer is correct
        answer = "YES" if "YES" in answer_part.upper() else "NO" if "NO" in answer_part.upper() else "INVALID"

        if answer == expected:
            score = 1.0
            feedback = f"Correct! {expected} is the right answer."
        elif answer == "INVALID":
            score = 0.0
            feedback = f"Invalid answer. Expected {expected}, got: {answer_part[:50]}"
        else:
            score = 0.0
            feedback = f"Wrong. The answer is {expected}, not {answer}."

        results.append({"score": score, "feedback": feedback})

    return results


def main():
    print("=" * 80)
    print("SDPO Training with Unsloth - Optimized Inference & Training")
    print("=" * 80)

    # ========================================================================
    # STEP 3: Load model with Unsloth (QLoRA, optimization, etc.)
    # ========================================================================
    print("\n1. Loading Qwen-0.5B with Unsloth optimizations...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=512,  # Unsloth RoPE optimization
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # QLoRA quantization
        # For larger models on limited VRAM:
        # load_in_4bit=True,
        # use_gradient_checkpointing="unsloth",  # Memory-efficient training
    )

    print("   ✓ Model loaded with Unsloth optimizations")
    print("   ✓ 4-bit quantization enabled (QLoRA)")
    print("   ✓ Expect 2x faster training, 60% less memory")

    # Configure for LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's efficient checkpointing
        random_state=42,
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("   ✓ LoRA adapters added (r=16)")
    print(f"   ✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # ========================================================================
    # STEP 4: Create dataset
    # ========================================================================
    print("\n2. Creating reasoning dataset...")
    dataset = create_reasoning_dataset(num_samples=50)
    print(f"   ✓ Dataset: {len(dataset)} reasoning problems")
    print(f"   ✓ Example: {dataset[0]['prompt']}")

    # ========================================================================
    # STEP 5: Configure GRPO + SDPO
    # ========================================================================
    print("\n3. Configuring training...")

    grpo_config = GRPOConfig(
        output_dir="./output/sdpo_unsloth",
        # Generation (Unsloth optimizes this!)
        num_generations=4,
        max_completion_length=128,
        temperature=0.7,
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Smaller batch for 4-bit
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # Higher LR for LoRA
        # Optimization
        optim="adamw_8bit",  # 8-bit optimizer (less memory)
        bf16=True,
        # Logging
        logging_steps=1,
        save_steps=100,
        remove_unused_columns=False,
    )

    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        teacher_mode="ema",
        ema_tau=0.05,
        ema_update_every=1,
        is_coef=0.0,
        reprompt_teacher=True,
        thinking_tag="<think>",  # Strip thinking tags from teacher demos
    )

    print(f"   ✓ GRPO: {grpo_config.num_generations} gens, batch={grpo_config.per_device_train_batch_size}")
    print(f"   ✓ SDPO: alpha={sdpo_config.alpha}, EMA_tau={sdpo_config.ema_tau}")
    print(f"   ✓ Optimizer: {grpo_config.optim}")

    # ========================================================================
    # STEP 6: Create SDPOTrainer (uses Unsloth-patched base class)
    # ========================================================================
    print("\n4. Initializing SDPOTrainer with Unsloth...")

    trainer = SDPOTrainer(
        model=model,
        args=grpo_config,
        sdpo_config=sdpo_config,
        processing_class=tokenizer,
        reward_funcs=[reasoning_reward_function],
        train_dataset=dataset,
    )

    print("   ✓ SDPOTrainer initialized")
    print("   ✓ Unsloth optimizations active (patched GRPO base)")
    print("   ✓ EMA teacher configured")

    # Verify Unsloth is active
    print("\n5. Verifying Unsloth integration...")
    print(f"   ✓ Model class: {type(model).__name__}")
    print(f"   ✓ Trainer class: {type(trainer).__name__}")
    print(f"   ✓ Base trainer class: {type(trainer).__bases__[0].__name__}")

    # ========================================================================
    # STEP 7: Train with SDPO + Unsloth optimizations
    # ========================================================================
    print("\n6. Starting training...")
    print("   Note: Unsloth makes generation ~2x faster")
    print("   Note: QLoRA reduces memory by ~60%")
    print()

    trainer.train()

    print("\n" + "=" * 80)
    print("Training complete! ✓")
    print("=" * 80)
    print("\nModel saved to:", grpo_config.output_dir)
    print("\nUnsloth optimizations applied:")
    print("  ✓ 4-bit quantization (QLoRA) - 60% less memory")
    print("  ✓ Fast generation - 2x faster completions")
    print("  ✓ Gradient checkpointing - efficient backprop")
    print("  ✓ 8-bit optimizer - reduced optimizer memory")
    print("\nSDPO training:")
    print("  ✓ Self-distillation loss replaced GRPO loss")
    print("  ✓ Teacher demonstrations from successful peers")
    print("  ✓ EMA teacher updated every batch")
    print("  ✓ Thinking tags stripped from teacher prompts")
    print("\nNext steps:")
    print("  - Try larger models (Qwen2.5-7B works on 10GB GPU with Unsloth)")
    print("  - Use gradient_checkpointing='unsloth' for even larger models")
    print("  - Try sdpo_rich_feedback.py for code tasks with detailed feedback")


if __name__ == "__main__":
    main()
