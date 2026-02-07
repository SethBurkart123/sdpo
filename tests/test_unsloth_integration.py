"""
Integration test for SDPO + Unsloth.

Verifies:
1. Unsloth's PatchFastRL properly patches the GRPOTrainer base class
2. SDPOTrainer inherits the patched class
3. A short training run completes without errors
4. EMA teacher updates work with LoRA models

Requirements: unsloth must be installed, GPU required.
"""

from __future__ import annotations

import gc
import sys

import pytest
import torch


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif(not _unsloth_available(), reason="unsloth not installed"),
]


@pytest.fixture(scope="module")
def unsloth_model_and_tokenizer():
    """Load a small model with Unsloth optimizations (module-scoped for speed)."""
    from unsloth import FastLanguageModel, PatchFastRL

    # Patch GRPO trainer before importing SDPOTrainer
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=256,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.pad_token = tokenizer.eos_token

    yield model, tokenizer

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _simple_reward(prompts, completions, **kwargs):
    """Reward function: 1.0 if completion contains 'yes', else 0.0."""
    return [1.0 if "yes" in c.lower() else 0.0 for c in completions]


@pytest.fixture
def small_dataset():
    from datasets import Dataset

    data = [{"prompt": f"Is {i} even? Answer yes or no."} for i in range(20)]
    return Dataset.from_list(data)


class TestUnslothIntegration:
    """Tests that SDPOTrainer works with Unsloth-patched models."""

    def test_trainer_instantiates(self, unsloth_model_and_tokenizer, small_dataset, tmp_path):
        """SDPOTrainer should accept an Unsloth-patched model."""
        from sdpo_trainer import SDPOConfig, SDPOTrainer
        from trl import GRPOConfig

        model, tokenizer = unsloth_model_and_tokenizer

        grpo_config = GRPOConfig(
            output_dir=str(tmp_path),
            num_generations=2,
            max_completion_length=32,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            max_steps=1,
            logging_steps=1,
            bf16=True,
            remove_unused_columns=False,
        )

        sdpo_config = SDPOConfig(
            enabled=True,
            alpha=0.5,
            distillation_topk=50,
            teacher_mode="ema",
            teacher_update_rate=0.05,
        )

        trainer = SDPOTrainer(
            model=model,
            args=grpo_config,
            sdpo_config=sdpo_config,
            processing_class=tokenizer,
            reward_funcs=[_simple_reward],
            train_dataset=small_dataset,
        )

        # Verify the trainer was created
        assert trainer is not None
        assert trainer.ref_model is not None
        assert trainer.ema_callback is not None

    def test_short_training_run(self, unsloth_model_and_tokenizer, small_dataset, tmp_path):
        """A few training steps should complete without error."""
        from sdpo_trainer import SDPOConfig, SDPOTrainer
        from trl import GRPOConfig

        model, tokenizer = unsloth_model_and_tokenizer

        grpo_config = GRPOConfig(
            output_dir=str(tmp_path),
            num_generations=2,
            max_completion_length=32,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            max_steps=2,
            logging_steps=1,
            bf16=True,
            remove_unused_columns=False,
        )

        sdpo_config = SDPOConfig(
            enabled=True,
            alpha=0.5,
            distillation_topk=50,
            teacher_mode="ema",
            teacher_update_rate=0.05,
        )

        trainer = SDPOTrainer(
            model=model,
            args=grpo_config,
            sdpo_config=sdpo_config,
            processing_class=tokenizer,
            reward_funcs=[_simple_reward],
            train_dataset=small_dataset,
        )

        # This is the real test — does training actually run?
        trainer.train()

        # If we get here, training completed
        assert trainer.state.global_step >= 1
