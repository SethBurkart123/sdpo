#!/usr/bin/env python3
"""
SDPO Training with Rich Feedback - Code Generation

Demonstrates SDPO's strength: learning from detailed error messages.
The reward function executes test cases and returns specific failure info,
which SDPO bakes into teacher prompts for failed attempts.

Usage:
    python examples/sdpo_rich_feedback.py

Expected runtime: ~5 minutes on RTX 3080 (10GB)
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from sdpo_rl import SDPOConfig, SDPOTrainer


def create_code_dataset(num_samples: int = 40) -> Dataset:
    """Simple Python function-writing tasks with test cases."""
    tasks = []

    for i in range(10):
        tasks.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list.",
                    }
                ],
                "test_cases": [([1, 2, 3], 6), ([10, -5, 3], 8), ([], 0)],
            }
        )

    for i in range(10):
        tasks.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Write a Python function `reverse_string(s)` that returns the string reversed.",
                    }
                ],
                "test_cases": [("hello", "olleh"), ("abc", "cba"), ("", "")],
            }
        )

    for i in range(10):
        tasks.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers.",
                    }
                ],
                "test_cases": [((1, 2, 3), 3), ((5, 2, 8), 8), ((-1, -5, -2), -1)],
            }
        )

    for i in range(10):
        tasks.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Write a Python function `count_vowels(s)` that returns the number of vowels in a string.",
                    }
                ],
                "test_cases": [("hello", 2), ("aeiou", 5), ("xyz", 0)],
            }
        )

    return Dataset.from_list(tasks)


class CodeReward:
    """
    Reward function that executes code against test cases.

    Returns list[float] for TRL compatibility. Stores detailed error
    messages on self.last_feedback for SDPO's teacher prompts.
    """

    def __init__(self):
        self.last_feedback: list[str] = []

    def __call__(self, prompts, completions, test_cases, **kwargs) -> list[float]:
        scores = []
        self.last_feedback = []

        for completion, tests in zip(completions, test_cases):
            score, feedback = self._evaluate(completion, tests)
            scores.append(score)
            self.last_feedback.append(feedback)

        return scores

    def _evaluate(self, completion: str, test_cases: list) -> tuple[float, str]:
        # Extract code from markdown blocks if present
        code = completion
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        code = code.strip()

        try:
            local_scope = {}
            exec(code, {}, local_scope)
        except SyntaxError as e:
            return 0.0, f"Syntax error: {e}"
        except Exception as e:
            return 0.0, f"Execution error: {type(e).__name__}: {e}"

        func = next((obj for obj in local_scope.values() if callable(obj)), None)
        if func is None:
            return 0.0, f"No function defined in: {code[:80]}"

        errors = []
        for test_input, expected in test_cases:
            try:
                result = func(*test_input) if isinstance(test_input, tuple) else func(test_input)
                if result != expected:
                    errors.append(f"input={test_input}: expected {expected}, got {result}")
            except Exception as e:
                errors.append(f"input={test_input}: {type(e).__name__}: {e}")

        if not errors:
            return 1.0, ""

        return 0.0, "Test failures:\n" + "\n".join(f"  - {e}" for e in errors)


def main():
    print("=== SDPO Training - Rich Feedback from Code Execution ===\n")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_code_dataset()
    print(f"Dataset: {len(dataset)} coding tasks\n")

    grpo_config = GRPOConfig(
        output_dir="./output/sdpo_rich_feedback",
        num_generations=6,  # More completions = more chances for a successful peer demo
        max_completion_length=256,
        temperature=0.8,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
    )

    # For rich feedback tasks, the paper uses alpha=1.0 (reverse KL) and topk=20.
    # We use alpha=0.5 (JSD) here for a gentler introduction, but both work.
    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        teacher_mode="ema",
        teacher_update_rate=0.05,
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,  # Only use feedback when no peer demo exists
    )

    reward_fn = CodeReward()

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
