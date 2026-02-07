#!/usr/bin/env python3
"""
SDPO Training with Rich Feedback - Code Generation Example

This script demonstrates SDPO's strength: learning from detailed error messages.
The reward function runs test cases and provides specific failure messages,
which SDPO uses as teacher demonstrations for failed attempts.

Usage:
    python examples/sdpo_rich_feedback.py

Expected runtime: ~5 minutes on RTX 3080 (10GB)

This is SDPO's killer feature: using environment feedback (test failures,
compiler errors, etc.) to improve the policy via self-distillation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig

from sdpo_trainer import SDPOTrainer, SDPOConfig


def create_code_dataset(num_samples: int = 40) -> Dataset:
    """Create simple Python function writing tasks."""
    tasks = []

    # Task 1: List operations
    for i in range(10):
        tasks.append(
            {
                "uid": f"list_sum_{i}",
                "prompt": "Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list.",
                "test_cases": [
                    ([1, 2, 3], 6),
                    ([10, -5, 3], 8),
                    ([0, 0, 0], 0),
                    ([], 0),
                ],
                "solution": "def sum_list(lst):\n    return sum(lst)",
            }
        )

    # Task 2: String operations
    for i in range(10):
        tasks.append(
            {
                "uid": f"reverse_string_{i}",
                "prompt": "Write a Python function `reverse_string(s)` that returns the string reversed.",
                "test_cases": [
                    ("hello", "olleh"),
                    ("abc", "cba"),
                    ("", ""),
                    ("a", "a"),
                ],
                "solution": "def reverse_string(s):\n    return s[::-1]",
            }
        )

    # Task 3: Conditional logic
    for i in range(10):
        tasks.append(
            {
                "uid": f"max_of_three_{i}",
                "prompt": "Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers.",
                "test_cases": [
                    ((1, 2, 3), 3),
                    ((5, 2, 8), 8),
                    ((7, 7, 7), 7),
                    ((-1, -5, -2), -1),
                ],
                "solution": "def max_of_three(a, b, c):\n    return max(a, b, c)",
            }
        )

    # Task 4: Loop logic
    for i in range(10):
        tasks.append(
            {
                "uid": f"count_vowels_{i}",
                "prompt": "Write a Python function `count_vowels(s)` that returns the number of vowels (a,e,i,o,u) in a string.",
                "test_cases": [
                    ("hello", 2),
                    ("aeiou", 5),
                    ("xyz", 0),
                    ("", 0),
                ],
                "solution": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
            }
        )

    return Dataset.from_list(tasks)


def code_reward_function(prompts: list[str], completions: list[str], **kwargs) -> list[dict]:
    """
    Reward function that executes test cases and provides detailed feedback.

    This is where SDPO shines:
    - Successful runs get score=1.0, generic feedback
    - Failed runs get score=0.0, SPECIFIC error messages
    - Teacher learns from successful peers and incorporates error feedback

    Returns:
        list[dict]: Each with {"score": float, "feedback": str}
    """
    dataset = kwargs.get("dataset", [])

    results = []
    for prompt, completion, sample in zip(prompts, completions, dataset):
        test_cases = sample["test_cases"]

        # Extract the function definition from completion
        try:
            # Remove markdown code blocks if present
            code = completion
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            code = code.strip()

            # Execute the code to define the function
            local_scope = {}
            exec(code, {}, local_scope)

            # Find the function (first callable in local scope)
            func = None
            for name, obj in local_scope.items():
                if callable(obj):
                    func = obj
                    break

            if func is None:
                results.append(
                    {
                        "score": 0.0,
                        "feedback": f"No function found in code. Expected a function definition. Got:\n{code[:100]}",
                    }
                )
                continue

            # Run test cases
            all_passed = True
            error_messages = []

            for test_input, expected_output in test_cases:
                try:
                    # Handle different input formats
                    if isinstance(test_input, tuple):
                        actual_output = func(*test_input)
                    else:
                        actual_output = func(test_input)

                    if actual_output != expected_output:
                        all_passed = False
                        error_messages.append(
                            f"Test failed: input={test_input}, expected={expected_output}, got={actual_output}"
                        )
                except Exception as e:
                    all_passed = False
                    error_messages.append(f"Test crashed: input={test_input}, error={type(e).__name__}: {str(e)}")

            if all_passed:
                results.append(
                    {
                        "score": 1.0,
                        "feedback": f"All tests passed! Your function works correctly.",
                    }
                )
            else:
                # Rich feedback: specific error messages
                feedback = "Tests failed:\n" + "\n".join(f"  - {msg}" for msg in error_messages)
                results.append(
                    {
                        "score": 0.0,
                        "feedback": feedback,
                    }
                )

        except SyntaxError as e:
            results.append(
                {
                    "score": 0.0,
                    "feedback": f"Syntax error in code: {str(e)}\nCode:\n{code[:100]}",
                }
            )
        except Exception as e:
            results.append(
                {
                    "score": 0.0,
                    "feedback": f"Error executing code: {type(e).__name__}: {str(e)}\nCode:\n{code[:100]}",
                }
            )

    return results


def main():
    print("=" * 80)
    print("SDPO Training - Rich Feedback from Code Execution")
    print("=" * 80)
    print("\nThis example demonstrates SDPO's key strength:")
    print("  - Reward function executes test cases")
    print("  - Failed tests return SPECIFIC error messages")
    print("  - Teacher learns from successful peers AND error feedback")
    print("  - Model improves by distilling from corrected attempts")
    print()

    # 1. Load model
    print("1. Loading Qwen-0.5B model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✓ Model loaded: {model_name}")

    # 2. Create dataset
    print("\n2. Creating code generation dataset...")
    dataset = create_code_dataset(num_samples=40)
    print(f"   ✓ Dataset: {len(dataset)} coding tasks")
    print(f"   ✓ Task types: list ops, string ops, conditionals, loops")
    print(f"   ✓ Example: {dataset[0]['prompt']}")

    # 3. Configure GRPO
    print("\n3. Configuring training...")
    grpo_config = GRPOConfig(
        output_dir="./output/sdpo_rich_feedback",
        # Generation
        num_generations=6,  # More completions = more chances for success
        max_completion_length=256,  # Room for code + explanation
        temperature=0.8,  # Higher temp for code diversity
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        # Optimization
        bf16=True,
        logging_steps=1,
        save_steps=100,
        remove_unused_columns=False,
    )

    # 4. Configure SDPO with feedback
    sdpo_config = SDPOConfig(
        enabled=True,
        alpha=0.5,
        distillation_topk=100,
        teacher_mode="ema",
        ema_tau=0.05,
        ema_update_every=1,
        is_coef=0.0,
        reprompt_teacher=True,  # CRITICAL: Use peer demos + feedback
        thinking_tag="<think>",
        # Template uses {feedback} placeholder - gets populated with test failures
        teacher_prompt_template=None,  # Uses default template with feedback support
    )

    print(f"   ✓ GRPO: {grpo_config.num_generations} completions per task")
    print(f"   ✓ SDPO: reprompting with feedback enabled")
    print(f"   ✓ Teacher: EMA (tau={sdpo_config.ema_tau})")

    # 5. Create trainer
    print("\n4. Initializing SDPOTrainer...")
    trainer = SDPOTrainer(
        model=model,
        args=grpo_config,
        sdpo_config=sdpo_config,
        processing_class=tokenizer,
        reward_funcs=[code_reward_function],  # Returns rich feedback
        train_dataset=dataset,
    )

    print("   ✓ SDPOTrainer initialized")
    print("   ✓ Reward function will execute test cases")

    # 6. Train!
    print("\n5. Starting training...")
    print("   Note: Watch for test failure feedback in teacher prompts")
    print("   Note: Model learns from both successful peers AND error messages")
    print()

    trainer.train()

    print("\n" + "=" * 80)
    print("Training complete! ✓")
    print("=" * 80)
    print("\nModel saved to:", grpo_config.output_dir)
    print("\nWhat happened during training:")
    print("  1. Generated 6 code solutions per task")
    print("  2. Executed test cases for each solution")
    print("  3. Successful solutions scored 1.0")
    print("  4. Failed solutions scored 0.0 with SPECIFIC error messages:")
    print("     - 'Test failed: input=..., expected=..., got=...'")
    print("     - 'Test crashed: input=..., error=...'")
    print("     - 'Syntax error in code: ...'")
    print("  5. Teacher prompts included:")
    print("     - Original task")
    print("     - Successful peer solution (if any)")
    print("     - Error feedback from failed attempts")
    print("  6. Model learned via self-distillation from teacher")
    print("\nWhy SDPO is perfect for code tasks:")
    print("  ✓ Rich feedback (test failures) guides improvement")
    print("  ✓ Successful peers provide working examples")
    print("  ✓ No need for human-written error corrections")
    print("  ✓ Scales to any task with executable feedback")
    print("\nNext steps:")
    print("  - Try more complex tasks (algorithms, data structures)")
    print("  - Use real test frameworks (pytest, unittest)")
    print("  - Combine with Unsloth for larger models")


if __name__ == "__main__":
    main()
