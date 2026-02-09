"""
MBPP dataset loading and prompt formatting for GRPO/SDPO benchmarking.

Loads the sanitized MBPP dataset from HuggingFace, formats prompts for
code generation, and produces train/eval splits ready for TRL.

Key design decisions:
- Train: MBPP train + validation (163 problems) — used for RL training.
- Eval: MBPP test (257 problems) — used for pass@1 evaluation.
- Prompts use a system message that instructs ```python code blocks.
- Test cases are NOT included in prompts (that would be cheating).
- Extra columns (test_list, test_imports, task_id) are preserved so
  GRPOTrainer can forward them to the reward function as kwargs.
"""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset


SYSTEM_PROMPT = (
    "You are a Python programming assistant. "
    "Write clean, correct Python code to solve the given task. "
    "Output your solution inside a ```python code block. "
    "Do not include test cases or example usage — just the function definition."
)


def format_prompt(description: str) -> list[dict[str, str]]:
    """
    Convert an MBPP problem description into a chat-formatted prompt.

    Returns a list of message dicts suitable for tokenizer.apply_chat_template().
    The system message sets the coding context, and the user message contains
    the problem description.

    Args:
        description: The natural language task description from MBPP.

    Returns:
        List of {"role": ..., "content": ...} message dicts.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]


def load_mbpp_splits() -> tuple[Dataset, Dataset]:
    """
    Load MBPP sanitized from HuggingFace and return (train, eval) splits.

    Train = MBPP train + validation (163 problems).
    Eval = MBPP test (257 problems).

    Both retain columns: prompt, test_list, test_imports, task_id.
    """
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    # Combine train + validation for the RL training set
    train = concatenate_datasets([ds["train"], ds["validation"]])
    eval_ds = ds["test"]

    return train, eval_ds


def make_train_dataset(raw: Dataset, seed: int = 42) -> Dataset:
    """
    Prepare the training dataset for TRL's GRPOTrainer.

    Converts MBPP problem descriptions into chat-formatted prompts
    and shuffles with a fixed seed for reproducibility.

    The resulting dataset has columns:
        - prompt: list[dict] (chat messages for apply_chat_template)
        - test_list: list[str] (assertion statements, forwarded to reward fn)
        - test_imports: list[str] (import statements, forwarded to reward fn)
        - task_id: int (for tracking which problem each sample came from)

    Args:
        raw: Raw MBPP dataset from load_mbpp_splits().
        seed: Random seed for shuffling.
    """

    def _format(example):
        example["prompt"] = format_prompt(example["prompt"])
        return example

    formatted = raw.map(_format)
    return formatted.shuffle(seed=seed)


def make_eval_dataset(raw: Dataset) -> Dataset:
    """
    Prepare the evaluation dataset.

    Same format as training but NOT shuffled (deterministic order for
    reproducible eval curves).

    Args:
        raw: Raw MBPP eval dataset from load_mbpp_splits().
    """

    def _format(example):
        example["prompt"] = format_prompt(example["prompt"])
        return example

    return raw.map(_format)
