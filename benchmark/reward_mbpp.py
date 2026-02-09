"""
MBPP reward function with sandboxed code execution and rich feedback.

Executes LLM-generated Python code against MBPP test assertions in a
subprocess with a timeout. Returns scalar rewards for TRL and stores
detailed feedback strings for SDPO's self-distillation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of executing code against a set of test assertions."""

    passed: int
    total: int
    error: str | None

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total and self.total > 0


def extract_code_block(completion: str) -> str:
    """
    Extract Python code from an LLM completion.

    Looks for fenced code blocks (```python ... ``` or ``` ... ```).
    If none found, returns the raw text (the model may have just written code).
    """
    if not completion:
        return ""

    # Try ```python ... ``` first, then ``` ... ```
    # The \n? makes the newline after the fence marker optional — small models
    # sometimes emit ```pythondef foo(): ... without a line break.
    patterns = [
        r"```python\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            return match.group(1).strip()

    return completion.strip()


def execute_code_with_tests(
    code: str,
    tests: list[str],
    imports: list[str] | None = None,
    timeout: int = 5,
) -> ExecutionResult:
    """
    Execute code + test assertions in a sandboxed subprocess.

    Runs each test individually to count how many pass and capture
    the first failure message for rich feedback.

    Args:
        code: The Python source code to test.
        imports: Optional list of import statements to prepend.
        tests: List of assert statements to run after the code.
        timeout: Maximum seconds per execution.

    Returns:
        ExecutionResult with pass count, total, and error details.
    """
    total = len(tests)
    if not code and not tests:
        return ExecutionResult(passed=0, total=0, error=None)

    import_block = "\n".join(imports) + "\n" if imports else ""

    # Run all tests in one shot first — fast path
    full_script = import_block + code + "\n" + "\n".join(tests)
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return ExecutionResult(passed=total, total=total, error=None)
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=0,
            total=total,
            error=f"Execution timed out after {timeout} seconds (possible infinite loop)",
        )

    # Something failed — run tests individually to count passes
    # and capture the specific error
    first_error = result.stderr.strip() if result.stderr else "Unknown error"
    passed = 0

    for test in tests:
        single_script = import_block + code + "\n" + test
        try:
            single_result = subprocess.run(
                [sys.executable, "-c", single_script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if single_result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass  # count as fail

    return ExecutionResult(passed=passed, total=total, error=first_error)


def _format_feedback(result: ExecutionResult) -> str:
    """
    Build a human-readable feedback string from an execution result.

    This is the rich feedback that SDPO uses for self-distillation —
    it gets baked into the teacher prompt so the model can learn from
    its mistakes in-context.
    """
    if result.all_passed:
        return f"All {result.total} tests passed."

    parts = []
    if result.passed > 0:
        parts.append(f"{result.passed}/{result.total} tests passed.")
    else:
        parts.append(f"All {result.total} tests failed.")

    if result.error:
        # Trim traceback to the last meaningful line for conciseness
        lines = result.error.strip().splitlines()
        # Keep the last 3 lines (usually the error type + message)
        relevant = lines[-3:] if len(lines) > 3 else lines
        parts.append("Error: " + "\n".join(relevant))

    return " ".join(parts)


def _unwrap_completion(completion) -> str:
    """Extract text from a completion that may be a string or chat message list."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion or ""


def score_code_format(completion: str) -> float:
    """
    Score a completion based on code formatting quality.

    Creates learning signal even when no tests pass — essential for breaking
    the GRPO cold-start problem where all 8 generations in a group fail,
    yielding zero advantage and zero gradient.

    Scoring:
        1.0 — syntactically valid Python inside a ```python block
        0.5 — syntactically valid Python (no code fence)
        0.2 — has a code fence but syntax error
        0.1 — contains 'def ' (trying to write a function)
        0.0 — empty or no code-like content
    """
    if not completion or not completion.strip():
        return 0.0

    code = extract_code_block(completion)
    if not code:
        return 0.0

    has_fence = "```" in completion
    has_def = "def " in code

    # Check if the extracted code is syntactically valid Python
    try:
        compile(code, "<completion>", "exec")
        is_valid = True
    except SyntaxError:
        is_valid = False

    if has_fence and is_valid:
        return 1.0
    if is_valid:
        return 0.5
    if has_fence:
        return 0.2
    if has_def:
        return 0.1
    return 0.0


class FormatRewardFunction:
    """
    TRL-compatible reward function that scores code formatting quality.

    Designed to be used alongside MBPPRewardFunction as a second reward
    function. Creates variance within generation groups even when the model
    can't solve any problems yet, breaking the GRPO cold-start deadlock.

    Usage with GRPOTrainer:
        reward_funcs=[MBPPRewardFunction(), FormatRewardFunction()]
        GRPOConfig(reward_weights=[1.0, 0.3])  # correctness dominates
    """

    def __init__(self):
        self.__name__ = "format_reward"

    def __call__(self, prompts=None, completions=None, **kwargs) -> list[float]:
        if completions is None:
            return []
        return [score_code_format(_unwrap_completion(c)) for c in completions]


class MBPPRewardFunction:
    """
    TRL-compatible reward function for MBPP code generation.

    Call signature matches what GRPOTrainer expects:
        reward_fn(prompts, completions, **kwargs) -> list[float]

    Test cases are passed via kwargs — GRPOTrainer forwards extra dataset
    columns by name. The MBPP dataset has 'test_list' and 'test_imports'
    columns, so those are the kwarg names.

    After each call, `self.last_feedback` contains the rich feedback
    strings that SDPOTrainer reads for self-distillation.
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.last_feedback: list[str] = []
        self.__name__ = "mbpp_reward"

    def __call__(
        self,
        prompts=None,
        completions=None,
        # Column names match the HuggingFace dataset exactly —
        # GRPOTrainer passes extra columns as **kwargs by name.
        test_list: list[list[str]] | None = None,
        test_imports: list[list[str]] | None = None,
        # Accept but ignore any other columns TRL might forward
        **kwargs,
    ) -> list[float]:
        """
        Score completions by executing them against test assertions.

        Args:
            prompts: Prompt strings or chat messages (unused, but TRL passes them).
            completions: LLM-generated code completions. Either list[str] or
                list[list[dict]] when the dataset is conversational (TRL wraps
                each completion as [{"role": "assistant", "content": "..."}]).
            test_list: Per-sample list of assert statements (from dataset column).
            test_imports: Per-sample list of import statements (from dataset column).

        Returns:
            list[float]: Reward for each completion.
                1.0 = all tests pass
                0.0 = no tests pass (or error)
                Proportional for partial passes.
        """
        if completions is None:
            raise ValueError("completions must be provided")
        if test_list is None:
            raise ValueError("test_list must be provided via kwargs. Make sure your dataset has a 'test_list' column.")
        if test_imports is None:
            test_imports = [[] for _ in completions]

        rewards = []
        feedback = []

        for completion, tests, imports in zip(completions, test_list, test_imports):
            completion = _unwrap_completion(completion)
            code = extract_code_block(completion)
            result = execute_code_with_tests(code, tests, imports=imports, timeout=self.timeout)

            if result.all_passed:
                rewards.append(1.0)
            elif result.total == 0:
                rewards.append(0.0)
            else:
                rewards.append(result.passed / result.total)

            feedback.append(_format_feedback(result))

        self.last_feedback = feedback
        return rewards
