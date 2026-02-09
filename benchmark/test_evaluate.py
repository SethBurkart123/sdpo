"""Tests for the MBPP evaluation module."""

from __future__ import annotations

import pytest

from evaluate import (
    evaluate_completion,
    evaluate_batch,
    compute_pass_at_1,
)
from reward_mbpp import ExecutionResult


# ---------------------------------------------------------------------------
# evaluate_completion
# ---------------------------------------------------------------------------


class TestEvaluateCompletion:
    """Evaluate a single LLM completion against test cases."""

    def test_correct_solution(self):
        completion = "def add(a, b): return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = evaluate_completion(completion, tests)
        assert result.all_passed

    def test_wrong_solution(self):
        completion = "def add(a, b): return a - b"
        tests = ["assert add(1, 2) == 3"]
        result = evaluate_completion(completion, tests)
        assert not result.all_passed

    def test_with_imports(self):
        completion = "import math\ndef area(r): return math.pi * r * r"
        tests = ["assert abs(area(1) - 3.14159) < 0.001"]
        result = evaluate_completion(completion, tests, imports=["import math"])
        assert result.all_passed

    def test_syntax_error(self):
        completion = "def add(a, b) return a + b"
        tests = ["assert add(1, 2) == 3"]
        result = evaluate_completion(completion, tests)
        assert not result.all_passed
        assert "SyntaxError" in result.error

    def test_extracts_code_block(self):
        """Should handle ```python fenced code blocks."""
        completion = "Here:\n```python\ndef add(a, b): return a + b\n```"
        tests = ["assert add(1, 2) == 3"]
        result = evaluate_completion(completion, tests)
        assert result.all_passed


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------


class TestEvaluateBatch:
    """Evaluate a batch of completions."""

    def test_batch_results(self):
        completions = [
            "def add(a, b): return a + b",
            "def add(a, b): return a - b",
        ]
        test_lists = [
            ["assert add(1, 2) == 3"],
            ["assert add(1, 2) == 3"],
        ]
        results = evaluate_batch(completions, test_lists)
        assert len(results) == 2
        assert results[0].all_passed
        assert not results[1].all_passed

    def test_with_imports(self):
        completions = [
            "import math\ndef area(r): return math.pi * r * r",
        ]
        test_lists = [["assert abs(area(1) - 3.14159) < 0.001"]]
        import_lists = [["import math"]]
        results = evaluate_batch(completions, test_lists, import_lists)
        assert results[0].all_passed

    def test_empty_batch(self):
        results = evaluate_batch([], [], [])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# compute_pass_at_1
# ---------------------------------------------------------------------------


class TestComputePassAt1:
    """Compute pass@1 from a list of execution results."""

    def test_all_pass(self):
        results = [
            ExecutionResult(passed=2, total=2, error=None),
            ExecutionResult(passed=3, total=3, error=None),
        ]
        assert compute_pass_at_1(results) == 1.0

    def test_none_pass(self):
        results = [
            ExecutionResult(passed=0, total=2, error="fail"),
            ExecutionResult(passed=1, total=3, error="partial"),
        ]
        assert compute_pass_at_1(results) == 0.0

    def test_half_pass(self):
        results = [
            ExecutionResult(passed=2, total=2, error=None),
            ExecutionResult(passed=0, total=2, error="fail"),
        ]
        assert compute_pass_at_1(results) == 0.5

    def test_empty_list(self):
        assert compute_pass_at_1([]) == 0.0

    def test_returns_fraction(self):
        results = [
            ExecutionResult(passed=1, total=1, error=None),
            ExecutionResult(passed=0, total=1, error="fail"),
            ExecutionResult(passed=1, total=1, error=None),
        ]
        assert abs(compute_pass_at_1(results) - 2 / 3) < 1e-6
