"""Tests for MBPP sandboxed code execution and reward/feedback extraction."""

from __future__ import annotations

import pytest

from reward_mbpp import (
    ExecutionResult,
    FormatRewardFunction,
    execute_code_with_tests,
    extract_code_block,
    score_code_format,
    MBPPRewardFunction,
    _unwrap_completion,
)


# ---------------------------------------------------------------------------
# extract_code_block
# ---------------------------------------------------------------------------


class TestExtractCodeBlock:
    """Parse python code blocks from LLM completions."""

    def test_fenced_block(self):
        text = "Here is the solution:\n```python\ndef foo():\n    return 42\n```\nDone."
        assert extract_code_block(text) == "def foo():\n    return 42"

    def test_fenced_block_no_language_tag(self):
        text = "Solution:\n```\ndef foo():\n    return 42\n```"
        assert extract_code_block(text) == "def foo():\n    return 42"

    def test_no_fence_raw_code(self):
        """When there's no code fence, treat the whole thing as code."""
        text = "def foo():\n    return 42"
        assert extract_code_block(text) == "def foo():\n    return 42"

    def test_multiple_blocks_takes_first(self):
        text = "```python\ndef a(): pass\n```\nand\n```python\ndef b(): pass\n```"
        assert extract_code_block(text) == "def a(): pass"

    def test_empty_completion(self):
        assert extract_code_block("") == ""

    def test_strips_whitespace(self):
        text = "```python\n\n  def foo():\n      return 1\n\n```"
        assert "def foo()" in extract_code_block(text)


# ---------------------------------------------------------------------------
# execute_code_with_tests
# ---------------------------------------------------------------------------


class TestExecuteCodeWithTests:
    """Sandboxed execution of code + assertion tests."""

    def test_all_pass(self):
        code = "def add(a, b): return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = execute_code_with_tests(code, tests)
        assert result.passed == 2
        assert result.total == 2
        assert result.all_passed
        assert result.error is None

    def test_assertion_failure(self):
        code = "def add(a, b): return a - b"  # wrong!
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = execute_code_with_tests(code, tests)
        assert result.passed < result.total
        assert not result.all_passed
        assert result.error is not None
        assert "AssertionError" in result.error or "assert" in result.error.lower()

    def test_syntax_error(self):
        code = "def add(a, b) return a + b"  # missing colon
        tests = ["assert add(1, 2) == 3"]
        result = execute_code_with_tests(code, tests)
        assert result.passed == 0
        assert not result.all_passed
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        code = "def divide(a, b): return a / b"
        tests = ["assert divide(1, 0) == 0"]
        result = execute_code_with_tests(code, tests)
        assert not result.all_passed
        assert "ZeroDivisionError" in result.error

    def test_timeout(self):
        code = "def slow():\n    while True: pass"
        tests = ["assert slow() is None"]
        result = execute_code_with_tests(code, tests, timeout=1)
        assert not result.all_passed
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    def test_import_handling(self):
        code = "import math\ndef area(r): return math.pi * r * r"
        tests = ["assert abs(area(1) - 3.14159) < 0.001"]
        result = execute_code_with_tests(code, tests, imports=["import math"])
        assert result.all_passed

    def test_partial_pass(self):
        """2 of 3 tests pass."""
        code = "def f(x): return x * 2"
        tests = [
            "assert f(2) == 4",  # pass
            "assert f(3) == 6",  # pass
            "assert f(0) == 1",  # fail — 0*2=0, not 1
        ]
        result = execute_code_with_tests(code, tests)
        assert result.passed == 2
        assert result.total == 3
        assert not result.all_passed

    def test_no_function_defined(self):
        code = "x = 42"  # no function
        tests = ["assert add(1, 2) == 3"]
        result = execute_code_with_tests(code, tests)
        assert not result.all_passed
        assert "NameError" in result.error

    def test_empty_code(self):
        result = execute_code_with_tests("", ["assert True"])
        # Empty code means assert True should pass
        assert result.all_passed


# ---------------------------------------------------------------------------
# MBPPRewardFunction (the TRL-compatible callable)
# ---------------------------------------------------------------------------


class TestMBPPRewardFunction:
    """The reward function that TRL's GRPOTrainer calls."""

    @pytest.fixture
    def reward_fn(self):
        return MBPPRewardFunction()

    def test_returns_list_of_floats(self, reward_fn):
        prompts = ["Write a function to add two numbers."]
        completions = ["```python\ndef add(a, b): return a + b\n```"]
        # Need to pass test cases as kwargs
        result = reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3", "assert add(0, 0) == 0"]],
            test_imports=[[]],
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_perfect_score(self, reward_fn):
        prompts = ["Write add"]
        completions = ["```python\ndef add(a, b): return a + b\n```"]
        result = reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3"]],
            test_imports=[[]],
        )
        assert result[0] == 1.0

    def test_zero_score_on_failure(self, reward_fn):
        prompts = ["Write add"]
        completions = ["```python\ndef add(a, b): return a - b\n```"]
        result = reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3"]],
            test_imports=[[]],
        )
        assert result[0] == 0.0

    def test_partial_score(self, reward_fn):
        prompts = ["Write f"]
        completions = ["```python\ndef f(x): return x * 2\n```"]
        result = reward_fn(
            prompts,
            completions,
            test_list=[["assert f(2) == 4", "assert f(3) == 6", "assert f(0) == 1"]],
            test_imports=[[]],
        )
        # 2/3 tests pass → partial reward
        assert 0.0 < result[0] < 1.0

    def test_feedback_stored_on_failure(self, reward_fn):
        """Feedback is stored after scoring — contains error details for failed code."""
        prompts = ["Write add"]
        completions = ["```python\ndef add(a, b): return a - b\n```"]
        result = reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3"]],
            test_imports=[[]],
        )
        assert result[0] == 0.0
        assert reward_fn.last_feedback is not None
        assert len(reward_fn.last_feedback) == 1
        assert isinstance(reward_fn.last_feedback[0], str)

    def test_feedback_stored(self, reward_fn):
        prompts = ["Write add"]
        completions = ["```python\ndef add(a, b): return a - b\n```"]
        reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3"]],
            test_imports=[[]],
        )
        assert reward_fn.last_feedback is not None
        assert len(reward_fn.last_feedback) == 1
        assert isinstance(reward_fn.last_feedback[0], str)
        # Feedback should mention the error
        assert "assert" in reward_fn.last_feedback[0].lower() or "error" in reward_fn.last_feedback[0].lower()

    def test_feedback_on_success(self, reward_fn):
        prompts = ["Write add"]
        completions = ["```python\ndef add(a, b): return a + b\n```"]
        reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3"]],
            test_imports=[[]],
        )
        fb = reward_fn.last_feedback[0]
        assert "pass" in fb.lower()

    def test_batch_processing(self, reward_fn):
        prompts = ["Write add", "Write sub"]
        completions = [
            "```python\ndef add(a, b): return a + b\n```",
            "```python\ndef sub(a, b): return a + b\n```",  # wrong!
        ]
        result = reward_fn(
            prompts,
            completions,
            test_list=[
                ["assert add(1, 2) == 3"],
                ["assert sub(3, 1) == 2"],
            ],
            test_imports=[[], []],
        )
        assert len(result) == 2
        assert result[0] == 1.0  # add passes
        assert result[1] == 0.0  # sub fails

    def test_has_name_attribute(self, reward_fn):
        """TRL requires __name__ for logging."""
        assert hasattr(reward_fn, "__name__")

    def test_chat_message_completions(self, reward_fn):
        """TRL passes completions as chat messages in conversational mode."""
        prompts = [[{"role": "user", "content": "Write add"}]]
        completions = [[{"role": "assistant", "content": "```python\ndef add(a, b): return a + b\n```"}]]
        result = reward_fn(
            prompts,
            completions,
            test_list=[["assert add(1, 2) == 3"]],
            test_imports=[[]],
        )
        assert result[0] == 1.0


# ---------------------------------------------------------------------------
# _unwrap_completion
# ---------------------------------------------------------------------------


class TestUnwrapCompletion:
    def test_string_passthrough(self):
        assert _unwrap_completion("hello") == "hello"

    def test_chat_message(self):
        msg = [{"role": "assistant", "content": "code here"}]
        assert _unwrap_completion(msg) == "code here"

    def test_empty_list(self):
        assert _unwrap_completion([]) == ""

    def test_none(self):
        assert _unwrap_completion(None) == ""


# ---------------------------------------------------------------------------
# extract_code_block (additional regex tests)
# ---------------------------------------------------------------------------


class TestExtractCodeBlockPermissive:
    """Test the more permissive regex patterns."""

    def test_no_newline_after_fence(self):
        """Model emits ```python immediately followed by code, no newline."""
        text = "```pythondef foo(): return 42```"
        result = extract_code_block(text)
        assert "def foo" in result

    def test_space_but_no_newline(self):
        text = "```python def foo(): return 42```"
        result = extract_code_block(text)
        assert "def foo" in result

    def test_normal_fence_still_works(self):
        text = "```python\ndef foo():\n    return 42\n```"
        assert extract_code_block(text) == "def foo():\n    return 42"


# ---------------------------------------------------------------------------
# score_code_format
# ---------------------------------------------------------------------------


class TestScoreCodeFormat:
    """Test the format scoring function for GRPO cold-start signal."""

    def test_perfect_fenced_valid_python(self):
        text = "```python\ndef add(a, b): return a + b\n```"
        assert score_code_format(text) == 1.0

    def test_valid_python_no_fence(self):
        text = "def add(a, b): return a + b"
        assert score_code_format(text) == 0.5

    def test_fenced_but_syntax_error(self):
        text = "```python\ndef add(a, b) return a + b\n```"
        assert score_code_format(text) == 0.2

    def test_has_def_but_broken(self):
        text = "def add(a, b) I think we should return a + b somehow"
        assert score_code_format(text) == 0.1

    def test_empty_string(self):
        assert score_code_format("") == 0.0

    def test_nonsense_text_with_apostrophe(self):
        # Apostrophe makes it a SyntaxError, no fence, no def → 0.0
        assert score_code_format("I don't know how to solve this.") == 0.0

    def test_plain_english_no_def(self):
        # Valid Python expression (string), no fence, no 'def ' → 0.5
        # Actually "Hello world" would be a bare expression...
        # But a full sentence with periods is a SyntaxError
        text = "Hello, I am a language model and I cannot write code."
        assert score_code_format(text) == 0.0  # SyntaxError, no fence, no def

    def test_valid_expression_no_fence(self):
        # A raw "x = 42" is valid Python, no fence → 0.5
        assert score_code_format("x = 42") == 0.5

    def test_multiline_valid_fenced(self):
        text = "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
        assert score_code_format(text) == 1.0

    def test_whitespace_only(self):
        assert score_code_format("   \n  \n  ") == 0.0


# ---------------------------------------------------------------------------
# FormatRewardFunction (TRL-compatible callable)
# ---------------------------------------------------------------------------


class TestFormatRewardFunction:
    """The format reward function used as second reward in GRPOTrainer."""

    @pytest.fixture
    def fmt_fn(self):
        return FormatRewardFunction()

    def test_returns_list(self, fmt_fn):
        result = fmt_fn(completions=["def foo(): pass"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_name(self, fmt_fn):
        assert fmt_fn.__name__ == "format_reward"

    def test_batch(self, fmt_fn):
        completions = [
            "```python\ndef add(a,b): return a+b\n```",
            "I don't know",
            "def foo(): pass",
        ]
        result = fmt_fn(completions=completions)
        assert len(result) == 3
        assert result[0] == 1.0  # fenced + valid
        assert result[2] == 0.5  # valid, no fence

    def test_chat_messages(self, fmt_fn):
        completions = [
            [{"role": "assistant", "content": "```python\ndef f(): pass\n```"}],
        ]
        result = fmt_fn(completions=completions)
        assert result[0] == 1.0

    def test_empty_completions(self, fmt_fn):
        assert fmt_fn(completions=[]) == []

    def test_accepts_kwargs(self, fmt_fn):
        """FormatRewardFunction ignores extra kwargs like test_list."""
        result = fmt_fn(
            prompts=["hello"],
            completions=["```python\ndef f(): pass\n```"],
            test_list=[["assert True"]],
            test_imports=[[]],
        )
        assert result[0] == 1.0
