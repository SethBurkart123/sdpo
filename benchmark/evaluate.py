"""
MBPP evaluation: execute completions against test cases and compute pass@1.

Used for periodic evaluation during training and final reporting.
"""

from __future__ import annotations

from reward_mbpp import ExecutionResult, execute_code_with_tests, extract_code_block


def evaluate_completion(
    completion: str,
    tests: list[str],
    imports: list[str] | None = None,
    timeout: int = 5,
) -> ExecutionResult:
    """
    Evaluate a single completion against test assertions.

    Handles code block extraction, so raw LLM output is fine.

    Args:
        completion: Raw LLM completion (may include ```python blocks).
        tests: List of assert statements.
        imports: Optional import statements to prepend.
        timeout: Execution timeout in seconds.

    Returns:
        ExecutionResult with pass/fail details.
    """
    code = extract_code_block(completion)
    return execute_code_with_tests(code, tests, imports=imports, timeout=timeout)


def evaluate_batch(
    completions: list[str],
    test_lists: list[list[str]],
    import_lists: list[list[str]] | None = None,
    timeout: int = 5,
) -> list[ExecutionResult]:
    """
    Evaluate a batch of completions against their respective test cases.

    Args:
        completions: Raw LLM completions.
        test_lists: Per-problem list of assert statements.
        import_lists: Per-problem list of import statements.
        timeout: Execution timeout per problem.

    Returns:
        List of ExecutionResult, one per completion.
    """
    if import_lists is None:
        import_lists = [[] for _ in completions]

    return [
        evaluate_completion(comp, tests, imports=imports, timeout=timeout)
        for comp, tests, imports in zip(completions, test_lists, import_lists)
    ]


def compute_pass_at_1(results: list[ExecutionResult]) -> float:
    """
    Compute pass@1: fraction of problems where all tests passed.

    Args:
        results: List of ExecutionResult from evaluate_batch.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.all_passed)
    return passed / len(results)
