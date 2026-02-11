"""
Tests for sdpo_rl.reprompting — teacher prompt construction.

Matches the behavior of _maybe_build_self_distillation_batch, _collect_solutions_by_uid,
_get_solution, and _build_teacher_message from verl/trainer/ppo/ray_trainer.py.
"""

import pytest

from sdpo_rl.reprompting import (
    build_teacher_prompts,
    select_demonstration,
    remove_thinking_tags,
    DEFAULT_REPROMPT_TEMPLATE,
    DEFAULT_SOLUTION_TEMPLATE,
    DEFAULT_FEEDBACK_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rollout_data():
    """
    Simulates a batch of 8 rollouts for 2 prompts (4 rollouts each).
    Prompt UIDs: [A, A, A, A, B, B, B, B]
    Rewards:     [1,  0,  0,  1,  0,  0,  1,  0]
    """
    return {
        "prompts": [
            "What is 2+2?",
            "What is 2+2?",
            "What is 2+2?",
            "What is 2+2?",
            "Write hello world in Python.",
            "Write hello world in Python.",
            "Write hello world in Python.",
            "Write hello world in Python.",
        ],
        "completions": [
            "The answer is 4.",  # idx 0 - correct
            "I think it's 5.",  # idx 1 - wrong
            "<think>Let me think...</think>\nThe answer is 3.",  # idx 2 - wrong
            "4",  # idx 3 - correct
            "print('goodbye')",  # idx 4 - wrong
            "print('hello world')\nBut wait",  # idx 5 - wrong
            "print('hello world')",  # idx 6 - correct
            "exit()",  # idx 7 - wrong
        ],
        "uids": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "rewards": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "feedback": [
            None,
            "Your answer is incorrect. The correct answer is 4.",
            "Your answer is incorrect. The correct answer is 4.",
            None,
            "RuntimeError: unexpected output",
            "Wrong Answer\nExpected: hello world\nGot: hello world\nBut wait",
            None,
            "RuntimeError: process exited",
        ],
    }


# ---------------------------------------------------------------------------
# 1. Thinking Tag Removal
# ---------------------------------------------------------------------------


class TestRemoveThinkingTags:
    def test_basic_removal(self):
        text = "<think>some reasoning</think>\nThe answer is 4."
        assert remove_thinking_tags(text) == "The answer is 4."

    def test_multiline_thinking(self):
        text = "<think>\nLine 1\nLine 2\nLine 3\n</think>\nAnswer."
        assert remove_thinking_tags(text) == "Answer."

    def test_no_tags_unchanged(self):
        text = "Just a normal response."
        assert remove_thinking_tags(text) == "Just a normal response."

    def test_multiple_think_blocks(self):
        text = "<think>first</think> middle <think>second</think> end"
        result = remove_thinking_tags(text)
        assert "<think>" not in result
        assert "end" in result

    def test_empty_think_block(self):
        text = "<think></think>Answer."
        assert remove_thinking_tags(text) == "Answer."


# ---------------------------------------------------------------------------
# 2. Demonstration Selection
# ---------------------------------------------------------------------------


class TestSelectDemonstration:
    def test_finds_successful_peer(self, rollout_data):
        """For idx=1 (wrong, uid=A), should find a successful peer (idx 0 or 3)."""
        demo = select_demonstration(
            idx=1,
            uids=rollout_data["uids"],
            rewards=rollout_data["rewards"],
            completions=rollout_data["completions"],
            success_threshold=1.0,
            exclude_self=True,
            remove_thinking=False,
        )
        assert demo is not None
        assert demo in ["The answer is 4.", "4"]

    def test_excludes_self_when_flagged(self, rollout_data):
        """For idx=0 (correct, uid=A), excluding self should still find idx=3."""
        demo = select_demonstration(
            idx=0,
            uids=rollout_data["uids"],
            rewards=rollout_data["rewards"],
            completions=rollout_data["completions"],
            success_threshold=1.0,
            exclude_self=True,
            remove_thinking=False,
        )
        assert demo is not None
        assert demo == "4"  # idx=3 is the only other successful peer

    def test_includes_self_when_not_excluded(self, rollout_data):
        """For idx=0 (correct), not excluding self can return own completion."""
        demo = select_demonstration(
            idx=0,
            uids=rollout_data["uids"],
            rewards=rollout_data["rewards"],
            completions=rollout_data["completions"],
            success_threshold=1.0,
            exclude_self=False,
            remove_thinking=False,
        )
        assert demo is not None
        # Could be idx 0 or idx 3

    def test_no_success_returns_none(self):
        """When no peer succeeded, should return None."""
        demo = select_demonstration(
            idx=0,
            uids=["X", "X", "X"],
            rewards=[0.0, 0.0, 0.0],
            completions=["a", "b", "c"],
            success_threshold=1.0,
            exclude_self=True,
        )
        assert demo is None

    def test_removes_thinking_tags(self, rollout_data):
        """When remove_thinking=True, <think> blocks should be stripped from demo."""
        # Make idx=2's completion the only successful one for uid A
        rewards = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        demo = select_demonstration(
            idx=1,
            uids=rollout_data["uids"],
            rewards=rewards,
            completions=rollout_data["completions"],
            success_threshold=1.0,
            exclude_self=True,
            remove_thinking=True,
        )
        assert demo is not None
        assert "<think>" not in demo
        assert "The answer is 3." in demo

    def test_threshold_filters_correctly(self, rollout_data):
        """Rewards below threshold should not be selected as demonstrations."""
        # With threshold=2.0, only rewards >= 2.0 qualify — none do
        demo = select_demonstration(
            idx=1,
            uids=rollout_data["uids"],
            rewards=rollout_data["rewards"],
            completions=rollout_data["completions"],
            success_threshold=2.0,
            exclude_self=True,
        )
        assert demo is None


# ---------------------------------------------------------------------------
# 3. Template Assembly
# ---------------------------------------------------------------------------


class TestBuildTeacherPrompts:
    def test_with_solution_and_feedback(self):
        """Full template: prompt + solution section + feedback section + suffix."""
        result = build_teacher_prompts(
            prompts=["What is 2+2?"],
            solutions=["4"],
            feedback_list=["Your answer was wrong."],
            feedback_only_without_solution=False,
        )
        assert len(result) == 1
        text = result[0]
        assert "What is 2+2?" in text
        assert "Correct solution:" in text
        assert "4" in text
        assert "feedback from your unsuccessful earlier attempt" in text
        assert "Your answer was wrong." in text
        assert "Correctly solve the original question." in text

    def test_with_solution_only(self):
        """Solution available, no feedback."""
        result = build_teacher_prompts(
            prompts=["What is 2+2?"],
            solutions=["4"],
            feedback_list=[None],
        )
        text = result[0]
        assert "Correct solution:" in text
        assert "feedback from your unsuccessful" not in text

    def test_with_feedback_only(self):
        """No solution, only feedback."""
        result = build_teacher_prompts(
            prompts=["What is 2+2?"],
            solutions=[None],
            feedback_list=["Wrong answer."],
        )
        text = result[0]
        assert "Correct solution:" not in text
        assert "Wrong answer." in text

    def test_neither_solution_nor_feedback(self):
        """No solution, no feedback -> raw prompt passthrough."""
        result = build_teacher_prompts(
            prompts=["What is 2+2?"],
            solutions=[None],
            feedback_list=[None],
        )
        text = result[0]
        assert text == "What is 2+2?"

    def test_feedback_only_without_solution_flag(self):
        """
        When feedback_only_without_solution=True and a solution exists,
        feedback should NOT be included (only the solution).
        """
        result = build_teacher_prompts(
            prompts=["What is 2+2?"],
            solutions=["4"],
            feedback_list=["Wrong answer."],
            feedback_only_without_solution=True,
        )
        text = result[0]
        assert "Correct solution:" in text
        assert "4" in text
        assert "Wrong answer." not in text  # feedback suppressed because solution exists

    def test_batch_processing(self, rollout_data):
        """Should handle a full batch correctly."""
        # Build solutions for each sample
        solutions = [None] * 8
        for i in range(8):
            solutions[i] = select_demonstration(
                idx=i,
                uids=rollout_data["uids"],
                rewards=rollout_data["rewards"],
                completions=rollout_data["completions"],
                success_threshold=1.0,
                exclude_self=True,
            )

        result = build_teacher_prompts(
            prompts=rollout_data["prompts"],
            solutions=solutions,
            feedback_list=rollout_data["feedback"],
        )
        assert len(result) == 8

    def test_custom_templates(self):
        """Users should be able to override the default templates."""
        result = build_teacher_prompts(
            prompts=["Q"],
            solutions=["A"],
            feedback_list=[None],
            reprompt_template="{prompt} | SOL: {solution}{feedback} | END",
            solution_template=" [{successful_previous_attempt}]",
            feedback_template=" FEEDBACK: {feedback_raw}",
        )
        assert result[0] == "Q | SOL:  [A] | END"


# ---------------------------------------------------------------------------
# 4. Self-Distillation Mask Construction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4a. Teacher Message Construction (structured chat messages)
# ---------------------------------------------------------------------------


class TestBuildTeacherMessages:
    """
    Tests for build_teacher_messages — the function that constructs
    properly structured chat message lists for the teacher prompt.

    Bug Fix 1: The original code wrapped everything in a single user message,
    dropping system messages. The reference preserves system messages from the
    original prompt. See SDPO_AUDIT.md Bug 1.
    """

    def test_preserves_system_messages(self):
        """System messages from the original prompt must be preserved."""
        from sdpo_rl.reprompting import build_teacher_messages

        original_messages = [
            [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        ]
        teacher_prompt_texts = ["What is 2+2?\nCorrect solution:\n\n4\n\n\n\nCorrectly solve the original question.\n"]

        result = build_teacher_messages(original_messages, teacher_prompt_texts)

        assert len(result) == 1
        messages = result[0]
        # First message should be the system message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful math assistant."
        # Last message should be the user message with the reprompted content
        assert messages[-1]["role"] == "user"
        assert "Correctly solve the original question." in messages[-1]["content"]

    def test_no_system_message(self):
        """When there's no system message, only the user message is returned."""
        from sdpo_rl.reprompting import build_teacher_messages

        original_messages = [[{"role": "user", "content": "What is 2+2?"}]]
        teacher_prompt_texts = ["What is 2+2?\n\nCorrectly solve the original question.\n"]

        result = build_teacher_messages(original_messages, teacher_prompt_texts)

        assert len(result) == 1
        messages = result[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_multiple_system_messages(self):
        """Multiple system/context messages should all be preserved."""
        from sdpo_rl.reprompting import build_teacher_messages

        original_messages = [
            [
                {"role": "system", "content": "You are a coder."},
                {"role": "user", "content": "Write hello world."},
                {"role": "assistant", "content": "print('hello')"},
                {"role": "user", "content": "Now write goodbye."},
            ]
        ]
        teacher_prompt_texts = ["Now write goodbye.\n\nCorrectly solve the original question.\n"]

        result = build_teacher_messages(original_messages, teacher_prompt_texts)

        messages = result[0]
        # All messages except the last user turn should be preserved
        assert len(messages) == 4  # system + user + assistant + new_user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert "Correctly solve the original question." in messages[3]["content"]

    def test_batch_processing(self):
        """Should handle a batch of messages correctly."""
        from sdpo_rl.reprompting import build_teacher_messages

        original_messages = [
            [
                {"role": "system", "content": "System A"},
                {"role": "user", "content": "Question A"},
            ],
            [
                {"role": "user", "content": "Question B"},  # no system msg
            ],
        ]
        teacher_prompt_texts = ["Reprompted A", "Reprompted B"]

        result = build_teacher_messages(original_messages, teacher_prompt_texts)

        assert len(result) == 2
        # First sample has system message
        assert result[0][0]["role"] == "system"
        assert result[0][1]["role"] == "user"
        assert result[0][1]["content"] == "Reprompted A"
        # Second sample has no system message
        assert len(result[1]) == 1
        assert result[1][0]["role"] == "user"
        assert result[1][0]["content"] == "Reprompted B"

    def test_plain_string_prompts_treated_as_user_message(self):
        """When original_messages contains plain strings, treat as single user message."""
        from sdpo_rl.reprompting import build_teacher_messages

        original_messages = [
            "What is 2+2?",  # plain string, not a message list
        ]
        teacher_prompt_texts = ["Reprompted text"]

        result = build_teacher_messages(original_messages, teacher_prompt_texts)

        assert len(result) == 1
        messages = result[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Reprompted text"


# ---------------------------------------------------------------------------
# 4b. Self-Distillation Mask Construction
# ---------------------------------------------------------------------------


class TestSelfDistillationMaskConstruction:
    def test_mask_reflects_teacher_availability(self, rollout_data):
        """Mask should be 1 where solution OR feedback exists, 0 otherwise."""
        solutions = [None] * 8
        for i in range(8):
            solutions[i] = select_demonstration(
                idx=i,
                uids=rollout_data["uids"],
                rewards=rollout_data["rewards"],
                completions=rollout_data["completions"],
                success_threshold=1.0,
                exclude_self=True,
            )

        from sdpo_rl.reprompting import compute_self_distillation_mask

        mask = compute_self_distillation_mask(solutions, rollout_data["feedback"])
        assert len(mask) == 8
        for i in range(8):
            has_signal = solutions[i] is not None or (
                rollout_data["feedback"][i] is not None and rollout_data["feedback"][i].strip()
            )
            assert mask[i] == (1.0 if has_signal else 0.0), f"Mask wrong at idx {i}"
