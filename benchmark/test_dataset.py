"""Tests for MBPP dataset loading, formatting, and prompt construction."""

from __future__ import annotations

import pytest
from datasets import Dataset

from dataset import (
    SYSTEM_PROMPT,
    format_prompt,
    load_mbpp_splits,
    make_train_dataset,
    make_eval_dataset,
)


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    """Tests for converting MBPP problem descriptions into model prompts."""

    def test_basic_formatting(self):
        result = format_prompt("Write a function to add two numbers.")
        assert isinstance(result, list)
        assert len(result) >= 1
        # Should contain the problem description
        found = any("add two numbers" in msg["content"] for msg in result)
        assert found, "Prompt should contain the problem description"

    def test_system_prompt_present(self):
        """The system prompt should set the coding context."""
        result = format_prompt("Write a function to add two numbers.")
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1, "Should have exactly one system message"
        assert "python" in system_msgs[0]["content"].lower()

    def test_user_message_present(self):
        result = format_prompt("Write a function to add two numbers.")
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "add two numbers" in user_msgs[0]["content"]

    def test_instructs_code_block_format(self):
        """The prompt should instruct the model to use ```python blocks."""
        result = format_prompt("Write a function.")
        full_text = " ".join(m["content"] for m in result)
        assert "```python" in full_text or "```" in full_text, "Prompt should instruct model to use fenced code blocks"

    def test_does_not_leak_test_cases(self):
        """The prompt must NOT reveal test assertions — that would be cheating."""
        result = format_prompt("Write a function to add.")
        full_text = " ".join(m["content"] for m in result)
        assert "assert" not in full_text.lower()

    def test_empty_description(self):
        """Should handle empty description gracefully."""
        result = format_prompt("")
        assert isinstance(result, list)

    def test_description_with_special_chars(self):
        desc = "Write a function that handles 'quotes' and \"double quotes\" and <brackets>."
        result = format_prompt(desc)
        user_msg = [m for m in result if m["role"] == "user"][0]
        assert "quotes" in user_msg["content"]

    def test_long_description(self):
        """Long descriptions should be preserved fully."""
        desc = "Write a function " * 50
        result = format_prompt(desc)
        user_msg = [m for m in result if m["role"] == "user"][0]
        assert len(user_msg["content"]) > 200


# ---------------------------------------------------------------------------
# load_mbpp_splits
# ---------------------------------------------------------------------------


class TestLoadMBPPSplits:
    """Tests for loading MBPP from HuggingFace."""

    @pytest.fixture(scope="class")
    def splits(self):
        """Load MBPP once for all tests in this class."""
        return load_mbpp_splits()

    def test_returns_train_and_eval(self, splits):
        train, eval_ds = splits
        assert train is not None
        assert eval_ds is not None

    def test_train_has_problems(self, splits):
        train, _ = splits
        assert len(train) > 0

    def test_eval_has_problems(self, splits):
        _, eval_ds = splits
        assert len(eval_ds) > 0

    def test_no_overlap(self, splits):
        """Train and eval should not share task_ids."""
        train, eval_ds = splits
        train_ids = set(train["task_id"])
        eval_ids = set(eval_ds["task_id"])
        overlap = train_ids & eval_ids
        assert len(overlap) == 0, f"Overlapping task_ids: {overlap}"

    def test_required_columns(self, splits):
        """Both splits need prompt, test_list, test_imports, task_id."""
        train, eval_ds = splits
        for ds in [train, eval_ds]:
            assert "prompt" in ds.column_names
            assert "test_list" in ds.column_names
            assert "test_imports" in ds.column_names
            assert "task_id" in ds.column_names

    def test_train_prompt_is_string(self, splits):
        train, _ = splits
        assert isinstance(train[0]["prompt"], str)

    def test_test_list_is_list_of_strings(self, splits):
        train, _ = splits
        tl = train[0]["test_list"]
        assert isinstance(tl, list)
        assert all(isinstance(t, str) for t in tl)
        assert all("assert" in t for t in tl)

    def test_test_imports_is_list(self, splits):
        train, _ = splits
        ti = train[0]["test_imports"]
        assert isinstance(ti, list)


# ---------------------------------------------------------------------------
# make_train_dataset
# ---------------------------------------------------------------------------


class TestMakeTrainDataset:
    """Tests for the TRL-ready training dataset."""

    @pytest.fixture(scope="class")
    def train_ds(self):
        raw_train, _ = load_mbpp_splits()
        return make_train_dataset(raw_train, seed=42)

    def test_has_prompt_column(self, train_ds):
        assert "prompt" in train_ds.column_names

    def test_prompt_is_chat_formatted(self, train_ds):
        """Prompts should be lists of message dicts for apply_chat_template."""
        p = train_ds[0]["prompt"]
        assert isinstance(p, list), f"Expected list, got {type(p)}"
        assert isinstance(p[0], dict)
        assert "role" in p[0]
        assert "content" in p[0]

    def test_has_test_list_column(self, train_ds):
        """test_list must survive into the final dataset for the reward function."""
        assert "test_list" in train_ds.column_names

    def test_has_test_imports_column(self, train_ds):
        assert "test_imports" in train_ds.column_names

    def test_deterministic_shuffle(self):
        """Same seed should produce same order."""
        raw_train, _ = load_mbpp_splits()
        ds1 = make_train_dataset(raw_train, seed=42)
        ds2 = make_train_dataset(raw_train, seed=42)
        assert ds1[0]["task_id"] == ds2[0]["task_id"]
        assert ds1[-1]["task_id"] == ds2[-1]["task_id"]

    def test_different_seed_different_order(self):
        raw_train, _ = load_mbpp_splits()
        ds1 = make_train_dataset(raw_train, seed=42)
        ds2 = make_train_dataset(raw_train, seed=99)
        # With high probability these will differ (120 problems)
        ids1 = [ds1[i]["task_id"] for i in range(min(10, len(ds1)))]
        ids2 = [ds2[i]["task_id"] for i in range(min(10, len(ds2)))]
        assert ids1 != ids2

    def test_size_matches_input(self):
        raw_train, _ = load_mbpp_splits()
        ds = make_train_dataset(raw_train, seed=42)
        assert len(ds) == len(raw_train)


# ---------------------------------------------------------------------------
# make_eval_dataset
# ---------------------------------------------------------------------------


class TestMakeEvalDataset:
    """Tests for the evaluation dataset."""

    @pytest.fixture(scope="class")
    def eval_ds(self):
        _, raw_eval = load_mbpp_splits()
        return make_eval_dataset(raw_eval)

    def test_has_prompt_column(self, eval_ds):
        assert "prompt" in eval_ds.column_names

    def test_has_test_list_column(self, eval_ds):
        assert "test_list" in eval_ds.column_names

    def test_has_task_id(self, eval_ds):
        assert "task_id" in eval_ds.column_names

    def test_prompt_is_string(self, eval_ds):
        """Eval prompts are plain strings (for generate + decode)."""
        p = eval_ds[0]["prompt"]
        assert isinstance(p, (str, list))

    def test_not_shuffled(self):
        """Eval dataset should be deterministic order (not shuffled)."""
        _, raw_eval = load_mbpp_splits()
        ds1 = make_eval_dataset(raw_eval)
        ds2 = make_eval_dataset(raw_eval)
        ids1 = [ds1[i]["task_id"] for i in range(min(5, len(ds1)))]
        ids2 = [ds2[i]["task_id"] for i in range(min(5, len(ds2)))]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """Sanity checks on the system prompt constant."""

    def test_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_mentions_python(self):
        assert "python" in SYSTEM_PROMPT.lower()

    def test_not_too_long(self):
        """Keep it concise — we need room for the actual problem + completion."""
        assert len(SYSTEM_PROMPT) < 1000

    def test_mentions_code_blocks(self):
        assert "```" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Integration: reward function receives dataset columns correctly
# ---------------------------------------------------------------------------


class TestRewardIntegration:
    """
    Verify that dataset columns would flow correctly to the reward function.

    We can't test TRL's actual column forwarding without a full trainer, but we
    CAN verify the dataset has the right shape and that our reward function
    accepts the columns by name.
    """

    def test_reward_fn_accepts_dataset_columns(self):
        """Simulate what GRPOTrainer does: repeat columns num_generations times."""
        from reward_mbpp import MBPPRewardFunction

        raw_train, _ = load_mbpp_splits()
        ds = make_train_dataset(raw_train, seed=42)
        num_generations = 4
        sample = ds[0]

        # Simulate GRPOTrainer's column expansion
        prompts = ["prompt text"] * num_generations
        completions = ["```python\ndef f(): pass\n```"] * num_generations
        test_list = [sample["test_list"]] * num_generations
        test_imports = [sample["test_imports"]] * num_generations

        reward_fn = MBPPRewardFunction()
        rewards = reward_fn(
            prompts=prompts,
            completions=completions,
            test_list=test_list,
            test_imports=test_imports,
        )
        assert len(rewards) == num_generations
        assert all(isinstance(r, float) for r in rewards)
