"""Tests for sdpo_rl.config — SDPOConfig validation."""

import pytest

from sdpo_rl.config import SDPOConfig


class TestSDPOConfigDefaults:
    """Verify defaults match the paper's experiment scripts, not verl yaml."""

    def test_alpha_is_jsd(self):
        """Paper uses alpha=0.5 (JSD), not verl's default 0.0."""
        cfg = SDPOConfig()
        assert cfg.alpha == 0.5

    def test_dont_reprompt_on_self_success(self):
        """Paper uses True, verl yaml defaults to False."""
        cfg = SDPOConfig()
        assert cfg.dont_reprompt_on_self_success is True

    def test_remove_thinking(self):
        """Paper uses True, verl yaml defaults to False."""
        cfg = SDPOConfig()
        assert cfg.remove_thinking_from_demonstration is True

    def test_include_environment_feedback(self):
        """Paper uses True, verl yaml defaults to False."""
        cfg = SDPOConfig()
        assert cfg.include_environment_feedback is True

    def test_teacher_mode_ema(self):
        cfg = SDPOConfig()
        assert cfg.teacher_mode == "ema"

    def test_teacher_update_rate(self):
        cfg = SDPOConfig()
        assert cfg.teacher_update_rate == 0.05

    def test_distillation_topk(self):
        cfg = SDPOConfig()
        assert cfg.distillation_topk == 100

    def test_is_clip(self):
        cfg = SDPOConfig()
        assert cfg.is_clip == 2.0

    def test_success_reward_threshold(self):
        cfg = SDPOConfig()
        assert cfg.success_reward_threshold == 1.0


class TestSDPOConfigValidation:
    def test_invalid_alpha_too_low(self):
        with pytest.raises(ValueError, match="alpha"):
            SDPOConfig(alpha=-0.1)

    def test_invalid_alpha_too_high(self):
        with pytest.raises(ValueError, match="alpha"):
            SDPOConfig(alpha=1.5)

    def test_valid_alpha_boundaries(self):
        SDPOConfig(alpha=0.0)
        SDPOConfig(alpha=1.0)

    def test_invalid_teacher_mode(self):
        with pytest.raises(ValueError, match="teacher_mode"):
            SDPOConfig(teacher_mode="invalid")

    def test_valid_teacher_modes(self):
        for mode in ("ema", "trust_region", "frozen"):
            SDPOConfig(teacher_mode=mode)

    def test_invalid_truncation(self):
        with pytest.raises(ValueError, match="reprompt_truncation"):
            SDPOConfig(reprompt_truncation="middle")

    def test_invalid_topk(self):
        with pytest.raises(ValueError, match="distillation_topk"):
            SDPOConfig(distillation_topk=0)

    def test_lora_ema_teacher_mode_valid(self):
        """lora_ema should be an accepted teacher_mode."""
        cfg = SDPOConfig(teacher_mode="lora_ema")
        assert cfg.teacher_mode == "lora_ema"


class TestSDPOConfigChatTemplateKwargs:
    """
    Bug Fix 2: apply_chat_template_kwargs must be forwarded to the
    tokenizer's apply_chat_template call. The reference passes
    enable_thinking and continue_final_message explicitly.
    See dev/audit.md Bug 2.
    """

    def test_default_is_empty_dict(self):
        cfg = SDPOConfig()
        assert cfg.apply_chat_template_kwargs == {}

    def test_custom_kwargs_stored(self):
        cfg = SDPOConfig(apply_chat_template_kwargs={"enable_thinking": True})
        assert cfg.apply_chat_template_kwargs["enable_thinking"] is True

    def test_multiple_kwargs(self):
        cfg = SDPOConfig(
            apply_chat_template_kwargs={
                "enable_thinking": False,
                "continue_final_message": False,
            }
        )
        assert cfg.apply_chat_template_kwargs["enable_thinking"] is False
        assert cfg.apply_chat_template_kwargs["continue_final_message"] is False
