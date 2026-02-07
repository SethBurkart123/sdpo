"""
Tests comparing our implementation against the verl reference (lasgroup/SDPO).

The reference functions are extracted verbatim from:
    SDPO_reference/verl/trainer/ppo/core_algos.py

They are standalone (no verl imports) so we can run them without ray/omegaconf.
We feed identical inputs to both our code and the reference, asserting outputs match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from sdpo_trainer.distillation import (
    add_tail_bucket,
    aggregate_loss,
    apply_importance_sampling_correction,
    compute_self_distillation_loss,
    top_k_kl_divergence,
)
from sdpo_trainer.reprompting import (
    build_teacher_prompts,
    compute_self_distillation_mask,
    remove_thinking_tags,
    select_demonstration,
)

# ---------------------------------------------------------------------------
# Reference implementations extracted verbatim from verl core_algos.py
# (only dependency: torch)
# ---------------------------------------------------------------------------


def _ref_masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """verl_F.masked_sum — simplified (no NaN handling needed in tests)."""
    return (values * mask).sum()


def _ref_add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Verbatim from compute_self_distillation_loss inner def."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _ref_renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
    """Verbatim from compute_self_distillation_loss inner def."""
    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
    return logp - logZ


def _ref_agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Verbatim from core_algos.py agg_loss (token-mean branch only)."""
    assert loss_agg_mode == "token-mean"
    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum()
    loss = _ref_masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    return loss


@dataclass
class _FakeConfig:
    """Minimal stand-in for verl's SelfDistillationConfig."""

    full_logit_distillation: bool = True
    distillation_topk: int = 10
    distillation_add_tail: bool = True
    alpha: float = 0.5
    is_clip: Optional[float] = 2.0


def _ref_compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: _FakeConfig,
    old_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Verbatim from core_algos.py compute_self_distillation_loss,
    with verl-specific imports replaced by inline equivalents.
    """
    metrics = {}
    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if config.full_logit_distillation:
        use_topk = config.distillation_topk is not None
        if use_topk:
            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if config.distillation_add_tail:
                student_distill_log_probs = _ref_add_tail(student_distill_log_probs)
                teacher_distill_log_probs = _ref_add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = _ref_renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = _ref_renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            raise ValueError("Non-topk full-logit not tested here.")

        if config.alpha == 0.0:
            kl_loss = F.kl_div(student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
        elif config.alpha == 1.0:
            kl_loss = F.kl_div(teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True)
        else:
            alpha = torch.tensor(
                config.alpha, dtype=student_distill_log_probs.dtype, device=student_distill_log_probs.device
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [student_distill_log_probs + torch.log(1 - alpha), teacher_distill_log_probs + torch.log(alpha)]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)

        per_token_loss = kl_loss.sum(-1)
    else:
        assert config.alpha == 1.0
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    is_clip = config.is_clip
    if is_clip is not None:
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = _ref_agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode="token-mean",
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )
    return loss, metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = "cpu"
DTYPE = torch.float64  # double precision for tighter tolerance
ATOL = 1e-10
RTOL = 1e-8


def _make_topk_log_probs(B: int, T: int, K: int) -> torch.Tensor:
    """Generate realistic top-K log-probs (negative, sums to < 1 in prob space)."""
    # Create raw logits, take softmax over a larger vocab, then pick top K
    raw = torch.randn(B, T, K * 4, dtype=DTYPE, device=DEVICE)
    log_probs_full = F.log_softmax(raw, dim=-1)
    topk_lp, _ = log_probs_full.topk(K, dim=-1)
    return topk_lp


def _make_per_token_log_probs(B: int, T: int) -> torch.Tensor:
    """Generate per-token log-probs (negative scalars)."""
    return -torch.rand(B, T, dtype=DTYPE, device=DEVICE).clamp(min=0.01) * 5


def _make_response_mask(B: int, T: int) -> torch.Tensor:
    """Generate a realistic response mask (1s followed by 0s)."""
    mask = torch.ones(B, T, dtype=DTYPE, device=DEVICE)
    # Randomly zero out some trailing tokens
    for i in range(B):
        cutoff = torch.randint(T // 2, T + 1, (1,)).item()
        mask[i, cutoff:] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Tests: add_tail
# ---------------------------------------------------------------------------


class TestAddTailMatch:
    """Compare our add_tail_bucket against the verl reference add_tail."""

    def test_identical_output_random(self):
        lp = _make_topk_log_probs(4, 16, 10)
        ours = add_tail_bucket(lp)
        ref = _ref_add_tail(lp)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_single_token(self):
        lp = _make_topk_log_probs(1, 1, 5)
        ours = add_tail_bucket(lp)
        ref = _ref_add_tail(lp)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_shape_preserved(self):
        lp = _make_topk_log_probs(2, 8, 20)
        ours = add_tail_bucket(lp)
        ref = _ref_add_tail(lp)
        assert ours.shape == ref.shape
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_extreme_values(self):
        """Near-uniform distribution where top-K covers almost all mass."""
        # K tokens each with prob ~1/K -> logsumexp close to 0
        K = 10
        lp = torch.full((2, 4, K), -2.302585, dtype=DTYPE, device=DEVICE)  # log(0.1)
        ours = add_tail_bucket(lp)
        ref = _ref_add_tail(lp)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Tests: top-K KL divergence (JSD, forward KL, reverse KL)
# ---------------------------------------------------------------------------


class TestTopKKLMatch:
    """Compare our top_k_kl_divergence against the reference KL computation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.B, self.T, self.K = 4, 16, 10
        self.student_lp = _make_topk_log_probs(self.B, self.T, self.K)
        self.teacher_lp = _make_topk_log_probs(self.B, self.T, self.K)

    def _ref_kl(self, student_lp, teacher_lp, alpha, add_tail):
        """Compute KL the reference way."""
        if add_tail:
            s = _ref_add_tail(student_lp)
            t = _ref_add_tail(teacher_lp)
        else:
            s = _ref_renorm_topk_log_probs(student_lp)
            t = _ref_renorm_topk_log_probs(teacher_lp)

        if alpha == 0.0:
            kl = F.kl_div(s, t, reduction="none", log_target=True)
        elif alpha == 1.0:
            kl = F.kl_div(t, s, reduction="none", log_target=True)
        else:
            alpha_t = torch.tensor(alpha, dtype=s.dtype, device=s.device)
            mixture = torch.logsumexp(
                torch.stack([s + torch.log(1.0 - alpha_t), t + torch.log(alpha_t)]),
                dim=0,
            )
            kl_t = F.kl_div(mixture, t, reduction="none", log_target=True)
            kl_s = F.kl_div(mixture, s, reduction="none", log_target=True)
            kl = torch.lerp(kl_s, kl_t, alpha)
        return kl.sum(dim=-1)

    def test_jsd_alpha_05(self):
        ours = top_k_kl_divergence(self.student_lp, self.teacher_lp, alpha=0.5, add_tail=True)
        ref = self._ref_kl(self.student_lp, self.teacher_lp, alpha=0.5, add_tail=True)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_forward_kl_alpha_0(self):
        ours = top_k_kl_divergence(self.student_lp, self.teacher_lp, alpha=0.0, add_tail=True)
        ref = self._ref_kl(self.student_lp, self.teacher_lp, alpha=0.0, add_tail=True)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_reverse_kl_alpha_1(self):
        ours = top_k_kl_divergence(self.student_lp, self.teacher_lp, alpha=1.0, add_tail=True)
        ref = self._ref_kl(self.student_lp, self.teacher_lp, alpha=1.0, add_tail=True)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_asymmetric_alpha(self):
        for alpha in [0.1, 0.3, 0.7, 0.9]:
            ours = top_k_kl_divergence(self.student_lp, self.teacher_lp, alpha=alpha, add_tail=True)
            ref = self._ref_kl(self.student_lp, self.teacher_lp, alpha=alpha, add_tail=True)
            torch.testing.assert_close(ours, ref, atol=1e-8, rtol=1e-6, msg=f"Failed for alpha={alpha}")

    def test_no_tail_renorm(self):
        """When add_tail=False, both implementations renormalize top-K log-probs."""
        ours = top_k_kl_divergence(self.student_lp, self.teacher_lp, alpha=0.5, add_tail=False)
        ref = self._ref_kl(self.student_lp, self.teacher_lp, alpha=0.5, add_tail=False)
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Tests: Importance sampling correction
# ---------------------------------------------------------------------------


class TestISCorrectionMatch:
    """Compare our apply_importance_sampling_correction against verl reference."""

    def test_basic_match(self):
        B, T = 4, 16
        loss = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        student_lp = _make_per_token_log_probs(B, T)
        old_lp = _make_per_token_log_probs(B, T)

        # Our implementation
        ours = apply_importance_sampling_correction(loss, student_lp, old_lp, is_clip=2.0)

        # Reference (inline from verl)
        negative_approx_kl = (student_lp - old_lp).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=2.0)
        ref = loss * ratio

        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_none_clip_passthrough(self):
        B, T = 2, 8
        loss = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        student_lp = _make_per_token_log_probs(B, T)
        old_lp = _make_per_token_log_probs(B, T)
        ours = apply_importance_sampling_correction(loss, student_lp, old_lp, is_clip=None)
        torch.testing.assert_close(ours, loss)

    def test_extreme_ratios_clamped(self):
        """When student >> old, ratio should be clamped at is_clip."""
        B, T = 2, 4
        loss = torch.ones(B, T, dtype=DTYPE, device=DEVICE)
        student_lp = torch.zeros(B, T, dtype=DTYPE, device=DEVICE)  # log(1) = 0
        old_lp = torch.full((B, T), -10.0, dtype=DTYPE, device=DEVICE)  # very unlikely under old

        ours = apply_importance_sampling_correction(loss, student_lp, old_lp, is_clip=2.0)
        # ratio = exp(0 - (-10)) = exp(10) >> 2, clamped to 2.0
        expected = loss * 2.0
        torch.testing.assert_close(ours, expected, atol=ATOL, rtol=RTOL)

    def test_various_clips(self):
        B, T = 4, 16
        loss = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        student_lp = _make_per_token_log_probs(B, T)
        old_lp = _make_per_token_log_probs(B, T)

        for clip in [1.0, 1.5, 3.0, 5.0, 10.0]:
            ours = apply_importance_sampling_correction(loss, student_lp, old_lp, is_clip=clip)
            neg_kl = (student_lp - old_lp).detach().clamp(-20, 20)
            ref = loss * torch.exp(neg_kl).clamp(max=clip)
            torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL, msg=f"Failed for clip={clip}")


# ---------------------------------------------------------------------------
# Tests: Loss aggregation
# ---------------------------------------------------------------------------


class TestAggregateLossMatch:
    """Compare our aggregate_loss against verl's agg_loss (token-mean mode)."""

    def test_without_distillation_mask(self):
        B, T = 4, 16
        loss_mat = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        mask = _make_response_mask(B, T)

        ours = aggregate_loss(loss_mat, mask)

        # Reference: sum(loss * mask) / sum(mask).clamp(1) * dp_size
        # dp_size=1, batch_num_tokens=mask.sum().clamp(1)
        ref = _ref_agg_loss(loss_mat, mask, batch_num_tokens=mask.sum().clamp(min=1.0))
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_with_distillation_mask(self):
        B, T = 4, 16
        loss_mat = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        response_mask = _make_response_mask(B, T)
        sd_mask = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE)

        ours = aggregate_loss(loss_mat, response_mask, sd_mask)

        # Reference: loss_mask = response_mask * sd_mask.unsqueeze(1)
        combined = response_mask * sd_mask.unsqueeze(1)
        ref = _ref_agg_loss(loss_mat, combined, batch_num_tokens=combined.sum().clamp(min=1.0))
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)

    def test_all_masked_returns_zero(self):
        B, T = 2, 8
        loss_mat = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        mask = torch.zeros(B, T, dtype=DTYPE, device=DEVICE)

        ours = aggregate_loss(loss_mat, mask)
        ref = _ref_agg_loss(loss_mat, mask, batch_num_tokens=mask.sum().clamp(min=1.0))
        torch.testing.assert_close(ours, ref, atol=ATOL, rtol=RTOL)
        assert ours.item() == 0.0


# ---------------------------------------------------------------------------
# Tests: Full compute_self_distillation_loss pipeline
# ---------------------------------------------------------------------------


class TestFullPipelineMatch:
    """Compare our compute_self_distillation_loss against the reference end-to-end."""

    def _run(self, alpha=0.5, is_clip=2.0, add_tail=True, use_sd_mask=True, use_rollout_weights=False):
        B, T, K = 4, 16, 10
        student_topk = _make_topk_log_probs(B, T, K)
        teacher_topk = _make_topk_log_probs(B, T, K)
        student_lp = _make_per_token_log_probs(B, T)
        teacher_lp = _make_per_token_log_probs(B, T)
        old_lp = _make_per_token_log_probs(B, T) if is_clip is not None else None
        mask = _make_response_mask(B, T)
        sd_mask = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=DTYPE, device=DEVICE) if use_sd_mask else None
        rollout_w = torch.rand(B, T, dtype=DTYPE, device=DEVICE) + 0.5 if use_rollout_weights else None

        # Our implementation
        our_loss, our_metrics = compute_self_distillation_loss(
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
            alpha=alpha,
            is_clip=is_clip,
            add_tail=add_tail,
            old_log_probs=old_lp,
            self_distillation_mask=sd_mask,
            rollout_is_weights=rollout_w,
        )

        # Reference implementation
        config = _FakeConfig(
            full_logit_distillation=True,
            distillation_topk=K,
            distillation_add_tail=add_tail,
            alpha=alpha,
            is_clip=is_clip,
        )
        ref_loss, _ = _ref_compute_self_distillation_loss(
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            config=config,
            old_log_probs=old_lp,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
            self_distillation_mask=sd_mask,
            rollout_is_weights=rollout_w,
        )

        return our_loss, ref_loss

    def test_default_jsd(self):
        our, ref = self._run(alpha=0.5, is_clip=2.0, add_tail=True)
        torch.testing.assert_close(our, ref, atol=1e-8, rtol=1e-6)

    def test_forward_kl(self):
        our, ref = self._run(alpha=0.0, is_clip=2.0, add_tail=True)
        torch.testing.assert_close(our, ref, atol=1e-8, rtol=1e-6)

    def test_reverse_kl(self):
        our, ref = self._run(alpha=1.0, is_clip=2.0, add_tail=True)
        torch.testing.assert_close(our, ref, atol=1e-8, rtol=1e-6)

    def test_no_is_correction(self):
        our, ref = self._run(alpha=0.5, is_clip=None, add_tail=True)
        torch.testing.assert_close(our, ref, atol=1e-8, rtol=1e-6)

    def test_no_sd_mask(self):
        our, ref = self._run(alpha=0.5, is_clip=2.0, use_sd_mask=False)
        torch.testing.assert_close(our, ref, atol=1e-8, rtol=1e-6)

    def test_with_rollout_weights(self):
        our, ref = self._run(alpha=0.5, is_clip=2.0, use_rollout_weights=True)
        torch.testing.assert_close(our, ref, atol=1e-8, rtol=1e-6)

    def test_non_full_logit_reverse_kl(self):
        """Non-topk path: REINFORCE-style reverse KL."""
        B, T = 4, 16
        student_lp = _make_per_token_log_probs(B, T)
        teacher_lp = _make_per_token_log_probs(B, T)
        old_lp = _make_per_token_log_probs(B, T)
        mask = _make_response_mask(B, T)
        sd_mask = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=DTYPE, device=DEVICE)

        our_loss, _ = compute_self_distillation_loss(
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            student_topk_log_probs=None,
            teacher_topk_log_probs=None,
            alpha=1.0,
            is_clip=2.0,
            old_log_probs=old_lp,
            self_distillation_mask=sd_mask,
        )

        config = _FakeConfig(
            full_logit_distillation=False,
            distillation_topk=None,
            alpha=1.0,
            is_clip=2.0,
        )
        ref_loss, _ = _ref_compute_self_distillation_loss(
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            config=config,
            old_log_probs=old_lp,
            self_distillation_mask=sd_mask,
        )
        torch.testing.assert_close(our_loss, ref_loss, atol=1e-8, rtol=1e-6)

    def test_all_masked_gives_zero(self):
        """When all samples are masked, both implementations should return 0."""
        B, T, K = 4, 16, 10
        student_topk = _make_topk_log_probs(B, T, K)
        teacher_topk = _make_topk_log_probs(B, T, K)
        student_lp = _make_per_token_log_probs(B, T)
        teacher_lp = _make_per_token_log_probs(B, T)
        old_lp = _make_per_token_log_probs(B, T)
        mask = _make_response_mask(B, T)
        sd_mask = torch.zeros(B, dtype=DTYPE, device=DEVICE)

        our_loss, _ = compute_self_distillation_loss(
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
            alpha=0.5,
            is_clip=2.0,
            add_tail=True,
            old_log_probs=old_lp,
            self_distillation_mask=sd_mask,
        )

        config = _FakeConfig(alpha=0.5, is_clip=2.0)
        ref_loss, _ = _ref_compute_self_distillation_loss(
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            config=config,
            old_log_probs=old_lp,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
            self_distillation_mask=sd_mask,
        )

        torch.testing.assert_close(our_loss, ref_loss, atol=ATOL, rtol=RTOL)
        assert our_loss.item() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests: Reprompting logic vs verl behavior
# ---------------------------------------------------------------------------


class TestRepromptingLogicMatch:
    """
    Test our reprompting module against expected verl behavior.
    We can't import verl's code, but we test the behavioral contracts.
    """

    def test_select_demonstration_first_successful(self):
        """verl takes solution_idxs[0] — the first successful peer."""
        uids = ["a", "a", "a", "a"]
        rewards = [0.5, 1.0, 1.0, 0.0]
        completions = ["bad", "good1", "good2", "bad2"]
        # For sample 0, the first successful peer (excluding self) is idx 1
        result = select_demonstration(0, uids, rewards, completions, success_threshold=1.0, exclude_self=True)
        assert result == "good1"

    def test_select_demonstration_self_exclusion(self):
        """When exclude_self=True and only self is successful, return None."""
        uids = ["a", "a", "a"]
        rewards = [1.0, 0.5, 0.5]
        completions = ["mine", "bad1", "bad2"]
        result = select_demonstration(0, uids, rewards, completions, success_threshold=1.0, exclude_self=True)
        assert result is None

    def test_remove_thinking_tags(self):
        """Matches verl's _remove_thinking_trace."""
        text = "<think>internal reasoning\nmore reasoning</think>\nActual answer"
        assert remove_thinking_tags(text) == "Actual answer"

    def test_mask_with_feedback_only_without_solution(self):
        """
        Verl computes: feedback_used = has_feedback and (not flag or not has_solution)
        When flag=True and has_solution=True, feedback is NOT used.
        """
        solutions = ["demo", "demo", None, None]
        feedback = ["fb1", None, "fb2", None]

        # flag=False: feedback always counts
        mask_no_flag = compute_self_distillation_mask(solutions, feedback, feedback_only_without_solution=False)
        assert mask_no_flag == [1.0, 1.0, 1.0, 0.0]

        # flag=True: feedback only counts when no solution
        mask_flag = compute_self_distillation_mask(solutions, feedback, feedback_only_without_solution=True)
        # sample 0: has_solution=True, feedback_used=False (flag blocks it), but has_solution -> mask=1
        # sample 1: has_solution=True, no feedback -> mask=1
        # sample 2: no solution, feedback_used=True -> mask=1
        # sample 3: nothing -> mask=0
        assert mask_flag == [1.0, 1.0, 1.0, 0.0]

    def test_build_teacher_prompts_structure(self):
        """Verify template assembly matches verl's _build_teacher_message."""
        prompts = ["What is 2+2?"]
        solutions = ["4"]
        feedback = ["Your calculation was wrong."]

        result = build_teacher_prompts(prompts, solutions, feedback)
        # Should contain: prompt, solution section, feedback section, "Correctly solve"
        assert "What is 2+2?" in result[0]
        assert "Correct solution:" in result[0]
        assert "4" in result[0]
        assert "feedback from your unsuccessful earlier attempt" in result[0]
        assert "Your calculation was wrong." in result[0]
        assert "Correctly solve the original question." in result[0]

    def test_no_signal_passthrough(self):
        """When neither solution nor feedback exists, return raw prompt."""
        prompts = ["What is 2+2?"]
        solutions = [None]
        feedback = [None]
        result = build_teacher_prompts(prompts, solutions, feedback)
        assert result[0] == "What is 2+2?"


# ---------------------------------------------------------------------------
# Tests: Edge cases and numerical stability
# ---------------------------------------------------------------------------


class TestNumericalEdgeCases:
    """Test edge cases that could cause numerical issues."""

    def test_identical_distributions_give_zero_kl(self):
        """JSD(P, P) should be 0."""
        lp = _make_topk_log_probs(2, 8, 10)
        kl = top_k_kl_divergence(lp, lp, alpha=0.5, add_tail=True)
        assert kl.abs().max().item() < 1e-6

    def test_very_peaked_distribution(self):
        """One token has ~100% probability."""
        B, T, K = 2, 4, 5
        lp = torch.full((B, T, K), -30.0, dtype=DTYPE, device=DEVICE)
        lp[:, :, 0] = -1e-8  # almost log(1)
        # Should not produce NaN or Inf
        result = add_tail_bucket(lp)
        assert torch.isfinite(result).all()

    def test_uniform_distribution(self):
        """All K tokens have equal probability."""
        K = 10
        lp = torch.full((2, 4, K), torch.tensor(1.0 / K).log().item(), dtype=DTYPE, device=DEVICE)
        result = add_tail_bucket(lp)
        assert torch.isfinite(result).all()

    def test_is_correction_with_identical_policies(self):
        """When student == old, ratio should be 1.0 (no correction)."""
        B, T = 2, 8
        loss = torch.randn(B, T, dtype=DTYPE, device=DEVICE).abs()
        lp = _make_per_token_log_probs(B, T)
        corrected = apply_importance_sampling_correction(loss, lp, lp, is_clip=2.0)
        torch.testing.assert_close(corrected, loss, atol=ATOL, rtol=RTOL)
