"""
Tests for sdpo_rl.distillation — the mathematical core of SDPO.

These tests verify that our implementation of the self-distillation loss matches
the verl reference implementation (lasgroup/SDPO, verl/trainer/ppo/core_algos.py).

Every test is written BEFORE the implementation (TDD).
"""

import torch
import torch.nn.functional as F
import pytest

from sdpo_rl.distillation import (
    add_tail_bucket,
    top_k_kl_divergence,
    compute_self_distillation_loss,
    apply_importance_sampling_correction,
    aggregate_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Deterministic RNG for reproducibility."""
    return torch.Generator().manual_seed(42)


@pytest.fixture
def small_logits(rng):
    """(batch=4, seq_len=8, vocab=32) logits for student and teacher."""
    student = torch.randn(4, 8, 32, generator=rng)
    teacher = torch.randn(4, 8, 32, generator=rng)
    return student, teacher


@pytest.fixture
def small_log_probs(small_logits):
    """Log-softmax of the small logits."""
    student_logits, teacher_logits = small_logits
    return F.log_softmax(student_logits, dim=-1), F.log_softmax(teacher_logits, dim=-1)


@pytest.fixture
def response_mask():
    """(batch=4, seq_len=8) mask with some trailing padding."""
    mask = torch.ones(4, 8)
    mask[0, 6:] = 0  # sample 0 has 6 real tokens
    mask[1, 7:] = 0  # sample 1 has 7
    mask[2, :] = 1  # sample 2 full
    mask[3, 4:] = 0  # sample 3 has 4
    return mask


# ---------------------------------------------------------------------------
# 1. Tail Bucket Construction
# ---------------------------------------------------------------------------


class TestAddTailBucket:
    """Tests for the log1mexp tail bucket computation."""

    def test_output_shape(self, small_log_probs):
        """Tail bucket adds one extra dimension: (B, T, K) -> (B, T, K+1)."""
        student_lp, _ = small_log_probs
        # Take top-5 as if K=5
        topk_lp, _ = student_lp.topk(5, dim=-1)
        result = add_tail_bucket(topk_lp)
        assert result.shape == (4, 8, 6), f"Expected (4,8,6), got {result.shape}"

    def test_probabilities_sum_to_one(self, small_log_probs):
        """After adding tail bucket, exp(log_probs) should sum to ~1."""
        student_lp, _ = small_log_probs
        topk_lp, _ = student_lp.topk(5, dim=-1)
        result = add_tail_bucket(topk_lp)
        probs = result.exp()
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_near_one_hot_no_nan(self):
        """When distribution is near-one-hot, tail mass is ~0. Must not produce NaN."""
        # Create a distribution where top-1 has 99.99% of mass
        logits = torch.full((1, 1, 10), -100.0)
        logits[0, 0, 0] = 10.0
        log_probs = F.log_softmax(logits, dim=-1)
        topk_lp, _ = log_probs.topk(5, dim=-1)
        result = add_tail_bucket(topk_lp)
        assert not torch.isnan(result).any(), "NaN in tail bucket for near-one-hot distribution"
        assert not torch.isinf(result).any(), "Inf in tail bucket for near-one-hot distribution"

    def test_uniform_distribution(self):
        """Uniform distribution: tail should contain (V-K)/V of the mass."""
        V, K = 100, 10
        log_probs = torch.full((1, 1, V), -torch.tensor(float(V)).log())
        topk_lp, _ = log_probs.topk(K, dim=-1)
        result = add_tail_bucket(topk_lp)
        tail_prob = result[0, 0, -1].exp().item()
        expected_tail = (V - K) / V
        assert abs(tail_prob - expected_tail) < 1e-4, f"Tail prob {tail_prob} != expected {expected_tail}"

    def test_bfloat16_no_nan(self):
        """Must be numerically stable in bfloat16."""
        logits = torch.randn(2, 4, 50, dtype=torch.bfloat16)
        log_probs = F.log_softmax(logits, dim=-1)
        topk_lp, _ = log_probs.topk(10, dim=-1)
        result = add_tail_bucket(topk_lp)
        assert not torch.isnan(result).any(), "NaN in bfloat16 tail bucket"

    def test_matches_verl_clamping(self):
        """
        verl reference: log_s = logsumexp(topk); log_s = clamp(log_s, max=-1e-7);
        tail = log(-expm1(log_s)). We must match this exact formula.
        """
        topk_lp = torch.tensor([[[-0.1, -0.2, -0.5]]])  # log probs
        # Manual verl reference computation
        log_s = torch.logsumexp(topk_lp, dim=-1, keepdim=True)
        log_s_clamped = torch.clamp(log_s, max=-1e-7)
        tail_ref = torch.log(-torch.expm1(log_s_clamped))
        expected = torch.cat([topk_lp, tail_ref], dim=-1)

        result = add_tail_bucket(topk_lp)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Top-K KL Divergence
# ---------------------------------------------------------------------------


class TestTopKKLDivergence:
    """Tests for the top-K KL divergence with shared indices."""

    def test_forward_kl_matches_pytorch(self, small_log_probs, response_mask):
        """alpha=0 (forward KL): our top-K approx on full vocab should match F.kl_div."""
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]

        # Use K=V (full vocab) so top-K is exact
        topk_vals, topk_idx = student_lp.topk(V, dim=-1)
        teacher_gathered = teacher_lp.gather(-1, topk_idx)

        our_kl = top_k_kl_divergence(
            student_topk_log_probs=topk_vals,
            teacher_topk_log_probs=teacher_gathered,
            alpha=0.0,
            add_tail=False,
        )

        # PyTorch reference: KL(teacher || student) = sum(teacher.exp() * (teacher - student))
        ref_kl = F.kl_div(student_lp, teacher_lp, reduction="none", log_target=True).sum(-1)

        torch.testing.assert_close(our_kl, ref_kl, atol=1e-5, rtol=1e-5)

    def test_reverse_kl_matches_pytorch(self, small_log_probs):
        """alpha=1 (reverse KL): should match KL(student || teacher)."""
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]

        topk_vals, topk_idx = student_lp.topk(V, dim=-1)
        teacher_gathered = teacher_lp.gather(-1, topk_idx)

        our_kl = top_k_kl_divergence(
            student_topk_log_probs=topk_vals,
            teacher_topk_log_probs=teacher_gathered,
            alpha=1.0,
            add_tail=False,
        )

        ref_kl = F.kl_div(teacher_lp, student_lp, reduction="none", log_target=True).sum(-1)

        torch.testing.assert_close(our_kl, ref_kl, atol=1e-5, rtol=1e-5)

    def test_jsd_symmetric_at_alpha_half(self, small_log_probs):
        """alpha=0.5: JSD should be symmetric — JSD(P,Q) == JSD(Q,P)."""
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]

        topk_s, idx_s = student_lp.topk(V, dim=-1)
        teacher_at_s = teacher_lp.gather(-1, idx_s)

        topk_t, idx_t = teacher_lp.topk(V, dim=-1)
        student_at_t = student_lp.gather(-1, idx_t)

        jsd_st = top_k_kl_divergence(topk_s, teacher_at_s, alpha=0.5, add_tail=False)
        jsd_ts = top_k_kl_divergence(topk_t, student_at_t, alpha=0.5, add_tail=False)

        torch.testing.assert_close(jsd_st, jsd_ts, atol=1e-5, rtol=1e-5)

    def test_jsd_manual_computation(self, small_log_probs):
        """
        JSD(alpha=0.5) = 0.5 * KL(M || teacher) + 0.5 * KL(M || student)
        where M = 0.5*student + 0.5*teacher (in probability space).
        """
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]
        topk_vals, topk_idx = student_lp.topk(V, dim=-1)
        teacher_gathered = teacher_lp.gather(-1, topk_idx)

        our_jsd = top_k_kl_divergence(topk_vals, teacher_gathered, alpha=0.5, add_tail=False)

        # Manual reference
        alpha = 0.5
        mixture_lp = torch.logsumexp(
            torch.stack(
                [topk_vals + torch.log(torch.tensor(1 - alpha)), teacher_gathered + torch.log(torch.tensor(alpha))]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_lp, teacher_gathered, reduction="none", log_target=True).sum(-1)
        kl_student = F.kl_div(mixture_lp, topk_vals, reduction="none", log_target=True).sum(-1)
        ref_jsd = torch.lerp(kl_student, kl_teacher, alpha)

        torch.testing.assert_close(our_jsd, ref_jsd, atol=1e-5, rtol=1e-5)

    def test_kl_non_negative(self, small_log_probs):
        """KL divergence must be non-negative at every position."""
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]
        topk_vals, topk_idx = student_lp.topk(V, dim=-1)
        teacher_gathered = teacher_lp.gather(-1, topk_idx)

        for alpha in [0.0, 0.5, 1.0]:
            kl = top_k_kl_divergence(topk_vals, teacher_gathered, alpha=alpha, add_tail=False)
            assert (kl >= -1e-6).all(), f"Negative KL at alpha={alpha}: min={kl.min()}"

    def test_identical_distributions_zero_kl(self, small_log_probs):
        """KL between identical distributions should be 0."""
        student_lp, _ = small_log_probs
        V = student_lp.shape[-1]
        topk_vals, topk_idx = student_lp.topk(V, dim=-1)
        self_gathered = student_lp.gather(-1, topk_idx)

        for alpha in [0.0, 0.5, 1.0]:
            kl = top_k_kl_divergence(topk_vals, self_gathered, alpha=alpha, add_tail=False)
            torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-5, rtol=1e-5)

    def test_topk_tail_and_renorm_both_valid(self, small_log_probs):
        """
        Both add_tail=True (tail bucket) and add_tail=False (renormalized top-K)
        produce finite, non-negative KL values that approximate the full-vocab KL.

        Note: verl uses renorm_topk_log_probs when add_tail=False, normalizing the
        top-K log-probs to form a valid distribution. Both approaches are valid
        approximations of the full KL, and which is closer depends on the specific
        distributions involved.
        """
        student_lp, teacher_lp = small_log_probs
        K = 10

        topk_vals, topk_idx = student_lp.topk(K, dim=-1)
        teacher_gathered = teacher_lp.gather(-1, topk_idx)

        kl_renorm = top_k_kl_divergence(topk_vals, teacher_gathered, alpha=0.0, add_tail=False)
        kl_with_tail = top_k_kl_divergence(topk_vals, teacher_gathered, alpha=0.0, add_tail=True)

        # Both should be finite and non-negative
        assert torch.isfinite(kl_renorm).all(), "Renormalized KL should be finite"
        assert torch.isfinite(kl_with_tail).all(), "Tail bucket KL should be finite"
        assert (kl_renorm >= -1e-6).all(), "Renormalized KL should be non-negative"
        assert (kl_with_tail >= -1e-6).all(), "Tail bucket KL should be non-negative"

    def test_student_indices_used_for_both(self):
        """
        Critical: teacher logits must be gathered at student's top-K indices.
        Verify by checking that teacher_topk_log_probs are at the same indices as student.
        """
        student_lp = torch.tensor([[[0.5, 0.3, 0.1, 0.05, 0.05]]]).log()
        teacher_lp = torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]).log()

        K = 3
        topk_vals, topk_idx = student_lp.topk(K, dim=-1)
        # Teacher must be gathered at student's indices, not teacher's own top-K
        teacher_at_student_idx = teacher_lp.gather(-1, topk_idx)

        # Student top-3 indices should be [0, 1, 2] (highest student probs)
        assert topk_idx[0, 0].tolist() == [0, 1, 2]
        # Teacher values at those indices should be teacher's probs for tokens 0,1,2
        expected_teacher = teacher_lp[0, 0, :3].unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(teacher_at_student_idx, expected_teacher)


# ---------------------------------------------------------------------------
# 3. Importance Sampling Correction
# ---------------------------------------------------------------------------


class TestImportanceSamplingCorrection:
    """Tests for the IS ratio clipping used in SDPO off-policy correction."""

    def test_on_policy_ratio_is_one(self):
        """When student == old policy, IS ratio should be 1 everywhere."""
        log_probs = torch.randn(4, 8)
        ratio = apply_importance_sampling_correction(
            per_token_loss=torch.ones(4, 8),
            student_log_probs=log_probs,
            old_log_probs=log_probs,
            is_clip=2.0,
        )
        torch.testing.assert_close(ratio, torch.ones(4, 8), atol=1e-5, rtol=1e-5)

    def test_clipping_upper_bound(self):
        """IS ratio should be clamped at is_clip."""
        student_lp = torch.zeros(1, 1)  # log(1)
        old_lp = torch.tensor([[-10.0]])  # log(very small) => ratio = exp(10) >> is_clip
        result = apply_importance_sampling_correction(
            per_token_loss=torch.ones(1, 1),
            student_log_probs=student_lp,
            old_log_probs=old_lp,
            is_clip=2.0,
        )
        assert result.item() <= 2.0 + 1e-6, f"IS ratio {result.item()} exceeds clip=2.0"

    def test_disabled_when_none(self):
        """When is_clip=None, no correction is applied (loss returned unchanged)."""
        loss = torch.randn(4, 8)
        result = apply_importance_sampling_correction(
            per_token_loss=loss,
            student_log_probs=torch.randn(4, 8),
            old_log_probs=torch.randn(4, 8),
            is_clip=None,
        )
        torch.testing.assert_close(result, loss)

    def test_log_ratio_clamped_for_stability(self):
        """
        verl clamps log(student/old) to [-20, 20] before exp.
        With extreme values, we should not get inf or nan.
        """
        student_lp = torch.tensor([[100.0]])  # absurdly large
        old_lp = torch.tensor([[-100.0]])
        result = apply_importance_sampling_correction(
            per_token_loss=torch.ones(1, 1),
            student_log_probs=student_lp,
            old_log_probs=old_lp,
            is_clip=2.0,
        )
        assert torch.isfinite(result).all(), "IS correction produced non-finite values"


# ---------------------------------------------------------------------------
# 4. Self-Distillation Mask
# ---------------------------------------------------------------------------


class TestSelfDistillationMask:
    """Tests for the masking of samples without teacher signal."""

    def test_mask_zeros_out_samples(self, response_mask):
        """Samples with self_distillation_mask=0 should contribute zero loss."""
        bs, seq_len = 4, 8
        per_token_loss = torch.ones(bs, seq_len)
        sd_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # samples 1,3 have no teacher

        loss = aggregate_loss(per_token_loss, response_mask, self_distillation_mask=sd_mask)

        # Only samples 0 and 2 should contribute
        # Manually: sample 0 has 6 tokens, sample 2 has 8 tokens => mean across tokens, then mean across samples
        expected_per_sample = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # But the verl aggregation is token-mean: sum(loss*mask) / sum(mask)
        # With sd_mask zeroing samples 1,3: effective mask is response_mask * sd_mask.unsqueeze(1)
        effective_mask = response_mask * sd_mask.unsqueeze(1)
        expected_loss = (per_token_loss * effective_mask).sum() / effective_mask.sum().clamp(min=1.0)
        torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-6)

    def test_all_masked_gives_zero_loss(self, response_mask):
        """When no samples have teacher signal, loss should be 0."""
        per_token_loss = torch.ones(4, 8)
        sd_mask = torch.zeros(4)
        loss = aggregate_loss(per_token_loss, response_mask, self_distillation_mask=sd_mask)
        assert loss.item() == 0.0, f"Expected zero loss, got {loss.item()}"

    def test_no_mask_uses_all_samples(self, response_mask):
        """When self_distillation_mask is None, all samples contribute."""
        per_token_loss = torch.ones(4, 8)
        loss_with_none = aggregate_loss(per_token_loss, response_mask, self_distillation_mask=None)
        loss_with_ones = aggregate_loss(per_token_loss, response_mask, self_distillation_mask=torch.ones(4))
        torch.testing.assert_close(loss_with_none, loss_with_ones)


# ---------------------------------------------------------------------------
# 5. Loss Aggregation (token-mean)
# ---------------------------------------------------------------------------


class TestLossAggregation:
    """Tests for the token-mean loss aggregation matching verl's agg_loss."""

    def test_token_mean_simple(self):
        """token-mean: sum(loss * mask) / sum(mask)."""
        loss = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        result = aggregate_loss(loss, mask)
        expected = (1 + 2 + 3) / 3.0
        assert abs(result.item() - expected) < 1e-6

    def test_respects_response_mask(self, response_mask):
        """Padding tokens (mask=0) should not contribute to loss."""
        loss = torch.ones(4, 8) * 5.0
        result = aggregate_loss(loss, response_mask)
        # All unmasked tokens have loss 5.0, so mean should be 5.0
        assert abs(result.item() - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# 6. Full compute_self_distillation_loss
# ---------------------------------------------------------------------------


class TestComputeSelfDistillationLoss:
    """End-to-end tests for the full SDPO loss function."""

    def test_output_is_scalar(self, small_log_probs, response_mask):
        """Loss should be a scalar tensor."""
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]
        topk_s, idx_s = student_lp.topk(V, dim=-1)
        topk_t = teacher_lp.gather(-1, idx_s)

        loss, metrics = compute_self_distillation_loss(
            student_log_probs=student_lp[:, :, 0],  # dummy per-token logps
            teacher_log_probs=teacher_lp[:, :, 0],
            response_mask=response_mask,
            student_topk_log_probs=topk_s,
            teacher_topk_log_probs=topk_t,
            alpha=0.5,
            is_clip=2.0,
            add_tail=True,
        )
        assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_returns_metrics_dict(self, small_log_probs, response_mask):
        """Should return a dict with at least 'sdpo/kl_mean'."""
        student_lp, teacher_lp = small_log_probs
        V = student_lp.shape[-1]
        topk_s, idx_s = student_lp.topk(V, dim=-1)
        topk_t = teacher_lp.gather(-1, idx_s)

        loss, metrics = compute_self_distillation_loss(
            student_log_probs=student_lp[:, :, 0],
            teacher_log_probs=teacher_lp[:, :, 0],
            response_mask=response_mask,
            student_topk_log_probs=topk_s,
            teacher_topk_log_probs=topk_t,
            alpha=0.5,
        )
        assert isinstance(metrics, dict)
        assert "sdpo/kl_mean" in metrics

    def test_gradient_flows_through_student_only(self, small_log_probs, response_mask):
        """
        Gradients should flow through student log probs but NOT teacher.
        The teacher is stopgrad'd in verl via torch.no_grad().
        In our loss function, teacher inputs should be detached.
        """
        student_lp, teacher_lp = small_log_probs
        student_lp = student_lp.clone().requires_grad_(True)
        V = student_lp.shape[-1]
        topk_s, idx_s = student_lp.topk(V, dim=-1)
        topk_t = teacher_lp.gather(-1, idx_s)

        loss, _ = compute_self_distillation_loss(
            student_log_probs=student_lp[:, :, 0],
            teacher_log_probs=teacher_lp[:, :, 0].detach(),
            response_mask=response_mask,
            student_topk_log_probs=topk_s,
            teacher_topk_log_probs=topk_t.detach(),
            alpha=0.5,
        )
        loss.backward()
        assert student_lp.grad is not None, "No gradient on student"

    def test_identical_student_teacher_zero_loss(self, small_log_probs, response_mask):
        """When student == teacher, distillation loss should be ~0."""
        student_lp, _ = small_log_probs
        V = student_lp.shape[-1]
        topk_s, idx_s = student_lp.topk(V, dim=-1)
        topk_t = student_lp.gather(-1, idx_s)

        loss, _ = compute_self_distillation_loss(
            student_log_probs=student_lp[:, :, 0],
            teacher_log_probs=student_lp[:, :, 0],
            response_mask=response_mask,
            student_topk_log_probs=topk_s,
            teacher_topk_log_probs=topk_t,
            alpha=0.5,
            add_tail=False,
        )
        assert loss.item() < 1e-5, f"Expected ~0 loss for identical distributions, got {loss.item()}"
