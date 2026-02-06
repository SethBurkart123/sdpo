"""
SDPO self-distillation loss computation.

Faithful reimplementation of compute_self_distillation_loss from
verl/trainer/ppo/core_algos.py in the lasgroup/SDPO repository.

Key design decisions matching the reference:
- Top-K KL uses STUDENT's top-K indices for BOTH student and teacher.
- Tail bucket uses log1mexp via clamp(logsumexp, max=-1e-7) for numerical stability.
- JSD (alpha=0.5) is the default KL mode, implemented as generalized Jensen-Shannon.
- IS correction clamps log(student/old) to [-20, 20] then exp and clamps at is_clip.
- self_distillation_mask zeros out samples without teacher signal.
- Loss aggregation is token-mean: sum(loss*mask) / sum(mask).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def add_tail_bucket(topk_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Append a tail probability bucket to top-K log probabilities.

    Given log-probs for the top-K tokens, computes the residual probability mass
    (1 - sum(exp(topk))) and appends it as a (K+1)th entry. This allows computing
    KL divergence that accounts for the full distribution.

    Uses the verl reference formula:
        log_s = logsumexp(topk_log_probs)
        log_s = clamp(log_s, max=-1e-7)   # prevent log(0)
        tail  = log(-expm1(log_s))         # = log(1 - exp(log_s))

    Args:
        topk_log_probs: (B, T, K) log-probabilities of the top-K tokens.

    Returns:
        (B, T, K+1) log-probabilities with tail bucket appended.
    """
    log_s = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    # Clamp to prevent log_s from reaching 0 (which would make tail = log(0) = -inf)
    # -1e-7 means the top-K probs sum to at most exp(-1e-7) ≈ 0.9999999
    log_s = torch.clamp(log_s, max=-1e-7)
    # log(1 - exp(log_s)) via the numerically stable expm1 path
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([topk_log_probs, tail_log], dim=-1)


def top_k_kl_divergence(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    alpha: float = 0.5,
    add_tail: bool = True,
) -> torch.Tensor:
    """
    Compute (approximate) KL divergence over top-K logits with optional tail bucket.

    The alpha parameter controls the KL variant:
        alpha=0.0 -> Forward KL: KL(teacher || student)
        alpha=1.0 -> Reverse KL: KL(student || teacher)
        alpha=0.5 -> Generalized Jensen-Shannon Divergence (symmetric)

    IMPORTANT: Both student and teacher log-probs must be indexed at the SAME
    vocabulary positions (the student's top-K indices). This matches the verl
    reference where `topk_indices=student_topk_indices` is passed to the teacher.

    Args:
        student_topk_log_probs: (B, T, K) student log-probs at top-K positions.
        teacher_topk_log_probs: (B, T, K) teacher log-probs at those SAME positions.
        alpha: KL interpolation coefficient.
        add_tail: Whether to append a tail bucket before computing KL.

    Returns:
        (B, T) per-token KL divergence (summed over the K or K+1 dimension).
    """
    if add_tail:
        student_lp = add_tail_bucket(student_topk_log_probs)
        teacher_lp = add_tail_bucket(teacher_topk_log_probs)
    else:
        student_lp = student_topk_log_probs
        teacher_lp = teacher_topk_log_probs

    if alpha == 0.0:
        # Forward KL: KL(teacher || student) = sum(teacher.exp() * (teacher - student))
        kl = F.kl_div(student_lp, teacher_lp, reduction="none", log_target=True)
    elif alpha == 1.0:
        # Reverse KL: KL(student || teacher) = sum(student.exp() * (student - teacher))
        kl = F.kl_div(teacher_lp, student_lp, reduction="none", log_target=True)
    else:
        # Generalized Jensen-Shannon Divergence
        # M = alpha * teacher + (1-alpha) * student (in probability space)
        # JSD = alpha * KL(M || teacher) + (1-alpha) * KL(M || student)
        # Computed in log-space via logsumexp for numerical stability.
        alpha_t = torch.tensor(alpha, dtype=student_lp.dtype, device=student_lp.device)
        mixture_lp = torch.logsumexp(
            torch.stack(
                [
                    student_lp + torch.log(1.0 - alpha_t),
                    teacher_lp + torch.log(alpha_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_lp, teacher_lp, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_lp, student_lp, reduction="none", log_target=True)
        kl = torch.lerp(kl_student, kl_teacher, alpha)

    # Sum over the vocabulary (K or K+1) dimension -> per-token KL
    return kl.sum(dim=-1)


def apply_importance_sampling_correction(
    per_token_loss: torch.Tensor,
    student_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    is_clip: float | None = 2.0,
) -> torch.Tensor:
    """
    Apply truncated importance sampling correction for off-policy training.

    When the student policy has drifted from the rollout policy (old), we weight
    each token's loss by min(student/old, is_clip) to correct for the distribution shift.

    Matches the verl reference:
        negative_approx_kl = clamp(student - old, -20, 20)
        ratio = exp(negative_approx_kl).clamp(max=is_clip)
        loss = loss * ratio

    Args:
        per_token_loss: (B, T) loss values to correct.
        student_log_probs: (B, T) current policy log-probs on sampled tokens.
        old_log_probs: (B, T) rollout policy log-probs on sampled tokens.
        is_clip: Upper bound for the IS ratio. None disables correction.

    Returns:
        (B, T) corrected loss values.
    """
    if is_clip is None:
        return per_token_loss

    log_ratio = (student_log_probs - old_log_probs).detach()
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio).clamp(max=is_clip)
    return per_token_loss * ratio


def aggregate_loss(
    loss_mat: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Token-mean loss aggregation matching verl's agg_loss with mode="token-mean".

    loss = sum(loss_mat * effective_mask) / sum(effective_mask)

    The self_distillation_mask is (B,) and zeros out entire samples that lack
    teacher signal (no successful peer + no environment feedback).

    Args:
        loss_mat: (B, T) per-token loss values.
        response_mask: (B, T) binary mask for valid completion tokens.
        self_distillation_mask: (B,) binary mask for samples with teacher signal. None means all.

    Returns:
        Scalar loss.
    """
    effective_mask = response_mask
    if self_distillation_mask is not None:
        effective_mask = effective_mask * self_distillation_mask.unsqueeze(1)

    numerator = (loss_mat * effective_mask).sum()
    denominator = effective_mask.sum().clamp(min=1.0)
    return numerator / denominator


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    student_topk_log_probs: torch.Tensor | None = None,
    teacher_topk_log_probs: torch.Tensor | None = None,
    alpha: float = 0.5,
    is_clip: float | None = 2.0,
    add_tail: bool = True,
    old_log_probs: torch.Tensor | None = None,
    self_distillation_mask: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the full SDPO self-distillation loss.

    This is the main entry point, matching verl's compute_self_distillation_loss.

    When student_topk_log_probs and teacher_topk_log_probs are provided (the normal
    case for full_logit_distillation=True), the loss is the top-K KL divergence
    between the student and teacher distributions.

    Args:
        student_log_probs: (B, T) per-token log-probs of student on sampled tokens.
        teacher_log_probs: (B, T) per-token log-probs of teacher on sampled tokens.
        response_mask: (B, T) mask for valid completion tokens.
        student_topk_log_probs: (B, T, K) student top-K log-probs (full logit mode).
        teacher_topk_log_probs: (B, T, K) teacher log-probs at student's top-K indices.
        alpha: KL interpolation. 0=forward, 1=reverse, 0.5=JSD.
        is_clip: Importance sampling clip value. None to disable.
        add_tail: Whether to add tail bucket for top-K KL.
        old_log_probs: (B, T) rollout policy log-probs. None means on-policy.
        self_distillation_mask: (B,) mask for samples with teacher signal.
        rollout_is_weights: (B, T) additional rollout IS correction weights.

    Returns:
        (loss, metrics) where loss is a scalar and metrics is a dict of logged values.
    """
    use_full_logit = student_topk_log_probs is not None and teacher_topk_log_probs is not None

    if use_full_logit:
        # Full-logit distillation: top-K KL divergence
        per_token_kl = top_k_kl_divergence(
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_topk_log_probs,
            alpha=alpha,
            add_tail=add_tail,
        )
    else:
        # Token-level reverse KL (fallback when full logits unavailable)
        # verl: log_ratio.detach() * student_log_probs (REINFORCE-style)
        log_ratio = student_log_probs - teacher_log_probs
        per_token_kl = log_ratio.detach() * student_log_probs

    per_token_loss = per_token_kl

    # Importance sampling correction
    if old_log_probs is not None and is_clip is not None:
        per_token_loss = apply_importance_sampling_correction(
            per_token_loss=per_token_loss,
            student_log_probs=student_log_probs,
            old_log_probs=old_log_probs,
            is_clip=is_clip,
        )

    # Additional rollout IS weights (from trainer-level correction)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    # Aggregate with masks
    loss = aggregate_loss(per_token_loss, response_mask, self_distillation_mask)

    # Metrics
    with torch.no_grad():
        effective_mask = response_mask
        if self_distillation_mask is not None:
            effective_mask = effective_mask * self_distillation_mask.unsqueeze(1)
        mask_sum = effective_mask.sum().clamp(min=1.0)

        metrics = {
            "sdpo/kl_mean": (per_token_kl * effective_mask).sum().item() / mask_sum.item(),
            "sdpo/loss": loss.item(),
            "sdpo/teacher_coverage": (
                self_distillation_mask.mean().item() if self_distillation_mask is not None else 1.0
            ),
        }

    return loss, metrics
