"""
sdpo-trainer: Self-Distilled Policy Optimization for TRL.
Faithful reimplementation of https://arxiv.org/abs/2601.20802 (lasgroup/SDPO).
"""

__version__ = "0.1.0"

from sdpo_trainer.config import SDPOConfig
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
from sdpo_trainer.teacher import EMATeacherCallback, ema_update
from sdpo_trainer.trainer import SDPOTrainer

__all__ = [
    "SDPOConfig",
    "SDPOTrainer",
    "add_tail_bucket",
    "aggregate_loss",
    "apply_importance_sampling_correction",
    "compute_self_distillation_loss",
    "top_k_kl_divergence",
    "build_teacher_prompts",
    "compute_self_distillation_mask",
    "remove_thinking_tags",
    "select_demonstration",
    "EMATeacherCallback",
    "ema_update",
]
