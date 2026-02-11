"""
sdpo-rl: Self-Distilled Policy Optimization for TRL.
Faithful reimplementation of https://arxiv.org/abs/2601.20802 (lasgroup/SDPO).
"""

__version__ = "0.1.0"

from sdpo_rl.config import SDPOConfig
from sdpo_rl.distillation import (
    add_tail_bucket,
    aggregate_loss,
    apply_importance_sampling_correction,
    compute_self_distillation_loss,
    top_k_kl_divergence,
)
from sdpo_rl.reprompting import (
    build_teacher_prompts,
    compute_self_distillation_mask,
    remove_thinking_tags,
    select_demonstration,
)
from sdpo_rl.teacher import (
    EMATeacherCallback,
    LORA_EMA_TEACHER_ADAPTER,
    LoraEMATeacherCallback,
    ema_update,
    ema_update_lora_adapters,
    init_lora_ema_teacher,
)
from sdpo_rl.trainer import SDPOTrainer

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
    "LORA_EMA_TEACHER_ADAPTER",
    "LoraEMATeacherCallback",
    "ema_update",
    "ema_update_lora_adapters",
    "init_lora_ema_teacher",
]
