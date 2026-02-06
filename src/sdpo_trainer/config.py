"""
SDPO configuration dataclass.

Defaults match the experiment scripts from lasgroup/SDPO (run_local_sdpo.sh),
which produce the paper's reported results. Note: the verl actor.yaml has
different defaults for some fields — we use the experiment values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sdpo_trainer.reprompting import (
    DEFAULT_FEEDBACK_TEMPLATE,
    DEFAULT_REPROMPT_TEMPLATE,
    DEFAULT_SOLUTION_TEMPLATE,
)


@dataclass
class SDPOConfig:
    """
    Configuration for SDPO self-distillation.

    These parameters control the self-distillation loss, teacher management,
    and reprompting behavior. They are passed alongside TRL's GRPOConfig
    when constructing the SDPOTrainer.

    Defaults match the experiment scripts from the paper (run_local_sdpo.sh),
    NOT the yaml defaults in verl/workers/config/actor.py.
    """

    # --- Loss mode ---
    # When True, SDPO replaces the GRPO clip loss entirely with the
    # self-distillation KL loss. This matches verl's loss_mode="sdpo".
    enabled: bool = True

    # --- KL divergence ---
    # Interpolation for KL variant: 0.0=forward KL(teacher||student),
    # 1.0=reverse KL(student||teacher), 0.5=Jensen-Shannon (symmetric).
    # Paper experiments use 0.5 (JSD). verl actor.yaml defaults to 0.0.
    alpha: float = 0.5

    # Use top-K logits for approximate KL instead of full vocabulary.
    # K=100 captures >99% of probability mass per the paper.
    full_logit_distillation: bool = True
    distillation_topk: int = 100
    distillation_add_tail: bool = True

    # --- Importance sampling correction ---
    # Clamps IS ratio at this value. None disables IS correction.
    is_clip: float | None = 2.0

    # --- Teacher management ---
    # "ema": Exponential moving average of student weights (default).
    # "trust_region": Interpolate logits between frozen ref and current student.
    # "frozen": Use the initial model as teacher (no updates).
    teacher_mode: str = "ema"
    # EMA update rate: teacher = (1-rate)*teacher + rate*student.
    teacher_update_rate: float = 0.05

    # --- Demonstration selection ---
    # Minimum scalar reward for a rollout to be considered "successful"
    # and eligible as a demonstration for peers.
    success_reward_threshold: float = 1.0
    # If True, a sample's own successful response is NOT used as its demo.
    # Paper experiments use True. verl actor.yaml defaults to False.
    dont_reprompt_on_self_success: bool = True
    # Strip <think>...</think> blocks from demonstration text.
    # Paper experiments use True. verl actor.yaml defaults to False.
    remove_thinking_from_demonstration: bool = True

    # --- Feedback ---
    # Whether to include environment feedback (test errors, etc.) in the teacher prompt.
    # Paper experiments use True. verl actor.yaml defaults to False.
    include_environment_feedback: bool = True
    # If True, only include feedback when no successful solution is available.
    environment_feedback_only_without_solution: bool = True

    # --- Reprompting ---
    max_reprompt_length: int = 10240
    reprompt_truncation: str = "right"

    # --- Templates ---
    reprompt_template: str = DEFAULT_REPROMPT_TEMPLATE
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE
    feedback_template: str = DEFAULT_FEEDBACK_TEMPLATE

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.teacher_mode not in ("ema", "trust_region", "frozen"):
            raise ValueError(f"teacher_mode must be 'ema', 'trust_region', or 'frozen', got '{self.teacher_mode}'")
        if self.reprompt_truncation not in ("left", "right", "error"):
            raise ValueError(
                f"reprompt_truncation must be 'left', 'right', or 'error', got '{self.reprompt_truncation}'"
            )
        if self.distillation_topk < 1:
            raise ValueError(f"distillation_topk must be >= 1, got {self.distillation_topk}")
