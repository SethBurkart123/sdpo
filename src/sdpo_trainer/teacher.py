"""
SDPO EMA teacher management.

Faithful reimplementation of the teacher update logic from
verl/workers/actor/dp_actor.py in the lasgroup/SDPO repository.

Key behaviors:
- EMA update: teacher = (1 - rate) * teacher + rate * student
- Update fires once per "training step" (after all num_iterations optimizer steps complete)
- No gradient graph is created during the update (torch.no_grad)
- The teacher model is the SAME object as ref_model (no third model in memory)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def ema_update(teacher: nn.Module, student: nn.Module, rate: float) -> None:
    """
    Perform an exponential moving average update of teacher weights toward student.

    teacher_param = (1 - rate) * teacher_param + rate * student_param

    Matches verl's DataParallelPPOActor._update_teacher exactly.

    Args:
        teacher: The teacher model whose parameters are updated in-place.
        student: The student model providing the target parameters.
        rate: EMA update rate. 0 = no update, 1 = full copy.
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            student_data = student_param.data.to(device=teacher_param.device)
            teacher_param.data.mul_(1.0 - rate).add_(student_data, alpha=rate)


class EMATeacherCallback:
    """
    Manages EMA teacher updates aligned with TRL's training loop.

    In verl, the EMA update fires once after all PPO epochs on a batch complete.
    In TRL, the equivalent is firing when global_step % num_iterations == 0,
    which marks the boundary between one batch's optimization and the next
    generation call.

    This class is intended to be called from a TrainerCallback.on_step_end hook.

    Args:
        teacher_model: The model to update (typically self.ref_model).
        student_model: The training model providing new weights.
        update_rate: EMA coefficient (default 0.05, matching verl).
        num_iterations: TRL's num_iterations param (how many optimizer steps per generation).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        update_rate: float = 0.05,
        num_iterations: int = 1,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.update_rate = update_rate
        self.num_iterations = num_iterations

    def should_update(self, global_step: int) -> bool:
        """Check if EMA update should fire at this global_step."""
        return global_step % self.num_iterations == 0

    def step(self, global_step: int) -> bool:
        """
        Conditionally perform EMA update.

        Returns True if the update was performed.
        """
        if self.should_update(global_step):
            ema_update(self.teacher_model, self.student_model, self.update_rate)
            return True
        return False
