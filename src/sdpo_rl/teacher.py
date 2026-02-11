"""
SDPO EMA teacher management.

Faithful reimplementation of the teacher update logic from
verl/workers/actor/dp_actor.py in the lasgroup/SDPO repository.

Key behaviors:
- EMA update: teacher = (1 - rate) * teacher + rate * student
- Update fires once per "training step" (after all num_iterations optimizer steps complete)
- No gradient graph is created during the update (torch.no_grad)
- The teacher model is the SAME object as ref_model (no third model in memory)

lora_ema mode:
- For PEFT/LoRA models, maintains two LoRA adapters on a shared base model
- "default" adapter = student (trainable), "sdpo_teacher" adapter = teacher (EMA'd)
- Saves ~4GB for 7B QLoRA models by not duplicating the quantized base weights
- Same EMA quality as the standard deepcopy approach
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

LORA_EMA_TEACHER_ADAPTER = "sdpo_teacher"


def ema_update(teacher: nn.Module, student: nn.Module, rate: float) -> None:
    """
    Perform an exponential moving average update of teacher weights toward student.

    teacher_param = (1 - rate) * teacher_param + rate * student_param

    Matches verl's DataParallelPPOActor._update_teacher exactly.

    For quantized models (e.g. 4-bit via bitsandbytes + Unsloth), some base
    parameters are stored as uint8 and cannot participate in float arithmetic.
    These are skipped — only floating-point parameters (the LoRA adapters)
    are EMA-updated, which is the correct behavior since the quantized base
    weights are frozen anyway.

    Args:
        teacher: The teacher model whose parameters are updated in-place.
        student: The student model providing the target parameters.
        rate: EMA update rate. 0 = no update, 1 = full copy.
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            if not teacher_param.data.is_floating_point():
                continue
            student_data = student_param.data.to(device=teacher_param.device, dtype=teacher_param.dtype)
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


# ---------------------------------------------------------------------------
# LoRA EMA — Multi-Adapter Teacher (teacher_mode="lora_ema")
# ---------------------------------------------------------------------------


def collect_lora_adapter_pairs(
    model: nn.Module,
    student_adapter: str = "default",
    teacher_adapter: str = LORA_EMA_TEACHER_ADAPTER,
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    """
    Collect matching (teacher_param, student_param) pairs from two LoRA
    adapters on the same PEFT model.

    Iterates over all modules that have a ``lora_A`` ModuleDict (the PEFT
    internal structure for LoRA layers) and pairs up the weights from the
    student and teacher adapters.

    Args:
        model: A PEFT model with at least two LoRA adapters.
        student_adapter: Name of the student adapter (default "default").
        teacher_adapter: Name of the teacher adapter.

    Returns:
        List of (teacher_param, student_param) tuples, covering both
        lora_A and lora_B weights for every LoRA-targeted module.

    Raises:
        ValueError: If the teacher adapter is not found on any LoRA module.
    """
    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for _name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not isinstance(module.lora_A, nn.ModuleDict):
            continue
        if teacher_adapter not in module.lora_A:
            continue
        # lora_A weights
        pairs.append(
            (
                module.lora_A[teacher_adapter].weight,
                module.lora_A[student_adapter].weight,
            )
        )
        # lora_B weights
        pairs.append(
            (
                module.lora_B[teacher_adapter].weight,
                module.lora_B[student_adapter].weight,
            )
        )

    if not pairs:
        raise ValueError(
            f"Teacher adapter '{teacher_adapter}' not found on any LoRA module. "
            f"Available adapters on first LoRA module: check model.peft_config"
        )
    return pairs


def ema_update_lora_adapters(
    model: nn.Module,
    rate: float,
    student_adapter: str = "default",
    teacher_adapter: str = LORA_EMA_TEACHER_ADAPTER,
) -> None:
    """
    Perform an in-place EMA update of teacher adapter weights toward student.

    teacher_param = (1 - rate) * teacher_param + rate * student_param

    Only touches LoRA adapter parameters — base model weights are untouched.
    This is the memory-efficient equivalent of ``ema_update`` for PEFT models.

    Args:
        model: A PEFT model with both student and teacher adapters.
        rate: EMA update rate. 0 = no change, 1 = full copy.
        student_adapter: Name of the student adapter.
        teacher_adapter: Name of the teacher adapter.
    """
    pairs = collect_lora_adapter_pairs(model, student_adapter, teacher_adapter)
    with torch.no_grad():
        for teacher_p, student_p in pairs:
            student_data = student_p.data.to(device=teacher_p.device, dtype=teacher_p.dtype)
            teacher_p.data.mul_(1.0 - rate).add_(student_data, alpha=rate)


def init_lora_ema_teacher(
    model: nn.Module,
    student_adapter: str = "default",
    teacher_adapter: str = LORA_EMA_TEACHER_ADAPTER,
) -> None:
    """
    Initialize the LoRA EMA teacher by adding a second adapter to the PEFT model.

    Creates a new LoRA adapter with the same configuration as the student,
    copies the student's current weights into it, and freezes it. The student
    adapter remains active and trainable.

    Args:
        model: A PEFT model with at least one LoRA adapter.
        student_adapter: Name of the student adapter to clone from.
        teacher_adapter: Name for the new teacher adapter.

    Raises:
        ValueError: If the model is not a PEFT model with LoRA adapters.
    """
    if not hasattr(model, "peft_config"):
        raise ValueError(
            "teacher_mode='lora_ema' requires a PEFT model with LoRA adapters. Got a plain model without peft_config."
        )

    if student_adapter not in model.peft_config:
        raise ValueError(
            f"Student adapter '{student_adapter}' not found in model.peft_config. "
            f"Available: {list(model.peft_config.keys())}"
        )

    adapter_config = model.peft_config[student_adapter]
    model.add_adapter(teacher_adapter, adapter_config)

    # Copy student weights into teacher
    pairs = collect_lora_adapter_pairs(model, student_adapter, teacher_adapter)
    with torch.no_grad():
        for teacher_p, student_p in pairs:
            teacher_p.data.copy_(student_p.data.to(teacher_p.dtype))

    # Freeze teacher adapter
    for teacher_p, _ in pairs:
        teacher_p.requires_grad = False

    # Ensure student adapter is active
    model.set_adapter(student_adapter)

    logger.info(
        "Initialized lora_ema teacher: adapter='%s', %d param pairs, base model weights shared (not duplicated).",
        teacher_adapter,
        len(pairs),
    )


class LoraEMATeacherCallback:
    """
    Manages EMA teacher updates for LoRA adapters.

    Equivalent to ``EMATeacherCallback`` but operates on adapter parameters
    instead of full model parameters. Only the LoRA weights are updated —
    base model weights are never touched.

    Args:
        model: The PEFT model with both student and teacher adapters.
        update_rate: EMA coefficient (default 0.05).
        num_iterations: TRL's num_iterations (optimizer steps per generation).
        student_adapter: Name of the student adapter.
        teacher_adapter: Name of the teacher adapter.
    """

    def __init__(
        self,
        model: nn.Module,
        update_rate: float = 0.05,
        num_iterations: int = 1,
        student_adapter: str = "default",
        teacher_adapter: str = LORA_EMA_TEACHER_ADAPTER,
    ):
        self.model = model
        self.update_rate = update_rate
        self.num_iterations = num_iterations
        self.student_adapter = student_adapter
        self.teacher_adapter = teacher_adapter

    def should_update(self, global_step: int) -> bool:
        """Check if EMA update should fire at this global_step."""
        return global_step % self.num_iterations == 0

    def step(self, global_step: int) -> bool:
        """
        Conditionally perform adapter-level EMA update.

        Returns True if the update was performed.
        """
        if self.should_update(global_step):
            ema_update_lora_adapters(
                self.model,
                rate=self.update_rate,
                student_adapter=self.student_adapter,
                teacher_adapter=self.teacher_adapter,
            )
            return True
        return False
