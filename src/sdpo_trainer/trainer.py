"""
SDPO Trainer - Self-Distilled Policy Optimization for TRL.

Faithful reimplementation of the SDPO algorithm from arxiv:2601.20802 (lasgroup/SDPO)
as a subclass of TRL's GRPOTrainer.

Key design decisions:
- SDPO replaces GRPO loss entirely when enabled (no lambda blending).
- ref_model is repurposed as the EMA teacher (no third model).
- Teacher forward pass uses reprompted input but same response tokens.
- UIDs are derived from RepeatSampler structure: rows [i*G : (i+1)*G] share a prompt.
- Reward functions return list[float]; feedback is captured via a wrapper pattern.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from sdpo_trainer.config import SDPOConfig
from sdpo_trainer.distillation import compute_self_distillation_loss
from sdpo_trainer.reprompting import (
    build_teacher_prompts,
    compute_self_distillation_mask,
    select_demonstration,
)
from sdpo_trainer.teacher import EMATeacherCallback, ema_update

logger = logging.getLogger(__name__)


class _EMAStepCallback(TrainerCallback):
    """
    TRL TrainerCallback that fires the EMA teacher update at the right time.

    In verl, _update_teacher fires ONCE after update_policy completes (after all
    micro-batches and gradient accumulation). We replicate this by hooking into
    on_step_end, which fires once per optimizer step.
    """

    def __init__(self, trainer: SDPOTrainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.ema_callback is not None:
            self.trainer.ema_callback.step(state.global_step)


class SDPOTrainer(GRPOTrainer):
    """
    SDPO trainer extending GRPOTrainer with self-distillation.

    Args:
        model: The policy model to train.
        args: TRL's GRPOConfig with training hyperparameters.
        sdpo_config: SDPO-specific configuration (distillation, reprompting, teacher).
        processing_class: Tokenizer or processor for the model.
        reward_funcs: List of reward functions. Each must return list[float].
            To capture feedback, use a callable that stores feedback on itself
            (e.g. a class with a `last_feedback` attribute).
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        **kwargs: Additional arguments passed to GRPOTrainer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        sdpo_config: SDPOConfig,
        processing_class: PreTrainedTokenizerBase,
        reward_funcs: list[Callable] | list[nn.Module] | None = None,
        train_dataset=None,
        eval_dataset=None,
        **kwargs,
    ):
        self.sdpo_config = sdpo_config

        # Disable TRL's KL penalty — SDPO handles regularization via teacher
        if args.beta != 0.0:
            logger.warning("Overriding beta=%s to 0.0 for SDPO mode.", args.beta)
            args.beta = 0.0

        processing_class.truncation_side = self.sdpo_config.reprompt_truncation

        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        # GRPOTrainer doesn't create ref_model when beta=0, so create it ourselves
        if self.ref_model is None:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
            if hasattr(self, "accelerator"):
                self.ref_model = self.accelerator.prepare(self.ref_model)

        if self.sdpo_config.teacher_mode == "ema":
            self.ema_callback = EMATeacherCallback(
                teacher_model=self.ref_model,
                student_model=self.model,
                update_rate=self.sdpo_config.teacher_update_rate,
                num_iterations=self.args.num_iterations,
            )
            # Register a proper TrainerCallback so EMA fires once per optimizer step
            self.add_callback(_EMAStepCallback(self))
        elif self.sdpo_config.teacher_mode == "frozen":
            self.ema_callback = None
        else:
            raise NotImplementedError(f"teacher_mode='{self.sdpo_config.teacher_mode}' not yet supported")

        # Stash for feedback strings collected from reward functions
        self._last_feedback_strings: list[str | None] | None = None
        # Stash for raw per-sample rewards (before advantage normalization)
        self._last_raw_rewards: torch.Tensor | None = None

    def _generate_and_score_completions(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Generate completions and prepare teacher prompt data.

        Overrides GRPOTrainer to:
        1. Capture raw rewards before advantage normalization.
        2. Group rollouts by prompt via RepeatSampler structure.
        3. Select demonstrations from successful peers.
        4. Build and tokenize teacher prompts with apply_chat_template.
        5. Concatenate teacher prompts with response tokens.
        6. Compute self_distillation_mask.
        """
        outputs = super()._generate_and_score_completions(inputs)

        completion_ids = outputs["completion_ids"]
        prompt_ids = outputs["prompt_ids"]
        completion_mask = outputs["completion_mask"]
        batch_size = completion_ids.size(0)

        # --- Recover per-sample rewards ---
        # In verl, _collect_solutions_by_uid uses reward_tensor.sum(dim=-1) (the
        # actual rewards, NOT advantages). TRL's outputs contain "advantages" which
        # are mean-centered per group. We need the raw rewards.
        #
        # Strategy: TRL stores per-reward-function rewards. We look for them in the
        # outputs dict. If not available, we reconstruct from advantages by reversing
        # the group normalization: reward_i = advantage_i + mean(rewards_in_group).
        # But since mean cancels, the ordering within a group is preserved. So we
        # can still use advantages for the threshold comparison if we set the threshold
        # relative to advantages (threshold=0 means "above group mean").
        #
        # However, the correct approach matching verl is to use actual rewards.
        # TRL's _generate_and_score_completions stores rewards before normalization.
        # We capture them via _last_raw_rewards if a reward wrapper sets them.
        rewards = self._last_raw_rewards
        if rewards is None:
            # Fallback: use advantages. The success_threshold should be set to 0.0
            # (meaning "above group mean") when using advantages as proxy.
            rewards = outputs["advantages"]

        # --- Collect feedback ---
        # Check if any reward function stored feedback on itself
        feedback_strings: list[str | None] = [None] * batch_size
        if self.sdpo_config.include_environment_feedback:
            for rf in self.reward_funcs:
                if hasattr(rf, "last_feedback") and rf.last_feedback:
                    fb = rf.last_feedback
                    for i in range(min(len(fb), batch_size)):
                        if fb[i] and isinstance(fb[i], str) and fb[i].strip():
                            feedback_strings[i] = fb[i]
                    break  # Use first reward func that has feedback

        # --- Decode texts ---
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)

        # --- Group rollouts and select demonstrations ---
        G = self.args.num_generations
        num_groups = batch_size // G
        solutions_list: list[str | None] = []

        for i in range(num_groups):
            s, e = i * G, (i + 1) * G
            group_rewards = rewards[s:e]
            group_completions = completions_text[s:e]
            group_uids = [i] * G

            for j in range(G):
                demo = select_demonstration(
                    idx=j,
                    uids=group_uids,
                    rewards=group_rewards.tolist() if isinstance(group_rewards, torch.Tensor) else group_rewards,
                    completions=group_completions,
                    success_threshold=self.sdpo_config.success_reward_threshold,
                    exclude_self=self.sdpo_config.dont_reprompt_on_self_success,
                    remove_thinking=self.sdpo_config.remove_thinking_from_demonstration,
                )
                solutions_list.append(demo)

        # --- Build teacher prompt texts ---
        teacher_prompts_list = build_teacher_prompts(
            prompts=prompts_text,
            solutions=solutions_list,
            feedback_list=feedback_strings,
            feedback_only_without_solution=self.sdpo_config.environment_feedback_only_without_solution,
            reprompt_template=self.sdpo_config.reprompt_template,
            solution_template=self.sdpo_config.solution_template,
            feedback_template=self.sdpo_config.feedback_template,
        )

        # --- Tokenize teacher prompts ---
        # In verl, this uses tokenizer.apply_chat_template with proper chat formatting.
        # We wrap each teacher prompt as a user message and apply the chat template.
        teacher_messages = [[{"role": "user", "content": tp}] for tp in teacher_prompts_list]

        # Try apply_chat_template first (matches verl), fall back to raw tokenization
        try:
            tokenized = self.processing_class.apply_chat_template(
                teacher_messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=self.sdpo_config.max_reprompt_length,
            )
            teacher_prompt_ids = tokenized["input_ids"].to(self.accelerator.device)
            teacher_prompt_mask = tokenized["attention_mask"].to(self.accelerator.device)
        except Exception:
            # Fallback for tokenizers that don't support batch apply_chat_template
            tokenized = self.processing_class(
                teacher_prompts_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.sdpo_config.max_reprompt_length,
            ).to(self.accelerator.device)
            teacher_prompt_ids = tokenized.input_ids
            teacher_prompt_mask = tokenized.attention_mask

        # Concatenate teacher prompt with completion tokens
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)

        # Position IDs accounting for padding
        teacher_position_ids = teacher_attention_mask.long().cumsum(dim=-1) - 1
        teacher_position_ids.masked_fill_(teacher_attention_mask == 0, 0)

        # --- Self-distillation mask ---
        # Matches verl: mask=1 if sample has a solution OR feedback is actually used
        # (accounting for environment_feedback_only_without_solution).
        sd_mask = compute_self_distillation_mask(
            solutions=solutions_list,
            feedback_list=feedback_strings,
            feedback_only_without_solution=self.sdpo_config.environment_feedback_only_without_solution,
        )
        self_distillation_mask = torch.tensor(sd_mask, dtype=torch.float32, device=self.accelerator.device)

        outputs["teacher_input_ids"] = teacher_input_ids
        outputs["teacher_attention_mask"] = teacher_attention_mask
        outputs["teacher_position_ids"] = teacher_position_ids
        outputs["self_distillation_mask"] = self_distillation_mask

        return outputs

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """
        Compute the SDPO self-distillation loss.

        Full override — does NOT call super().compute_loss() in SDPO mode.
        The GRPO clipped surrogate loss is completely replaced.
        """
        if not self.sdpo_config.enabled:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs.get("prompt_mask", torch.ones_like(prompt_ids))
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        old_per_token_logps = inputs.get("old_per_token_logps")

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        teacher_position_ids = inputs["teacher_position_ids"]
        self_distillation_mask = inputs["self_distillation_mask"]

        logits_to_keep = completion_ids.size(1)
        K = self.sdpo_config.distillation_topk

        # --- Student forward pass ---
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        )

        student_logits = student_outputs.logits[:, :-1, :]
        student_logits = student_logits[:, -logits_to_keep:, :]

        student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
        student_topk_log_probs, student_topk_indices = torch.topk(student_log_probs_full, k=K, dim=-1)

        student_per_token_logps = torch.gather(
            student_log_probs_full, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        # --- Teacher forward pass ---
        with torch.no_grad():
            teacher_outputs = self.ref_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                position_ids=teacher_position_ids,
                logits_to_keep=logits_to_keep + 1,
            )

            teacher_logits = teacher_outputs.logits[:, :-1, :]
            teacher_logits = teacher_logits[:, -logits_to_keep:, :]

            teacher_log_probs_full = torch.log_softmax(teacher_logits, dim=-1)

            # Gather teacher log probs at STUDENT's top-K indices (critical)
            teacher_topk_log_probs = torch.gather(teacher_log_probs_full, dim=-1, index=student_topk_indices)

        # --- Compute SDPO loss ---
        loss, metrics = compute_self_distillation_loss(
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_topk_log_probs,
            student_log_probs=student_per_token_logps,
            teacher_log_probs=None,
            response_mask=completion_mask,
            self_distillation_mask=self_distillation_mask,
            old_log_probs=old_per_token_logps,
            alpha=self.sdpo_config.alpha,
            add_tail=self.sdpo_config.distillation_add_tail,
            is_clip=self.sdpo_config.is_clip,
        )

        # Log metrics safely
        if hasattr(self, "log"):
            self.log(metrics)

        # NOTE: EMA update is handled by _EMAStepCallback.on_step_end,
        # NOT here. Placing it here would fire on every micro-batch during
        # gradient accumulation, which is incorrect. verl fires _update_teacher
        # once after the full optimizer step completes.

        return (loss, student_outputs) if return_outputs else loss
