"""
SDPO reprompting / teacher prompt construction.

Faithful reimplementation of the reprompting logic from
verl/trainer/ppo/ray_trainer.py in the lasgroup/SDPO repository.

Key behaviors:
- Demonstration selection picks the first successful peer in the same UID group.
- Self-exclusion (dont_reprompt_on_self_success) prevents using own response as demo.
- Thinking tag removal strips <think>...</think> blocks from demonstrations.
- Template assembly follows the exact verl format: {prompt}{solution}{feedback}\n\nCorrectly solve...
- When neither solution nor feedback exists, the raw prompt is passed through unchanged.
"""

from __future__ import annotations

import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Default templates matching verl/workers/config/actor.py SelfDistillationConfig
# ---------------------------------------------------------------------------

DEFAULT_REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"

DEFAULT_SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"

DEFAULT_FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"


def remove_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> tags and their content from text.

    Uses non-greedy matching with re.DOTALL so newlines within <think> blocks
    are consumed. Trailing whitespace after the closing tag is also removed.

    Matches verl's _remove_thinking_trace.
    """
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def select_demonstration(
    idx: int,
    uids: list,
    rewards: list[float],
    completions: list[str],
    success_threshold: float = 1.0,
    exclude_self: bool = True,
    remove_thinking: bool = False,
) -> str | None:
    """
    Select a successful demonstration from the same prompt group.

    Matches verl's _collect_solutions_by_uid + _get_solution.

    Args:
        idx: Index of the current sample.
        uids: List of prompt UIDs (samples with same UID share a prompt).
        rewards: List of scalar rewards per sample.
        completions: List of completion strings.
        success_threshold: Minimum reward to count as "successful".
        exclude_self: If True, exclude the sample's own completion as a demo.
        remove_thinking: If True, strip <think>...</think> from the demo.

    Returns:
        The demonstration string, or None if no successful peer exists.
    """
    target_uid = uids[idx]

    # Collect successful indices for this UID
    candidates = [j for j in range(len(uids)) if uids[j] == target_uid and rewards[j] >= success_threshold]

    if exclude_self:
        candidates = [j for j in candidates if j != idx]

    if not candidates:
        return None

    # verl takes solution_idxs[0] — effectively the first successful one found
    demo_idx = candidates[0]
    demo = completions[demo_idx]

    if remove_thinking:
        demo = remove_thinking_tags(demo)

    return demo


def build_teacher_prompts(
    prompts: list[str],
    solutions: list[str | None],
    feedback_list: list[str | None],
    feedback_only_without_solution: bool = False,
    reprompt_template: str = DEFAULT_REPROMPT_TEMPLATE,
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE,
    feedback_template: str = DEFAULT_FEEDBACK_TEMPLATE,
) -> list[str]:
    """
    Construct the teacher prompt for each sample in the batch.

    Matches verl's _build_teacher_message logic:
    - If a solution exists, include it via solution_template.
    - If feedback exists (and is allowed by feedback_only_without_solution), include it.
    - If neither exists, return the raw prompt unchanged (no reprompting).

    Args:
        prompts: Original user prompts.
        solutions: Successful demonstration for each sample, or None.
        feedback_list: Environment feedback for each sample, or None.
        feedback_only_without_solution: If True, only include feedback when no solution exists.
        reprompt_template: Template combining prompt + solution + feedback.
        solution_template: Template for the solution section.
        feedback_template: Template for the feedback section.

    Returns:
        List of teacher prompt strings, one per sample.
    """
    results = []
    for i in range(len(prompts)):
        has_solution = solutions[i] is not None
        has_feedback = feedback_list[i] is not None and str(feedback_list[i]).strip() != ""

        # Determine whether to use feedback
        use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

        # Build sections
        solution_section = ""
        if has_solution:
            solution_section = solution_template.format(successful_previous_attempt=solutions[i])

        feedback_section = ""
        if use_feedback:
            feedback_section = feedback_template.format(feedback_raw=feedback_list[i])

        # Assemble
        if has_solution or use_feedback:
            text = reprompt_template.format(
                prompt=prompts[i],
                solution=solution_section,
                feedback=feedback_section,
            )
        else:
            # No reprompting — raw prompt passthrough
            text = prompts[i]

        results.append(text)

    return results


def build_teacher_messages(
    original_messages: list,
    teacher_prompt_texts: list[str],
) -> list[list[dict]]:
    """
    Build structured chat message lists for the teacher prompt.

    Matches verl's _build_teacher_message which preserves system messages
    from the original prompt and only replaces the final user turn with
    the reprompted content.

    Reference: SDPO_reference/verl/trainer/ppo/ray_trainer.py:710-744
        system_messages = raw_prompt[i][:-1]
        return system_messages + [{"role": "user", "content": reprompt_text}]

    Args:
        original_messages: The raw prompt messages for each sample. Each
            element is either a list of message dicts (conversational) or
            a plain string.
        teacher_prompt_texts: The reprompted teacher prompt text for each
            sample (output of build_teacher_prompts).

    Returns:
        List of chat message lists, one per sample. Each message list
        preserves all system/context messages from the original prompt
        with the final user turn replaced by the teacher prompt text.
    """
    results = []
    for i in range(len(original_messages)):
        orig = original_messages[i]
        teacher_text = teacher_prompt_texts[i]

        if isinstance(orig, list) and len(orig) > 0 and isinstance(orig[0], dict):
            # Conversational format: preserve all messages except last user turn
            system_messages = orig[:-1]
            results.append(system_messages + [{"role": "user", "content": teacher_text}])
        else:
            # Plain string: wrap as single user message
            results.append([{"role": "user", "content": teacher_text}])

    return results


def compute_self_distillation_mask(
    solutions: list[str | None],
    feedback_list: list[str | None],
    feedback_only_without_solution: bool = False,
) -> list[float]:
    """
    Compute the self-distillation mask indicating which samples have teacher signal.

    A sample has teacher signal (mask=1.0) if it has either:
    - A successful peer demonstration (solution is not None), OR
    - Feedback that is actually used (accounting for feedback_only_without_solution).

    Matches verl's self_distillation_mask construction:
        feedback_used[i] = has_feedback and (not feedback_only_without_solution or not has_solution)
        mask[i] = has_solution or feedback_used[i]

    Args:
        solutions: Demonstration string per sample, or None.
        feedback_list: Feedback string per sample, or None.
        feedback_only_without_solution: If True, feedback only counts when no solution exists.

    Returns:
        List of 1.0/0.0 values, one per sample.
    """
    mask = []
    for i in range(len(solutions)):
        has_solution = solutions[i] is not None
        has_feedback = (
            feedback_list[i] is not None and isinstance(feedback_list[i], str) and feedback_list[i].strip() != ""
        )
        feedback_used = has_feedback and (not feedback_only_without_solution or not has_solution)
        mask.append(1.0 if (has_solution or feedback_used) else 0.0)
    return mask
