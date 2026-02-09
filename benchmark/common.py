"""
Shared configuration and setup for all benchmark training runs.

Centralizes model loading, hyperparameters, eval callback, and CSV logging
so the three run scripts (GRPO, SDPO, SDPO-audit) stay DRY and only differ
in their trainer class and reward function setup.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from csv_logger import CSVLogger
from dataset import load_mbpp_splits, make_eval_dataset, make_train_dataset
from evaluate import compute_pass_at_1, evaluate_batch
from reward_mbpp import MBPPRewardFunction

# ---------------------------------------------------------------------------
# Constants — same for all three runs
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42  # Same init seed for all runs → differences come from algorithm, not init
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
NUM_GENERATIONS = 4  # 4 completions per prompt — safe for 10GB with SDPO's dual models
MAX_COMPLETION_LENGTH = 256  # MBPP solutions are short (~5-15 lines); 256 is plenty
BATCH_SIZE = 1  # 1 prompt per device — SDPO needs room for teacher model
GRAD_ACCUM = 4  # effective batch = 1 * 4 = 4 prompts = 16 completions
LEARNING_RATE = 5e-6
TEMPERATURE = 0.7
EVAL_EVERY = 50  # evaluate on test set every N steps
MAX_STEPS_DEFAULT = 200

RESULTS_DIR = Path(__file__).parent / "results"


def load_model_and_tokenizer():
    """
    Load Qwen2.5-0.5B-Instruct with Unsloth + 4-bit QLoRA.

    Returns (model, tokenizer) ready for training.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_datasets() -> tuple[Dataset, Dataset]:
    """Load and prepare MBPP train/eval datasets."""
    raw_train, raw_eval = load_mbpp_splits()
    train_ds = make_train_dataset(raw_train, seed=SEED)
    eval_ds = make_eval_dataset(raw_eval)
    return train_ds, eval_ds


class EvalCallback(TrainerCallback):
    """
    TrainerCallback that runs pass@1 evaluation every N steps.

    Generates one completion per eval problem using the current model,
    executes the code, and logs the pass rate to an eval CSV.
    """

    def __init__(
        self,
        model,
        tokenizer,
        eval_dataset: Dataset,
        eval_csv: CSVLogger,
        eval_every: int = EVAL_EVERY,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_csv = eval_csv
        self.eval_every = eval_every

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._maybe_eval(state)

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # TRL/Unsloth sometimes skips on_step_end but always fires on_log.
        # Belt-and-suspenders: try eval from both hooks.
        self._maybe_eval(state)

    def _maybe_eval(self, state: TrainerState):
        if state.global_step % self.eval_every != 0:
            return
        if state.global_step == 0:
            return
        # Guard against double-eval when both hooks fire for the same step
        if hasattr(self, "_last_eval_step") and self._last_eval_step == state.global_step:
            return
        self._last_eval_step = state.global_step

        pass_rate = self._run_eval(state.global_step)
        self.eval_csv.log(
            {
                "step": state.global_step,
                "pass_at_1": pass_rate,
                "timestamp": time.time(),
            }
        )
        print(f"  [eval] step={state.global_step}  pass@1={pass_rate:.4f}")

    def _run_eval(self, step: int) -> float:
        """Generate completions for eval set and compute pass@1."""
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)

        completions = []
        for i in range(len(self.eval_dataset)):
            sample = self.eval_dataset[i]
            prompt_messages = sample["prompt"]

            input_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Prompt tokenization limit (not MAX_COMPLETION_LENGTH)
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=0.01,  # near-greedy for deterministic eval
                    do_sample=True,
                    top_p=1.0,
                )

            # Decode only the generated part
            new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            completions.append(completion)

        # Evaluate
        test_lists = [self.eval_dataset[i]["test_list"] for i in range(len(self.eval_dataset))]
        import_lists = [self.eval_dataset[i]["test_imports"] for i in range(len(self.eval_dataset))]
        results = evaluate_batch(completions, test_lists, import_lists)
        pass_rate = compute_pass_at_1(results)

        # Put model back in training mode
        FastLanguageModel.for_training(self.model)

        return pass_rate


class ForceExitCallback(TrainerCallback):
    """
    Force-exit after the last training step.

    GRPOTrainer + Unsloth can hang or silently os._exit inside trainer.train()
    after the final step (TRL #3671, #3739). Unsloth sometimes calls os._exit
    before our on_step_end fires for the last step.

    Strategy: split post-training into two phases:
    1. save_fn — called at max_steps - 1 (one step early) to save the adapter
       before Unsloth can kill the process. Does NOT close CSVs or exit.
    2. post_train_fn — called at max_steps or on_train_end to close CSVs and
       report final stats, then os._exit(0).

    If Unsloth kills the process at step max_steps before we get a chance,
    we've already saved the adapter. The CSV files flush on every write so
    data is safe.

    Usage:
        trainer.add_callback(ForceExitCallback(
            max_steps=N, save_fn=fn, post_train_fn=fn))
    """

    def __init__(self, max_steps: int, save_fn=None, post_train_fn=None):
        self.max_steps = max_steps
        self.save_fn = save_fn
        self.post_train_fn = post_train_fn
        self._saved = False
        self._exited = False

    def _do_save(self):
        """Save adapter. Can be called multiple times — re-saves to capture latest weights."""
        try:
            if self.save_fn:
                self.save_fn()
                self._saved = True
        except Exception as e:
            print(f"[ForceExit] save error: {e}", flush=True)

    def _do_exit(self):
        """Run post_train_fn (close CSVs, report stats) then exit."""
        self._do_save()
        if self._exited:
            return
        self._exited = True
        try:
            if self.post_train_fn:
                self.post_train_fn(None)
        except Exception as e:
            print(f"[ForceExit] post_train error: {e}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        # os._exit bypasses atexit/finally but is needed because TRL/Unsloth
        # can hang indefinitely after the last step. CSV data is safe because
        # CSVLogger flushes on every write.
        import os

        os._exit(0)

    def on_step_end(self, args, state, control, **kwargs):
        # Save adapter one step early as insurance against Unsloth killing
        # the process at max_steps. Re-save at max_steps to get final weights.
        if state.global_step >= self.max_steps - 1:
            self._do_save()
        if state.global_step >= self.max_steps:
            print(f"  [ForceExit] on_step_end step={state.global_step} — exiting", flush=True)
            self._do_exit()

    def on_log(self, args, state, control, **kwargs):
        if state.global_step >= self.max_steps:
            print(f"  [ForceExit] on_log step={state.global_step} — exiting", flush=True)
            self._do_exit()

    def on_train_end(self, args, state, control, **kwargs):
        print(f"  [ForceExit] on_train_end step={state.global_step}", flush=True)
        self._do_exit()


def cleanup():
    """Free GPU memory between runs."""
    gc.collect()
    torch.cuda.empty_cache()
