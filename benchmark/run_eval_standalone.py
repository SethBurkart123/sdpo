#!/usr/bin/env python3
"""
Standalone evaluation: load a saved LoRA adapter and compute pass@1 on MBPP.

Usage:
    uv run python benchmark/run_eval_standalone.py --adapter benchmark/results/sdpo_adapter --label sdpo
    uv run python benchmark/run_eval_standalone.py --adapter benchmark/results/grpo_adapter --label grpo
    uv run python benchmark/run_eval_standalone.py --base-only --label baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from unsloth import FastLanguageModel

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import MAX_COMPLETION_LENGTH, MAX_SEQ_LENGTH, MODEL_NAME, RESULTS_DIR
from csv_logger import CSVLogger
from dataset import load_mbpp_splits, make_eval_dataset
from evaluate import compute_pass_at_1, evaluate_batch
from reward_mbpp import extract_code_block


def run_eval(model, tokenizer, eval_ds, label: str):
    """Generate completions and compute pass@1."""
    FastLanguageModel.for_inference(model)

    print(f"Evaluating {len(eval_ds)} problems for '{label}'...")
    completions = []
    for i in range(len(eval_ds)):
        sample = eval_ds[i]
        prompt_messages = sample["prompt"]

        input_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_LENGTH,
                temperature=0.01,
                do_sample=True,
                top_p=1.0,
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(completion)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eval_ds)} completions generated")

    test_lists = [eval_ds[i]["test_list"] for i in range(len(eval_ds))]
    import_lists = [eval_ds[i]["test_imports"] for i in range(len(eval_ds))]
    results = evaluate_batch(completions, test_lists, import_lists)
    pass_rate = compute_pass_at_1(results)

    print(
        f"  [{label}] pass@1 = {pass_rate:.4f} ({sum(r.passed == r.total and r.total > 0 for r in results)}/{len(results)} solved)"
    )
    return pass_rate


def main():
    parser = argparse.ArgumentParser(description="Standalone MBPP evaluation")
    parser.add_argument("--adapter", type=str, help="Path to saved LoRA adapter directory")
    parser.add_argument("--base-only", action="store_true", help="Evaluate base model without adapter")
    parser.add_argument("--label", type=str, required=True, help="Label for this eval (grpo/sdpo/baseline)")
    args = parser.parse_args()

    if not args.adapter and not args.base_only:
        parser.error("Provide --adapter or --base-only")

    if args.adapter:
        # Load base model + adapter in one shot via Unsloth.
        # Unsloth's from_pretrained handles LoRA adapters saved with
        # model.save_pretrained() — it reads adapter_config.json and
        # loads the quantized base + adapter together.
        print(f"Loading model + adapter from {args.adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        # Base model only (no adapter)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval dataset
    _, raw_eval = load_mbpp_splits()
    eval_ds = make_eval_dataset(raw_eval)

    # Run eval
    start = time.time()
    pass_rate = run_eval(model, tokenizer, eval_ds, args.label)
    elapsed = time.time() - start

    # Write result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"{args.label}_standalone_eval.csv"
    eval_csv = CSVLogger(
        str(RESULTS_DIR / csv_name),
        fieldnames=["step", "pass_at_1", "timestamp"],
    )
    eval_csv.log({"step": args.label, "pass_at_1": pass_rate, "timestamp": time.time()})
    eval_csv.close()

    print(f"\nDone in {elapsed / 60:.1f} min. Result written to {csv_name}")


if __name__ == "__main__":
    main()
