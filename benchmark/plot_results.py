#!/usr/bin/env python3
"""
Generate publication-quality plots from benchmark CSV results.

Reads CSV files from benchmark/results/ and produces:
  1. pass_at_1.png    — Test pass@1 over training steps (GRPO vs SDPO) — the hero plot
  2. train_reward.png — Training reward curves
  3. sdpo_internals.png — SDPO-specific metrics (teacher coverage, KL, loss)
  4. audit_delta.png  — |our_loss - ref_loss| over steps (correctness proof)

Usage:
    uv run python benchmark/plot_results.py [--results-dir benchmark/results]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"

# Consistent styling
COLORS = {
    "baseline": "#9E9E9E",  # grey
    "grpo": "#2196F3",  # blue
    "sdpo": "#FF5722",  # deep orange
    "sdpo_audit": "#9C27B0",  # purple
}
LABELS = {
    "baseline": "Base model",
    "grpo": "GRPO",
    "sdpo": "SDPO (ours)",
    "sdpo_audit": "SDPO + audit",
}


def read_csv(path: Path) -> dict[str, list[float]]:
    """Read a CSV into a dict of column_name -> list[float]. Skips empty values."""
    data: dict[str, list[float]] = {}
    if not path.exists():
        return data
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(float("nan"))
    return data


def deduplicate_by_step(data: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    When multiple CSV rows share the same step, keep only the row with the most
    non-NaN values (the richest log entry). This handles cases where on_log fires
    multiple times per step (e.g. per micro-batch SDPO metrics + per-step TRL metrics).
    """
    if "step" not in data or not data["step"]:
        return data
    keys = list(data.keys())
    rows = list(zip(*(data[k] for k in keys)))
    step_idx = keys.index("step")

    best: dict[float, tuple] = {}
    for row in rows:
        step = row[step_idx]
        richness = sum(1 for v in row if not np.isnan(v))
        if step not in best or richness > best[step][1]:
            best[step] = (row, richness)

    deduped = {k: [] for k in keys}
    for step in sorted(best.keys()):
        row = best[step][0]
        for k, v in zip(keys, row):
            deduped[k].append(v)
    return deduped


def smooth(values: list[float], window: int = 5) -> np.ndarray:
    """Simple moving average for smoother curves."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad the start with original values so length matches
    pad = arr[: len(arr) - len(smoothed)]
    return np.concatenate([pad, smoothed])


def setup_style():
    """Apply clean publication styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


# ---------------------------------------------------------------------------
# Plot 1: pass@1 (the hero plot)
# ---------------------------------------------------------------------------


def plot_pass_at_1(results_dir: Path, output_dir: Path):
    """Test pass@1 over training steps — GRPO vs SDPO."""
    grpo = read_csv(results_dir / "grpo_eval.csv")
    sdpo = read_csv(results_dir / "sdpo_eval.csv")
    audit = read_csv(results_dir / "sdpo_audit_eval.csv")

    if not grpo and not sdpo:
        print("  [skip] No eval CSVs found for pass@1 plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for data, key in [(grpo, "grpo"), (sdpo, "sdpo"), (audit, "sdpo_audit")]:
        if "step" in data and "pass_at_1" in data:
            steps = data["step"]
            scores = data["pass_at_1"]
            ax.plot(steps, scores, marker="o", markersize=4, color=COLORS[key], label=LABELS[key], linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Test Pass@1")
    ax.set_title("MBPP Test Pass@1: GRPO vs SDPO\n(Qwen2.5-0.5B-Instruct, 4-bit QLoRA)")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)

    path = output_dir / "pass_at_1.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Training reward
# ---------------------------------------------------------------------------


def plot_train_reward(results_dir: Path, output_dir: Path):
    """Training reward (mean) over steps."""
    grpo = deduplicate_by_step(read_csv(results_dir / "grpo_train.csv"))
    sdpo = deduplicate_by_step(read_csv(results_dir / "sdpo_train.csv"))

    if not grpo and not sdpo:
        print("  [skip] No train CSVs found for reward plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for data, key in [(grpo, "grpo"), (sdpo, "sdpo")]:
        if "step" in data and "reward_mean" in data:
            steps = np.array(data["step"])
            rewards = data["reward_mean"]
            # Filter NaN
            valid = ~np.isnan(rewards) if isinstance(rewards, np.ndarray) else [not np.isnan(r) for r in rewards]
            steps_v = [s for s, v in zip(steps, valid) if v]
            rewards_v = [r for r, v in zip(rewards, valid) if v]
            if rewards_v:
                smoothed = smooth(rewards_v, window=10)
                ax.plot(steps_v, smoothed, color=COLORS[key], label=LABELS[key], linewidth=1.5, alpha=0.9)
                # Light raw values behind
                ax.plot(steps_v, rewards_v, color=COLORS[key], linewidth=0.3, alpha=0.25)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward Over Time")
    ax.legend(loc="lower right")

    path = output_dir / "train_reward.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: SDPO internals
# ---------------------------------------------------------------------------


def plot_sdpo_internals(results_dir: Path, output_dir: Path):
    """SDPO-specific metrics: teacher coverage, KL, distillation loss."""
    sdpo = deduplicate_by_step(read_csv(results_dir / "sdpo_train.csv"))

    if not sdpo or "step" not in sdpo:
        print("  [skip] No SDPO train CSV found for internals plot")
        return

    steps = sdpo["step"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Teacher coverage
    if "sdpo_teacher_coverage" in sdpo:
        vals = sdpo["sdpo_teacher_coverage"]
        valid_s = [s for s, v in zip(steps, vals) if not np.isnan(v)]
        valid_v = [v for v in vals if not np.isnan(v)]
        if valid_v:
            axes[0].plot(valid_s, valid_v, color=COLORS["sdpo"], linewidth=1.5)
            axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Teacher Coverage\n(fraction with demo/feedback)")

    # KL divergence
    if "sdpo_kl_mean" in sdpo:
        vals = sdpo["sdpo_kl_mean"]
        valid_s = [s for s, v in zip(steps, vals) if not np.isnan(v)]
        valid_v = [v for v in vals if not np.isnan(v)]
        if valid_v:
            smoothed = smooth(valid_v, window=10)
            axes[1].plot(valid_s, smoothed, color=COLORS["sdpo"], linewidth=1.5)
            axes[1].plot(valid_s, valid_v, color=COLORS["sdpo"], linewidth=0.3, alpha=0.25)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("KL (nats)")
    axes[1].set_title("Student-Teacher KL Divergence")

    # Distillation loss
    if "sdpo_loss" in sdpo:
        vals = sdpo["sdpo_loss"]
        valid_s = [s for s, v in zip(steps, vals) if not np.isnan(v)]
        valid_v = [v for v in vals if not np.isnan(v)]
        if valid_v:
            smoothed = smooth(valid_v, window=10)
            axes[2].plot(valid_s, smoothed, color=COLORS["sdpo"], linewidth=1.5)
            axes[2].plot(valid_s, valid_v, color=COLORS["sdpo"], linewidth=0.3, alpha=0.25)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Self-Distillation Loss")

    fig.suptitle("SDPO Training Internals", fontsize=15, y=1.02)
    fig.tight_layout()

    path = output_dir / "sdpo_internals.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Audit delta
# ---------------------------------------------------------------------------


def plot_audit_delta(results_dir: Path, output_dir: Path):
    """|our_loss - ref_loss| per step — proves implementation correctness."""
    audit = deduplicate_by_step(read_csv(results_dir / "sdpo_audit_train.csv"))

    if not audit or "step" not in audit or "audit_delta" not in audit:
        print("  [skip] No audit train CSV found for delta plot")
        return

    steps = audit["step"]
    deltas = audit["audit_delta"]
    valid_s = [s for s, d in zip(steps, deltas) if not np.isnan(d) and d > 0]
    valid_d = [d for d in deltas if not np.isnan(d) and d > 0]

    if not valid_d:
        print("  [skip] No valid audit deltas found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: delta over steps
    axes[0].semilogy(valid_s, valid_d, color=COLORS["sdpo_audit"], linewidth=1.2, marker=".", markersize=3)
    axes[0].axhline(y=1e-4, color="green", linestyle="--", alpha=0.7, label="1e-4 threshold")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("|our_loss - ref_loss|")
    axes[0].set_title("Loss Delta Per Step (log scale)")
    axes[0].legend()

    # Right: our vs ref scatter
    if "audit_our_loss" in audit and "audit_ref_loss" in audit:
        ours = audit["audit_our_loss"]
        refs = audit["audit_ref_loss"]
        valid_pairs = [(o, r) for o, r in zip(ours, refs) if not np.isnan(o) and not np.isnan(r)]
        if valid_pairs:
            o_arr = [p[0] for p in valid_pairs]
            r_arr = [p[1] for p in valid_pairs]
            lo = min(min(o_arr), min(r_arr))
            hi = max(max(o_arr), max(r_arr))
            margin = max((hi - lo) * 0.1, 1e-6)
            lim_lo = lo - margin
            lim_hi = hi + margin
            axes[1].scatter(r_arr, o_arr, s=12, alpha=0.6, color=COLORS["sdpo_audit"])
            axes[1].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, label="y=x (perfect match)")
            axes[1].set_xlabel("Reference Loss")
            axes[1].set_ylabel("Our Loss")
            axes[1].set_title("Our Loss vs Reference Loss")
            axes[1].legend(loc="upper left")
            axes[1].set_xlim(lim_lo, lim_hi)
            axes[1].set_ylim(lim_lo, lim_hi)
            axes[1].set_aspect("equal")

    max_delta = max(valid_d)
    fig.suptitle(
        f"Reference Audit: max |delta| = {max_delta:.2e}  ({'PASS' if max_delta < 1e-4 else 'INVESTIGATE'})",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    path = output_dir / "audit_delta.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save PNGs (defaults to results-dir)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"Reading results from: {results_dir}")
    print(f"Saving plots to: {output_dir}\n")

    plot_pass_at_1(results_dir, output_dir)
    plot_train_reward(results_dir, output_dir)
    plot_sdpo_internals(results_dir, output_dir)
    plot_audit_delta(results_dir, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
