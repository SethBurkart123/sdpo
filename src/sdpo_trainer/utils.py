"""
Utility functions for sdpo-trainer.

Includes Unsloth detection and import order validation.
"""

from __future__ import annotations

import warnings


def check_unsloth_import_order() -> bool:
    """
    Verify that Unsloth's PatchFastRL was called before SDPOTrainer was imported.

    When using Unsloth, PatchFastRL("GRPO", ...) replaces trl.GRPOTrainer in the
    module namespace. If SDPOTrainer(GRPOTrainer) was imported before that
    replacement, super() calls will go to the unpatched GRPOTrainer and Unsloth's
    generation optimizations won't apply.

    Returns True if Unsloth is not installed or if import order is correct.
    """
    try:
        import trl

        trainer_cls = getattr(trl, "GRPOTrainer", None)
        if trainer_cls is None:
            return True

        # Unsloth's patched class typically has "Unsloth" in its name or module
        cls_name = trainer_cls.__name__
        cls_module = trainer_cls.__module__ or ""

        if "unsloth" in cls_module.lower() or "unsloth" in cls_name.lower():
            # Unsloth is active and patched — good
            return True

        # Check if unsloth is importable but hasn't patched yet
        try:
            import unsloth  # noqa: F401

            warnings.warn(
                "Unsloth is installed but PatchFastRL was not called before importing sdpo_trainer. "
                "For Unsloth optimizations, call PatchFastRL('GRPO', FastLanguageModel) BEFORE "
                "importing from sdpo_trainer.",
                UserWarning,
                stacklevel=3,
            )
            return False
        except ImportError:
            # Unsloth not installed — no issue
            return True

    except ImportError:
        return True
