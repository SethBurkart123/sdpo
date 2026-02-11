"""
Tests for sdpo_rl.teacher — EMA teacher management.

Matches the behavior of DataParallelPPOActor._update_teacher from
verl/workers/actor/dp_actor.py.
"""

import torch
import torch.nn as nn
import pytest

from sdpo_rl.teacher import ema_update, EMATeacherCallback

peft = pytest.importorskip("peft", reason="peft required for lora_ema tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Two-layer model for testing EMA updates."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 2, bias=False)


@pytest.fixture
def student():
    return TinyModel(seed=0)


@pytest.fixture
def teacher():
    return TinyModel(seed=0)  # same init as student


# ---------------------------------------------------------------------------
# 1. EMA Update Formula
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    def test_formula_correct(self, student, teacher):
        """
        verl: teacher = (1 - rate) * teacher + rate * student
        After one update with rate=0.1, teacher should be 90% old + 10% student.
        """
        # Diverge the student weights
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)

        old_teacher_w = {n: p.clone() for n, p in teacher.named_parameters()}
        student_w = {n: p.clone() for n, p in student.named_parameters()}

        ema_update(teacher, student, rate=0.1)

        for name, p in teacher.named_parameters():
            expected = (1 - 0.1) * old_teacher_w[name] + 0.1 * student_w[name]
            torch.testing.assert_close(p.data, expected, atol=1e-6, rtol=1e-6)

    def test_rate_zero_no_change(self, student, teacher):
        """rate=0 means teacher should not change at all."""
        with torch.no_grad():
            for p in student.parameters():
                p.add_(5.0)

        old_teacher_w = {n: p.clone() for n, p in teacher.named_parameters()}
        ema_update(teacher, student, rate=0.0)

        for name, p in teacher.named_parameters():
            torch.testing.assert_close(p.data, old_teacher_w[name])

    def test_rate_one_copies_student(self, student, teacher):
        """rate=1 means teacher becomes an exact copy of student."""
        with torch.no_grad():
            for p in student.parameters():
                p.add_(3.0)

        ema_update(teacher, student, rate=1.0)

        for (_, tp), (_, sp) in zip(teacher.named_parameters(), student.named_parameters()):
            torch.testing.assert_close(tp.data, sp.data)

    def test_convergence_after_many_steps(self, student, teacher):
        """After many EMA updates with constant student, teacher should converge to student."""
        with torch.no_grad():
            for p in student.parameters():
                p.add_(2.0)

        for _ in range(200):
            ema_update(teacher, student, rate=0.05)

        for (_, tp), (_, sp) in zip(teacher.named_parameters(), student.named_parameters()):
            torch.testing.assert_close(tp.data, sp.data, atol=1e-3, rtol=1e-3)

    def test_no_grad_context(self, student, teacher):
        """EMA update should not create a computation graph."""
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)

        ema_update(teacher, student, rate=0.1)
        for p in teacher.parameters():
            assert not p.requires_grad or p.grad is None


# ---------------------------------------------------------------------------
# 2. EMATeacherCallback
# ---------------------------------------------------------------------------


class TestEMATeacherCallback:
    def test_fires_on_correct_step(self):
        """
        With num_iterations=2, EMA should fire when global_step % 2 == 0.
        (i.e., after finishing all iterations on one batch, before next generation)
        """
        callback = EMATeacherCallback(
            teacher_model=TinyModel(seed=1),
            student_model=TinyModel(seed=0),
            update_rate=0.05,
            num_iterations=2,
        )

        # Simulate steps
        steps_updated = []
        for step in range(1, 7):
            did_update = callback.should_update(global_step=step)
            if did_update:
                steps_updated.append(step)

        # Should fire at steps 2, 4, 6 (every num_iterations)
        assert steps_updated == [2, 4, 6]

    def test_fires_every_step_when_iterations_one(self):
        """With num_iterations=1 (default), should fire every step."""
        callback = EMATeacherCallback(
            teacher_model=TinyModel(seed=1),
            student_model=TinyModel(seed=0),
            update_rate=0.05,
            num_iterations=1,
        )
        for step in range(1, 5):
            assert callback.should_update(global_step=step)


# ---------------------------------------------------------------------------
# 3. LoRA EMA — Multi-Adapter Teacher
# ---------------------------------------------------------------------------

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig

from sdpo_rl.teacher import (
    collect_lora_adapter_pairs,
    ema_update_lora_adapters,
    init_lora_ema_teacher,
    LoraEMATeacherCallback,
    LORA_EMA_TEACHER_ADAPTER,
)


@pytest.fixture
def peft_model():
    """A small PEFT model with a single 'default' LoRA adapter."""
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    base = AutoModelForCausalLM.from_config(config)
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    return get_peft_model(base, lora_cfg)


class TestCollectLoraAdapterPairs:
    """Tests for collect_lora_adapter_pairs — the function that finds
    matching (teacher_param, student_param) pairs across two LoRA adapters
    on the same PEFT model."""

    def test_finds_all_pairs(self, peft_model):
        """Should find a pair for every lora_A and lora_B weight in every targeted module."""
        peft_model.add_adapter("teacher", LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
        pairs = collect_lora_adapter_pairs(peft_model, student_adapter="default", teacher_adapter="teacher")
        # SmolLM2-135M has 30 layers, 2 target modules (q_proj, v_proj),
        # 2 matrices each (lora_A, lora_B) = 30 * 2 * 2 = 120
        assert len(pairs) == 120

    def test_pairs_have_matching_shapes(self, peft_model):
        """Every (teacher, student) pair must have identical shapes."""
        peft_model.add_adapter("teacher", LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
        pairs = collect_lora_adapter_pairs(peft_model, student_adapter="default", teacher_adapter="teacher")
        for teacher_p, student_p in pairs:
            assert teacher_p.shape == student_p.shape

    def test_raises_on_missing_adapter(self, peft_model):
        """Should raise ValueError if the requested adapter doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            collect_lora_adapter_pairs(peft_model, student_adapter="default", teacher_adapter="nonexistent")


class TestInitLoraEmaTeacher:
    """Tests for init_lora_ema_teacher — creates the teacher adapter on a
    PEFT model and copies student weights into it."""

    def test_creates_teacher_adapter(self, peft_model):
        """Should create the 'sdpo_teacher' adapter on the model."""
        init_lora_ema_teacher(peft_model)
        # Verify teacher adapter exists
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A") and isinstance(module.lora_A, nn.ModuleDict):
                assert LORA_EMA_TEACHER_ADAPTER in module.lora_A
                break
        else:
            pytest.fail("No LoRA modules found with teacher adapter")

    def test_teacher_weights_match_student_after_init(self, peft_model):
        """After init, teacher adapter weights should be a copy of the student's."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )
        for teacher_p, student_p in pairs:
            torch.testing.assert_close(teacher_p.data.float(), student_p.data.float(), atol=1e-4, rtol=1e-4)

    def test_student_remains_active_after_init(self, peft_model):
        """The student ('default') adapter must be active after init."""
        init_lora_ema_teacher(peft_model)
        assert peft_model.active_adapter == ["default"] or peft_model.active_adapter == "default"

    def test_raises_on_non_peft_model(self):
        """Should raise ValueError when given a plain nn.Module."""
        plain_model = TinyModel(seed=0)
        with pytest.raises(ValueError, match="PEFT"):
            init_lora_ema_teacher(plain_model)

    def test_teacher_adapter_frozen(self, peft_model):
        """Teacher adapter weights should not require grad."""
        init_lora_ema_teacher(peft_model)
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A") and isinstance(module.lora_A, nn.ModuleDict):
                if LORA_EMA_TEACHER_ADAPTER in module.lora_A:
                    for p in module.lora_A[LORA_EMA_TEACHER_ADAPTER].parameters():
                        assert not p.requires_grad, f"{name}.lora_A teacher should be frozen"
                    for p in module.lora_B[LORA_EMA_TEACHER_ADAPTER].parameters():
                        assert not p.requires_grad, f"{name}.lora_B teacher should be frozen"


class TestEmaUpdateLoraAdapters:
    """Tests for ema_update_lora_adapters — in-place EMA update on adapter params only."""

    def test_formula_correct(self, peft_model):
        """EMA: teacher = (1-rate)*teacher + rate*student, applied to adapter params."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )

        # Diverge student weights
        with torch.no_grad():
            for _, student_p in pairs:
                student_p.add_(1.0)

        # Snapshot before
        old_teacher = [tp.clone() for tp, _ in pairs]
        old_student = [sp.clone() for _, sp in pairs]

        rate = 0.1
        ema_update_lora_adapters(peft_model, rate=rate)

        for i, (teacher_p, _) in enumerate(pairs):
            expected = (1.0 - rate) * old_teacher[i].float() + rate * old_student[i].float()
            torch.testing.assert_close(teacher_p.data.float(), expected, atol=1e-3, rtol=1e-3)

    def test_rate_zero_no_change(self, peft_model):
        """rate=0: teacher should not change."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )

        with torch.no_grad():
            for _, student_p in pairs:
                student_p.add_(5.0)

        old_teacher = [tp.clone() for tp, _ in pairs]
        ema_update_lora_adapters(peft_model, rate=0.0)

        for i, (teacher_p, _) in enumerate(pairs):
            torch.testing.assert_close(teacher_p.data, old_teacher[i])

    def test_rate_one_copies_student(self, peft_model):
        """rate=1: teacher becomes an exact copy of student."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )

        with torch.no_grad():
            for _, student_p in pairs:
                student_p.add_(3.0)

        ema_update_lora_adapters(peft_model, rate=1.0)

        # bf16 teacher has ~0.008 precision at this magnitude range,
        # so tolerance must accommodate the dtype conversion
        for teacher_p, student_p in pairs:
            torch.testing.assert_close(teacher_p.data.float(), student_p.data.float(), atol=0.01, rtol=0.01)

    def test_no_grad_context(self, peft_model):
        """EMA update should not create a computation graph."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )
        with torch.no_grad():
            for _, sp in pairs:
                sp.add_(1.0)
        ema_update_lora_adapters(peft_model, rate=0.1)
        for teacher_p, _ in pairs:
            assert teacher_p.grad is None

    def test_base_weights_untouched(self, peft_model):
        """EMA should NOT modify base model weights — only adapter params."""
        # Snapshot base weights before
        base_weights = {}
        for name, p in peft_model.named_parameters():
            if "lora" not in name.lower():
                base_weights[name] = p.data.clone()

        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )
        with torch.no_grad():
            for _, sp in pairs:
                sp.add_(1.0)
        ema_update_lora_adapters(peft_model, rate=0.5)

        for name, p in peft_model.named_parameters():
            if "lora" not in name.lower():
                torch.testing.assert_close(p.data, base_weights[name])

    def test_convergence(self, peft_model):
        """After many updates, teacher adapter should converge to student."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )
        with torch.no_grad():
            for _, sp in pairs:
                sp.add_(2.0)

        for _ in range(200):
            ema_update_lora_adapters(peft_model, rate=0.05)

        # bf16 precision limits convergence — accumulated rounding over 200
        # mul_/add_ steps causes ~0.15 max drift. We verify the mean error
        # is small (proves convergence) and max error is bounded (proves no blowup).
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        n_params = 0
        for teacher_p, student_p in pairs:
            diff = (teacher_p.data.float() - student_p.data.float()).abs()
            max_abs_diff = max(max_abs_diff, diff.max().item())
            mean_abs_diff += diff.sum().item()
            n_params += diff.numel()
        mean_abs_diff /= n_params
        assert mean_abs_diff < 0.05, f"Mean diff {mean_abs_diff:.4f} too large"
        assert max_abs_diff < 0.2, f"Max diff {max_abs_diff:.4f} too large"


class TestLoraEMATeacherCallback:
    """Tests for the callback that wraps adapter-level EMA with step gating."""

    def test_fires_on_correct_step(self, peft_model):
        """With num_iterations=2, should fire at steps 2, 4, 6."""
        init_lora_ema_teacher(peft_model)
        callback = LoraEMATeacherCallback(
            model=peft_model,
            update_rate=0.05,
            num_iterations=2,
        )
        steps_updated = [s for s in range(1, 7) if callback.should_update(s)]
        assert steps_updated == [2, 4, 6]

    def test_step_performs_update(self, peft_model):
        """Calling step() should actually change teacher weights."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )

        # Diverge student
        with torch.no_grad():
            for _, sp in pairs:
                sp.add_(1.0)
        old_teacher = [tp.clone() for tp, _ in pairs]

        callback = LoraEMATeacherCallback(model=peft_model, update_rate=0.1, num_iterations=1)
        did_update = callback.step(global_step=1)

        assert did_update is True
        # Teacher should have moved
        for i, (teacher_p, _) in enumerate(pairs):
            assert not torch.equal(teacher_p.data, old_teacher[i])

    def test_adapter_switching_produces_different_outputs(self, peft_model):
        """After EMA divergence, switching adapters should produce different logits."""
        init_lora_ema_teacher(peft_model)
        pairs = collect_lora_adapter_pairs(
            peft_model, student_adapter="default", teacher_adapter=LORA_EMA_TEACHER_ADAPTER
        )

        # Diverge student
        with torch.no_grad():
            for _, sp in pairs:
                sp.add_(1.0)

        x = torch.randint(0, 1000, (1, 5))

        peft_model.set_adapter("default")
        out_student = peft_model(x).logits

        peft_model.set_adapter(LORA_EMA_TEACHER_ADAPTER)
        with torch.no_grad():
            out_teacher = peft_model(x).logits

        # Restore student adapter
        peft_model.set_adapter("default")

        assert (out_student - out_teacher).abs().max().item() > 0.01
