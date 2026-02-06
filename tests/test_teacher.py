"""
Tests for sdpo_trainer.teacher — EMA teacher management.

Matches the behavior of DataParallelPPOActor._update_teacher from
verl/workers/actor/dp_actor.py.
"""

import torch
import torch.nn as nn
import pytest

from sdpo_trainer.teacher import ema_update, EMATeacherCallback


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
