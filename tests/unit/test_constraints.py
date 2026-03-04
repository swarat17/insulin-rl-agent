"""Unit tests for src/safety/constraints.py — Phase 2."""

import pytest
from src.safety.constraints import DoseConstraintChecker, ClinicianBaseline


@pytest.fixture
def checker():
    return DoseConstraintChecker(min_dose=0.0, max_dose=0.5)


def test_safe_dose_passes(checker):
    assert checker.is_safe(0.25) is True
    assert checker.constraint_violation(0.25) == pytest.approx(0.0)


def test_overdose_detected(checker):
    assert checker.is_safe(0.6) is False
    assert checker.constraint_violation(0.6) > 0.0
    assert checker.constraint_violation(0.6) == pytest.approx(0.1)


def test_underdose_detected(checker):
    assert checker.is_safe(-0.1) is False
    assert checker.constraint_violation(-0.1) > 0.0


def test_clip_to_safe_lower_bound(checker):
    assert checker.clip_to_safe(-0.1) == pytest.approx(checker.min_dose)


def test_clip_to_safe_upper_bound(checker):
    assert checker.clip_to_safe(99.0) == pytest.approx(checker.max_dose)


def test_batch_violations_length_matches_input(checker):
    actions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, 0.25]
    result = checker.batch_violations(actions)
    assert len(result) == len(actions)
    assert all(isinstance(v, float) for v in result)
