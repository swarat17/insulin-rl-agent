"""Unit tests for src/safety/lagrangian.py — Phase 4."""

import pytest
from src.safety.lagrangian import LagrangianMultiplier


@pytest.fixture
def multiplier():
    return LagrangianMultiplier(
        lambda_init=0.01,
        lr_lambda=0.001,
        epsilon=0.05,
        lambda_max=10.0,
    )


def test_lambda_increases_when_violated(multiplier):
    """Repeated update() with violations > epsilon → λ strictly increases."""
    prev = multiplier.get_lambda()
    for _ in range(5):
        multiplier.update([0.2])  # 0.2 > epsilon (0.05)
        current = multiplier.get_lambda()
        assert current > prev, f"λ did not increase: {prev} → {current}"
        prev = current


def test_lambda_decreases_when_safe(multiplier):
    """Repeated update() with zero violations → λ decreases toward 0."""
    # Start from a non-zero λ so there's room to decrease
    m = LagrangianMultiplier(lambda_init=0.5, lr_lambda=0.001, epsilon=0.05)
    initial = m.get_lambda()
    for _ in range(100):
        m.update([0.0])
    assert m.get_lambda() < initial


def test_lambda_never_below_zero(multiplier):
    """λ is clamped at 0 regardless of how many safe updates are applied."""
    m = LagrangianMultiplier(lambda_init=0.001, lr_lambda=0.1, epsilon=0.05)
    for _ in range(1000):
        m.update([0.0])
    assert m.get_lambda() >= 0.0


def test_lambda_never_exceeds_max(multiplier):
    """λ is clamped at lambda_max regardless of violation frequency."""
    for _ in range(100_000):
        multiplier.update([1.0])  # large violation
    assert multiplier.get_lambda() <= multiplier.lambda_max


def test_augmented_reward_penalized(multiplier):
    """With λ=1.0 and violation=0.5, augmented reward = base_reward - 0.5."""
    m = LagrangianMultiplier(lambda_init=1.0)
    base_reward = 2.0
    result = m.augment_reward(base_reward, violation=0.5)
    assert result == pytest.approx(base_reward - 0.5)


def test_augmented_reward_unchanged_when_safe(multiplier):
    """With violation=0.0, augmented reward equals base reward exactly."""
    base_reward = 1.0
    result = multiplier.augment_reward(base_reward, violation=0.0)
    assert result == pytest.approx(base_reward)
