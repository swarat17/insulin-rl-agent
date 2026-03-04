"""Unit tests for LagrangianGlucoseEnv — Phase 4."""

import pytest
from unittest.mock import MagicMock, patch

from src.env.glucose_env import LagrangianGlucoseEnv
from src.safety.constraints import DoseConstraintChecker
from src.safety.lagrangian import LagrangianMultiplier


def _mock_sim_obs(cgm: float):
    obs = MagicMock()
    obs.CGM = cgm
    return obs


_MOCK_INFO = {"time": None, "meal": 0.0}


def test_returns_augmented_reward():
    """
    With known λ, a tight checker, and a mocked CGM=120 (in-range → base +1.0),
    the returned reward must equal base_reward - λ * violation.
    """
    # λ=1.0, checker max=0.1 → action 5 (dose≈0.25) produces violation=0.15
    multiplier = LagrangianMultiplier(lambda_init=1.0)
    tight_checker = DoseConstraintChecker(min_dose=0.0, max_dose=0.1)

    env = LagrangianGlucoseEnv(
        multiplier=multiplier,
        checker=tight_checker,
        patient_name="adult#001",
        action_type="discrete",
    )
    env.reset(seed=0)

    action = 5  # DOSE_LEVELS[5] = 0.25 U/hr
    dose = env.DOSE_LEVELS[action]
    violation = tight_checker.constraint_violation(dose)  # 0.25 - 0.1 = 0.15
    expected_base = 1.0  # CGM=120 → in-range reward
    expected_reward = expected_base - 1.0 * violation

    with patch.object(
        env._sim_env, "_step", return_value=(_mock_sim_obs(120.0), 0.0, False, _MOCK_INFO)
    ):
        _, reward, _, _, _ = env.step(action)

    assert reward == pytest.approx(expected_reward)


def test_original_action_passed_to_simulator():
    """
    The dose arriving at Simglucose must equal the dose derived from the action,
    NOT a clipped version — even when the checker flags it as unsafe.
    """
    multiplier = LagrangianMultiplier(lambda_init=0.5)
    # Tight checker: any nonzero dose is a violation
    tight_checker = DoseConstraintChecker(min_dose=0.0, max_dose=0.0)

    env = LagrangianGlucoseEnv(
        multiplier=multiplier,
        checker=tight_checker,
        patient_name="adult#001",
        action_type="discrete",
    )
    env.reset(seed=0)

    action = 5  # dose = DOSE_LEVELS[5] ≈ 0.25 U/hr
    expected_dose = float(env.DOSE_LEVELS[action])

    captured = []
    original_step = env._sim_env._step

    def recording_step(dose):
        captured.append(dose)
        return original_step(dose)

    with patch.object(env._sim_env, "_step", side_effect=recording_step):
        env.step(action)

    assert len(captured) == 1
    assert captured[0] == pytest.approx(expected_dose), (
        f"Expected dose {expected_dose} to be passed to simulator, got {captured[0]}"
    )
