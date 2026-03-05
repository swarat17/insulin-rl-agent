"""
Integration tests that verify the full pipeline components work together.
No trained checkpoints required — uses live GlucoseEnv episodes.
"""

import pytest

from src.env.glucose_env import GlucoseEnv, LagrangianGlucoseEnv
from src.evaluation.metrics import compute_all_metrics
from src.safety.constraints import DoseConstraintChecker
from src.safety.lagrangian import LagrangianMultiplier


def _run_short_episode(env, steps=10):
    obs, _ = env.reset(seed=0)
    cgm_hist = []
    for _ in range(steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        cgm_hist.append(info["cgm"])
        if terminated or truncated:
            break
    return cgm_hist


def test_env_cgm_output_compatible_with_metrics():
    """Full GlucoseEnv episode produces a CGM list that compute_all_metrics accepts."""
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    obs, _ = env.reset(seed=42)
    cgm_hist = []
    for _ in range(480):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        cgm_hist.append(info["cgm"])
        if terminated or truncated:
            break
    env.close()

    # Must not raise
    result = compute_all_metrics(cgm_hist)
    assert "time_in_range" in result
    assert "hypoglycemia_rate" in result


def test_lagrangian_env_compatible_with_metrics():
    """LagrangianGlucoseEnv episode produces a CGM list that compute_all_metrics accepts."""
    multiplier = LagrangianMultiplier()
    env = LagrangianGlucoseEnv(
        patient_name="adult#001", action_type="discrete", multiplier=multiplier
    )
    cgm_hist = _run_short_episode(env, steps=20)
    env.close()

    result = compute_all_metrics(cgm_hist)
    assert isinstance(result["time_in_range"], float)


def test_constraint_bounds_compatible_with_dose_levels():
    """DoseConstraintChecker.max_dose >= max value in GlucoseEnv.DOSE_LEVELS."""
    checker = DoseConstraintChecker()
    env = GlucoseEnv(action_type="discrete")
    assert checker.max_dose >= float(env.DOSE_LEVELS.max())
    env.close()
