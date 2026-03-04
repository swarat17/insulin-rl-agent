"""Unit tests for src/env/glucose_env.py — Phase 1."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.env.glucose_env import GlucoseEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def discrete_env():
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    env.reset(seed=0)
    return env


@pytest.fixture
def continuous_env():
    return GlucoseEnv(patient_name="adult#001", action_type="continuous")


# ---------------------------------------------------------------------------
# Observation & action spaces
# ---------------------------------------------------------------------------


def test_observation_space_shape(discrete_env):
    assert discrete_env.observation_space.shape == (6,)


def test_discrete_action_space_has_11_levels(discrete_env):
    assert discrete_env.action_space.n == 11


def test_continuous_action_space_bounds(continuous_env):
    assert continuous_env.action_space.shape == (1,)
    assert continuous_env.action_space.low[0] == pytest.approx(0.0)
    assert continuous_env.action_space.high[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_returns_valid_observation():
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    obs, info = env.reset(seed=42)

    assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "Observation contains NaN values"
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Step return types
# ---------------------------------------------------------------------------


def test_step_returns_correct_tuple_types(discrete_env):
    result = discrete_env.step(0)
    assert len(result) == 5, "step() must return a 5-tuple"
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def test_in_range_reward_is_positive():
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    assert env._compute_reward(120.0) == pytest.approx(1.0)
    assert env._compute_reward(70.0) == pytest.approx(1.0)
    assert env._compute_reward(180.0) == pytest.approx(1.0)


def test_hypo_reward_is_negative_two():
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    assert env._compute_reward(69.9) == pytest.approx(-2.0)
    assert env._compute_reward(60.0) == pytest.approx(-2.0)
    assert env._compute_reward(54.0) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------


def _make_sim_obs(cgm_value: float):
    """Create a mock simglucose Observation namedtuple with the given CGM."""
    obs = MagicMock()
    obs.CGM = cgm_value
    return obs


def test_severe_hypo_terminates_episode():
    """CGM < 40 mg/dL must set terminated=True."""
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    env.reset(seed=0)

    mock_info = {"time": None, "meal": 0.0}
    with patch.object(
        env._sim_env,
        "_step",
        return_value=(_make_sim_obs(35.0), -1.0, False, mock_info),
    ):
        _, _, terminated, _, _ = env.step(0)

    assert terminated is True, "Episode must terminate when CGM < 40 mg/dL"


def test_episode_truncates_at_480_steps():
    """Running 480 steps must set truncated=True."""
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    env.reset(seed=0)

    mock_info = {"time": None, "meal": 0.0}
    mock_obs = _make_sim_obs(120.0)

    with patch.object(
        env._sim_env,
        "_step",
        return_value=(mock_obs, 0.0, False, mock_info),
    ):
        truncated = False
        for _ in range(480):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated:
                break

    assert truncated is True, "Episode must truncate after 480 steps"
