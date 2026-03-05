"""
End-to-end tests — require trained model checkpoints in models/.

Run with: pytest tests/e2e/ -v -m e2e
"""

import os
from pathlib import Path

import pytest

from src.agents.ppo_agent import PPOAgent
from src.env.glucose_env import GlucoseEnv
from src.evaluation.metrics import time_in_range
from src.safety.constraints import ClinicianBaseline, DoseConstraintChecker

pytestmark = pytest.mark.e2e

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
_CHECKER = DoseConstraintChecker()
_N_EPISODES = 3


def _find_checkpoint(constrained: bool) -> Path | None:
    """Return first PPO checkpoint matching constrained flag, or None."""
    import json

    for p in sorted(_MODELS_DIR.glob("ppo_*.zip")):
        sidecar = p.with_suffix(".json")
        if sidecar.exists():
            with open(sidecar) as f:
                cfg = json.load(f)
            if cfg.get("constrained", False) == constrained:
                return p
    return None


def _eval_unsafe_fraction(agent, action_type: str, n_episodes: int) -> float:
    fracs = []
    for seed in range(n_episodes):
        env = GlucoseEnv(patient_name="adult#001", action_type=action_type)
        obs, _ = env.reset(seed=seed)
        violations = 0
        steps = 0
        for _ in range(480):
            action = agent.predict(obs)
            if action_type == "discrete":
                dose = float(env.DOSE_LEVELS[int(action)])
            else:
                import numpy as np
                dose = float(np.clip(action, 0.0, 1.0)) * env.max_basal_dose
            violations += 0 if _CHECKER.is_safe(dose) else 1
            steps += 1
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
        fracs.append(violations / steps if steps > 0 else 0.0)
    return sum(fracs) / len(fracs)


def test_clinician_baseline_produces_nonzero_tir():
    """1 episode on adult patient → time_in_range > 0%."""
    env = GlucoseEnv(patient_name="adult#001", action_type="discrete")
    baseline = ClinicianBaseline()
    obs, _ = env.reset(seed=0)
    cgm_hist = []
    for _ in range(480):
        cgm_raw = obs[0] * 400.0
        tod = obs[3]
        dose = baseline.recommend(cgm_raw, tod)
        idx = max(0, min(10, int(round(dose / 0.5 * 10))))
        obs, _, terminated, truncated, info = env.step(idx)
        cgm_hist.append(info["cgm"])
        if terminated or truncated:
            break
    env.close()
    assert time_in_range(cgm_hist) > 0.0


@pytest.mark.skipif(
    not (_find_checkpoint(True) and _find_checkpoint(False)),
    reason="Requires both constrained and unconstrained PPO checkpoints in models/",
)
def test_constrained_fewer_violations_than_unconstrained():
    """
    Constrained PPO has mean unsafe_action_fraction <= unconstrained PPO
    over 3 evaluation episodes.
    """
    constrained_path = _find_checkpoint(True)
    unconstrained_path = _find_checkpoint(False)

    env_c = GlucoseEnv(action_type="discrete")
    env_u = GlucoseEnv(action_type="discrete")
    agent_c = PPOAgent.load(str(constrained_path), env=env_c)
    agent_u = PPOAgent.load(str(unconstrained_path), env=env_u)

    frac_c = _eval_unsafe_fraction(agent_c, "discrete", _N_EPISODES)
    frac_u = _eval_unsafe_fraction(agent_u, "discrete", _N_EPISODES)

    assert frac_c <= frac_u
