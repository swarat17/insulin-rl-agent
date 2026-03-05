"""
Gymnasium wrapper around Simglucose's T1DSimEnv.

Observation space (6 features, all normalized to [0, 1]):
  [0] Current CGM reading / 400.0
  [1] CGM 1 step ago / 400.0
  [2] CGM 2 steps ago / 400.0
  [3] Time of day (0.0 = midnight, 1.0 = next midnight)
  [4] Steps since last meal event / episode length
  [5] Last dose taken / max_basal_dose

Action space:
  discrete  — Discrete(11): 11 evenly-spaced dose levels in [0, max_basal_dose] U/hr
  continuous — Box(0, 1, (1,)): maps linearly to [0, max_basal_dose]

Reward function (asymmetric, clinically motivated):
  +1.0  CGM in [70, 180] mg/dL  — target range
  -2.0  CGM < 70 mg/dL          — hypoglycemia (dangerous)
  -4.0  CGM < 54 mg/dL          — severe hypoglycemia (life-threatening; overrides -2.0)
  -0.5  CGM > 180 mg/dL         — hyperglycemia (harmful but less immediately dangerous)

Episode length: 480 steps (3-minute intervals = 24 simulated hours).
Termination: 480 steps reached (truncated=True) OR CGM < 40 mg/dL (terminated=True).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simglucose.envs import T1DSimEnv as _SimEnv

MAX_STEPS = 480  # 24 hours × 60 min / 3 min per step
MAX_CGM = 400.0  # normalisation denominator (mg/dL)

# Reward thresholds (mg/dL)
_SEVERE_HYPO = 54.0
_HYPO = 70.0
_HYPER = 180.0
_CRISIS = 40.0  # early-termination threshold


class GlucoseEnv(gym.Env):
    """Gymnasium-compatible wrapper for the Simglucose T1D simulator."""

    metadata = {"render_modes": []}

    # Default dose grid — overwritten in __init__ when max_basal_dose differs
    DOSE_LEVELS: np.ndarray = np.linspace(0.0, 0.5, 11)

    def __init__(
        self,
        patient_name: str = "adult#001",
        action_type: str = "discrete",
        max_basal_dose: float = 0.5,
    ):
        """
        Parameters
        ----------
        patient_name:
            Simglucose patient identifier, e.g. 'adult#001', 'child#001'.
        action_type:
            'discrete' or 'continuous'.
        max_basal_dose:
            Upper bound on basal insulin rate (U/hr).
        """
        super().__init__()

        assert action_type in (
            "discrete",
            "continuous",
        ), f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        self.patient_name = patient_name
        self.action_type = action_type
        self.max_basal_dose = max_basal_dose
        self.max_steps = MAX_STEPS

        # Action space
        if action_type == "discrete":
            self.DOSE_LEVELS = np.linspace(0.0, max_basal_dose, 11)
            self.action_space = spaces.Discrete(11)
        else:
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )

        # Observation space: 6 features, all in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Internal episode state
        self._cgm_history: list[float] = [120.0, 120.0, 120.0]
        self._last_action: float = 0.0
        self._steps_since_meal: int = MAX_STEPS
        self._step_count: int = 0
        self._current_time = None

        # Underlying simglucose environment
        self._sim_env = _SimEnv(patient_name=patient_name, seed=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self._current_time is not None:
            time_of_day = (self._current_time.hour * 60 + self._current_time.minute) / (
                24 * 60
            )
        else:
            time_of_day = 0.0

        last_action_norm = (
            self._last_action / self.max_basal_dose if self.max_basal_dose > 0 else 0.0
        )

        return np.array(
            [
                np.clip(self._cgm_history[-1] / MAX_CGM, 0.0, 1.0),
                np.clip(self._cgm_history[-2] / MAX_CGM, 0.0, 1.0),
                np.clip(self._cgm_history[-3] / MAX_CGM, 0.0, 1.0),
                float(time_of_day),
                min(self._steps_since_meal / self.max_steps, 1.0),
                float(np.clip(last_action_norm, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, cgm: float) -> float:
        """Asymmetric reward based on clinical glucose thresholds."""
        if cgm < _SEVERE_HYPO:
            return -4.0
        if cgm < _HYPO:
            return -2.0
        if cgm <= _HYPER:
            return 1.0
        return -0.5

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Propagate seed into simglucose's internal RNG
        if seed is not None:
            self._sim_env._seed(seed)

        # Reset the simglucose simulation (recreates patient + scenario)
        obs = self._sim_env._reset()
        initial_cgm = float(obs.CGM)

        # Reset episode state
        self._cgm_history = [initial_cgm, initial_cgm, initial_cgm]
        self._last_action = 0.0
        self._steps_since_meal = self.max_steps
        self._step_count = 0
        self._current_time = None

        return self._get_obs(), {}

    def step(self, action):
        # Decode action → dose in U/hr
        if self.action_type == "discrete":
            dose = float(self.DOSE_LEVELS[int(action)])
        else:
            dose = float(np.clip(action[0], 0.0, 1.0)) * self.max_basal_dose

        # Advance simglucose simulation by one 3-minute step
        obs, _sim_reward, _done, info = self._sim_env._step(dose)

        # Update CGM history (rolling window of 3)
        cgm = float(obs.CGM)
        self._cgm_history.pop(0)
        self._cgm_history.append(cgm)

        # Track time and meal events
        self._current_time = info.get("time")
        if info.get("meal", 0.0) > 0.0:
            self._steps_since_meal = 0
        else:
            self._steps_since_meal = min(self._steps_since_meal + 1, self.max_steps)

        self._last_action = dose
        self._step_count += 1

        # Reward, termination
        reward = self._compute_reward(cgm)
        terminated = bool(cgm < _CRISIS)
        truncated = bool(self._step_count >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, {"cgm": cgm}

    def render(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Lagrangian-constrained environment
# ---------------------------------------------------------------------------


class LagrangianGlucoseEnv(GlucoseEnv):
    """
    GlucoseEnv subclass that augments the reward with a Lagrangian penalty.

    At each step the dose is checked against clinical bounds. If it violates
    them, the base reward is reduced by λ × violation. λ is updated via dual
    gradient descent at the end of every episode, so it rises when the agent
    is unsafe and falls when it is safe.

    Crucially, the *original* action is always passed to Simglucose unchanged.
    The Lagrangian approach discourages unsafe doses through the reward signal,
    not by blocking them — this preserves a smooth, informative gradient for
    the policy to learn from.

    Parameters
    ----------
    multiplier : LagrangianMultiplier
        The shared Lagrange multiplier instance.
    checker : DoseConstraintChecker, optional
        Clinical bounds checker. Defaults to DoseConstraintChecker().
    **kwargs
        Passed through to GlucoseEnv (patient_name, action_type, etc.)
    """

    def __init__(self, multiplier, checker=None, **kwargs) -> None:
        # Lazy imports to avoid circular dependencies at module load time
        from src.safety.constraints import DoseConstraintChecker
        from src.safety.lagrangian import LagrangianMultiplier  # noqa: F401 (type hint)

        super().__init__(**kwargs)
        self._multiplier = multiplier
        self._checker = checker if checker is not None else DoseConstraintChecker()
        self._episode_violations: list[float] = []

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._episode_violations = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # 1. Compute the dose this action maps to (for violation check only)
        if self.action_type == "discrete":
            dose = float(self.DOSE_LEVELS[int(action)])
        else:
            dose = (
                float(np.clip(np.atleast_1d(action)[0], 0.0, 1.0)) * self.max_basal_dose
            )

        violation = self._checker.constraint_violation(dose)
        self._episode_violations.append(violation)

        # 2. Step the parent — passes the ORIGINAL action to Simglucose
        obs, base_reward, terminated, truncated, info = super().step(action)

        # 3. Penalise the reward proportional to λ and violation magnitude
        augmented_reward = self._multiplier.augment_reward(base_reward, violation)

        # 4. At episode end, update λ with the episode's violation history
        if terminated or truncated:
            self._multiplier.update(self._episode_violations)
            self._episode_violations = []

        return obs, augmented_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_constraint_stats(self) -> dict:
        """Return recent mean violation and current λ for logging."""
        mean_violation = (
            float(np.mean(self._episode_violations))
            if self._episode_violations
            else 0.0
        )
        return {
            "mean_violation": mean_violation,
            "lambda": self._multiplier.get_lambda(),
        }
