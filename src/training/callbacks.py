"""
Stable-Baselines3 callbacks for training the insulin dosing agents.

WandbCallback
    At each eval interval, runs one evaluation episode and logs to W&B:
    mean episode reward, TIR %, hypoglycemia rate, learning rate, timesteps.

SafetyMonitorCallback
    At each step, checks the constraint violation of the action taken and
    accumulates it. At each eval interval, logs the fraction of unsafe
    actions in that window to W&B. Monitoring only — does NOT block actions.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.evaluation.metrics import compute_all_metrics
from src.safety.constraints import DoseConstraintChecker


def _action_to_dose(action, env) -> float:
    """Convert a raw SB3 action to dose in U/hr using env's mapping."""
    val = float(np.atleast_1d(np.squeeze(action))[0])
    if env.action_type == "discrete":
        return float(env.DOSE_LEVELS[int(val)])
    return float(np.clip(val, 0.0, 1.0)) * env.max_basal_dose


class WandbCallback(BaseCallback):
    """
    Runs one deterministic eval episode every `eval_freq` steps and logs
    clinical metrics + reward to Weights & Biases.

    If a LagrangianMultiplier is supplied, also logs the current λ value
    so the dual learning dynamics are visible in W&B.
    """

    def __init__(self, eval_env, eval_freq: int = 10_000, multiplier=None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.multiplier = multiplier

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_eval()
        return True

    def _run_eval(self) -> None:
        import wandb

        obs, _ = self.eval_env.reset()
        cgm_history: list[float] = []
        total_reward = 0.0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            cgm_history.append(info["cgm"])

        metrics = compute_all_metrics(cgm_history)

        lr = self.model.learning_rate
        if callable(lr):
            lr = lr(self.model._current_progress_remaining)

        if wandb.run is not None:
            log_dict = {
                "eval/episode_reward": total_reward,
                "eval/tir": metrics["time_in_range"],
                "eval/hypo_rate": metrics["hypoglycemia_rate"],
                "eval/learning_rate": float(lr),
                "timesteps": self.num_timesteps,
            }
            if self.multiplier is not None:
                log_dict["lagrangian/lambda"] = self.multiplier.get_lambda()
            wandb.log(log_dict)


class SafetyMonitorCallback(BaseCallback):
    """
    Tracks the fraction of steps where the agent's action violates the
    dose constraint bounds. Logs to W&B every `eval_freq` steps.

    Monitoring only — does NOT alter or block any actions.
    The Lagrangian enforcement layer (Phase 4) handles actual penalisation.
    """

    def __init__(
        self,
        env,
        checker: DoseConstraintChecker | None = None,
        eval_freq: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env = env
        self.checker = checker or DoseConstraintChecker()
        self.eval_freq = eval_freq
        self._window_unsafe: list[bool] = []

    def _on_step(self) -> bool:
        import wandb

        actions = self.locals.get("actions")
        if actions is not None:
            dose = _action_to_dose(actions, self.env)
            self._window_unsafe.append(not self.checker.is_safe(dose))

        if self.n_calls % self.eval_freq == 0 and self._window_unsafe:
            unsafe_fraction = sum(self._window_unsafe) / len(self._window_unsafe)
            if wandb.run is not None:
                wandb.log(
                    {
                        "safety/unsafe_action_fraction": unsafe_fraction,
                        "timesteps": self.num_timesteps,
                    }
                )
            self._window_unsafe = []

        return True
