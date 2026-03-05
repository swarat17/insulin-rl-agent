"""
Multi-agent benchmark runner.

Evaluator
---------
For each (agent_name, agent) × patient combination, runs n_eval_episodes
complete episodes and averages the clinical metrics across them.

Supports both RL agent wrappers (with .predict(obs)) and the ClinicianBaseline
(with .recommend(cgm, time_of_day)).

Results are saved to results/benchmark_results.csv with columns:
    agent_name | patient | tir | hypo_rate | severe_hypo_rate |
    hyperglycemia_rate | glucose_variability | unsafe_action_fraction |
    mean_episode_reward
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from gymnasium import spaces

from src.env.glucose_env import GlucoseEnv
from src.evaluation.metrics import compute_all_metrics
from src.safety.constraints import DoseConstraintChecker

# Ordered CSV column names
_CSV_COLS = [
    "agent_name",
    "patient",
    "tir",
    "hypo_rate",
    "severe_hypo_rate",
    "hyperglycemia_rate",
    "glucose_variability",
    "unsafe_action_fraction",
    "mean_episode_reward",
]

# Mapping from compute_all_metrics keys → short CSV column names
_METRIC_MAP = {
    "time_in_range": "tir",
    "hypoglycemia_rate": "hypo_rate",
    "severe_hypoglycemia_rate": "severe_hypo_rate",
    "hyperglycemia_rate": "hyperglycemia_rate",
    "glucose_variability": "glucose_variability",
}


class Evaluator:
    """
    Benchmarks a list of agents across one or more patient cohorts.

    Parameters
    ----------
    agents : list of (name, agent) tuples
        Each agent must expose either:
        - `predict(obs) -> action`  — RL agent wrappers (DQNAgent, PPOAgent, SACAgent)
        - `recommend(cgm, time_of_day) -> dose_uhr`  — ClinicianBaseline
    patient_configs : list of dicts
        Each dict must have a ``patient_id`` key (e.g. "adult#001").
    n_eval_episodes : int
        Number of episodes to average per agent × patient cell.
    save_dir : str
        Directory where ``benchmark_results.csv`` is written.
    """

    def __init__(
        self,
        agents: list[tuple[str, object]],
        patient_configs: list[dict],
        n_eval_episodes: int = 10,
        save_dir: str = "results",
    ) -> None:
        self.agents = agents
        self.patient_configs = patient_configs
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.checker = DoseConstraintChecker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run the full benchmark and return a nested result dict.

        Returns
        -------
        dict
            ``{agent_name: {patient_id: {metric: value}}}``
            Values are averaged across ``n_eval_episodes`` episodes.
        """
        results: dict = {}
        all_rows: list[dict] = []

        for agent_name, agent in self.agents:
            results[agent_name] = {}
            action_type = self._infer_action_type(agent)

            for patient_cfg in self.patient_configs:
                patient_id = patient_cfg["patient_id"]
                env = GlucoseEnv(patient_name=patient_id, action_type=action_type)

                episode_metrics: list[dict] = []
                for ep_idx in range(self.n_eval_episodes):
                    cgm_hist, doses, total_reward = self._run_episode(
                        agent, env, seed=ep_idx
                    )
                    raw = compute_all_metrics(cgm_hist)
                    violations = self.checker.batch_violations(doses)
                    n_unsafe = sum(1 for v in violations if v > 0)
                    unsafe_frac = n_unsafe / len(violations) if violations else 0.0

                    episode_metrics.append(
                        {
                            **{short: raw[long] for long, short in _METRIC_MAP.items()},
                            "unsafe_action_fraction": float(
                                np.clip(unsafe_frac, 0.0, 1.0)
                            ),
                            "mean_episode_reward": float(total_reward),
                        }
                    )

                avg = {
                    k: float(np.mean([m[k] for m in episode_metrics]))
                    for k in episode_metrics[0]
                }
                results[agent_name][patient_id] = avg
                all_rows.append(
                    {"agent_name": agent_name, "patient": patient_id, **avg}
                )

        self._save_csv(all_rows)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(
        self, agent, env: GlucoseEnv, seed: int | None = None
    ) -> tuple[list[float], list[float], float]:
        """Run one episode; return (cgm_history, doses_uhr, total_reward)."""
        obs, _ = env.reset(seed=seed)
        cgm_history: list[float] = []
        doses: list[float] = []
        total_reward = 0.0
        done = False

        while not done:
            if hasattr(agent, "predict"):
                action = agent.predict(obs)
                dose = self._action_to_dose(action, env)
            else:
                # ClinicianBaseline
                cgm = float(obs[0]) * 400.0
                time_of_day = float(obs[3])
                dose = float(agent.recommend(cgm, time_of_day))
                action = self._dose_to_action(dose, env)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            cgm_history.append(info["cgm"])
            doses.append(dose)

        return cgm_history, doses, total_reward

    def _action_to_dose(self, action, env: GlucoseEnv) -> float:
        if env.action_type == "discrete":
            return float(env.DOSE_LEVELS[int(np.atleast_1d(action).flat[0])])
        return (
            float(np.clip(np.atleast_1d(action).flat[0], 0.0, 1.0)) * env.max_basal_dose
        )

    def _dose_to_action(self, dose: float, env: GlucoseEnv):
        if env.action_type == "discrete":
            return int(np.argmin(np.abs(env.DOSE_LEVELS - dose)))
        return np.array(
            [np.clip(dose / env.max_basal_dose, 0.0, 1.0)], dtype=np.float32
        )

    def _infer_action_type(self, agent) -> str:
        """Detect action type from the SB3 model's action space, defaulting to discrete."""
        if hasattr(agent, "model") and hasattr(agent.model, "action_space"):
            if isinstance(agent.model.action_space, spaces.Box):
                return "continuous"
        return "discrete"

    def _save_csv(self, rows: list[dict]) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        csv_path = os.path.join(self.save_dir, "benchmark_results.csv")
        pd.DataFrame(rows, columns=_CSV_COLS).to_csv(csv_path, index=False)
