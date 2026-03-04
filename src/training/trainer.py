"""
Unified training loop for all three RL algorithms.

Usage:
    trainer = Trainer(config, algorithm="ppo")
    checkpoint_path = trainer.run()
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import wandb
from stable_baselines3 import DQN, PPO, SAC

from src.env.glucose_env import GlucoseEnv, LagrangianGlucoseEnv
from src.safety.constraints import DoseConstraintChecker
from src.safety.lagrangian import LagrangianMultiplier
from src.training.callbacks import SafetyMonitorCallback, WandbCallback
from src.utils.logger import get_logger

log = get_logger(__name__)

_ALGO_MAP = {"dqn": DQN, "ppo": PPO, "sac": SAC}


class Trainer:
    """
    Instantiates a Stable-Baselines3 model from a config dict, attaches
    W&B and safety callbacks, trains, and saves a checkpoint with a JSON
    sidecar for reproducibility.

    Parameters
    ----------
    config : dict
        Hyperparameters and training settings (typically loaded from YAML).
    algorithm : str
        One of "dqn", "ppo", "sac".
    seed : int
        Random seed passed to the SB3 model.
    wandb_mode : str
        W&B run mode: "online", "offline", or "disabled".
        Use "disabled" in unit tests to avoid network calls.
    """

    def __init__(
        self,
        config: dict,
        algorithm: str,
        seed: int = 42,
        wandb_mode: str = "online",
    ) -> None:
        self.config = config
        self.algorithm = algorithm.lower()
        self.seed = seed
        self.wandb_mode = wandb_mode

        if self.algorithm not in _ALGO_MAP:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from {list(_ALGO_MAP)}"
            )

        patient = config.get("patient", "adult#001")
        action_type = config.get("action_type", "discrete")
        constrained = config.get("constrained", False)

        if constrained:
            self._multiplier = LagrangianMultiplier()
            self.env = LagrangianGlucoseEnv(
                multiplier=self._multiplier,
                patient_name=patient,
                action_type=action_type,
            )
        else:
            self._multiplier = None
            self.env = GlucoseEnv(patient_name=patient, action_type=action_type)

        self.eval_env = GlucoseEnv(patient_name=patient, action_type=action_type)

        self.model = self._build_model()
        self._checkpoint_path: str | None = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        cls = _ALGO_MAP[self.algorithm]
        kwargs = self._model_kwargs()
        return cls("MlpPolicy", self.env, **kwargs, seed=self.seed, verbose=0)

    def _model_kwargs(self) -> dict:
        cfg = self.config
        if self.algorithm == "dqn":
            return {
                "learning_rate": cfg.get("learning_rate", 1e-4),
                "buffer_size": cfg.get("buffer_size", 100_000),
                "exploration_fraction": cfg.get("exploration_fraction", 0.3),
                "batch_size": cfg.get("batch_size", 32),
            }
        if self.algorithm == "ppo":
            return {
                "learning_rate": cfg.get("learning_rate", 3e-4),
                "n_steps": cfg.get("n_steps", 2048),
                "batch_size": cfg.get("batch_size", 64),
                "n_epochs": cfg.get("n_epochs", 10),
            }
        if self.algorithm == "sac":
            return {
                "learning_rate": cfg.get("learning_rate", 3e-4),
                "buffer_size": cfg.get("buffer_size", 100_000),
                "batch_size": cfg.get("batch_size", 256),
            }
        return {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self, n_timesteps: int | None = None, save_dir: str = "models") -> str:
        """
        Train the model, save checkpoint + JSON sidecar, return checkpoint path.

        Parameters
        ----------
        n_timesteps : int, optional
            Override config["n_timesteps"] — useful for short test runs.
        save_dir : str
            Directory to write checkpoints into.

        Returns
        -------
        str
            Absolute path to the saved .zip checkpoint.
        """
        total_steps = n_timesteps or self.config.get("n_timesteps", 500_000)
        eval_freq = self.config.get("eval_freq", 10_000)

        run = wandb.init(
            project="insulin-rl-agent",
            config=self.config,
            mode=self.wandb_mode,
            reinit=True,
        )

        if self.wandb_mode != "disabled":
            log.info(f"W&B run: {run.url}")

        callbacks = [
            WandbCallback(
                eval_env=self.eval_env,
                eval_freq=eval_freq,
                multiplier=self._multiplier,
            ),
            SafetyMonitorCallback(
                env=self.env,
                checker=DoseConstraintChecker(),
                eval_freq=eval_freq,
            ),
        ]

        log.info(
            f"Training {self.algorithm.upper()} for {total_steps:,} steps "
            f"on patient '{self.config.get('patient', 'adult#001')}'"
        )
        self.model.learn(total_timesteps=total_steps, callback=callbacks)

        checkpoint_path = self._save(save_dir)
        wandb.finish()

        log.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def _save(self, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_slug = self.config.get("patient", "unknown").replace("#", "")
        stem = os.path.join(save_dir, f"{self.algorithm}_{patient_slug}_{timestamp}")

        self.model.save(stem)
        checkpoint_path = stem + ".zip"
        self._checkpoint_path = checkpoint_path

        sidecar = dict(self.config)
        if self._multiplier is not None:
            sidecar["final_lambda"] = self._multiplier.get_lambda()

        with open(stem + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)

        return checkpoint_path
