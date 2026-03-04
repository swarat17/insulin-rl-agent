"""Thin wrapper around Stable-Baselines3 PPO exposing the uniform agent interface."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    """Uniform interface wrapper for a trained PPO model."""

    def __init__(self, model: PPO) -> None:
        self.model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic action for the given observation."""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    @classmethod
    def load(cls, checkpoint_path: str, env=None) -> "PPOAgent":
        """Load a saved checkpoint and return a PPOAgent."""
        model = PPO.load(checkpoint_path, env=env)
        return cls(model)
