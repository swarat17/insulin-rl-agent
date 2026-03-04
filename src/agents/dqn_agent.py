"""Thin wrapper around Stable-Baselines3 DQN exposing the uniform agent interface."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import DQN


class DQNAgent:
    """Uniform interface wrapper for a trained DQN model."""

    def __init__(self, model: DQN) -> None:
        self.model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic action for the given observation."""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    @classmethod
    def load(cls, checkpoint_path: str, env=None) -> "DQNAgent":
        """Load a saved checkpoint and return a DQNAgent."""
        model = DQN.load(checkpoint_path, env=env)
        return cls(model)
