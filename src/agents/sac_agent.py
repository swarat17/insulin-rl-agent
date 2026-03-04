"""Thin wrapper around Stable-Baselines3 SAC exposing the uniform agent interface."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import SAC


class SACAgent:
    """Uniform interface wrapper for a trained SAC model."""

    def __init__(self, model: SAC) -> None:
        self.model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic action for the given observation."""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    @classmethod
    def load(cls, checkpoint_path: str, env=None) -> "SACAgent":
        """Load a saved checkpoint and return a SACAgent."""
        model = SAC.load(checkpoint_path, env=env)
        return cls(model)
