"""Integration test — requires environment setup but no pre-trained checkpoints."""

import pytest

from src.training.trainer import Trainer

_PPO_CONFIG = {
    "algorithm": "ppo",
    "patient": "adult#001",
    "action_type": "discrete",
    "learning_rate": 3e-4,
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 1,
    "n_timesteps": 1000,
    "eval_freq": 10_000,
}


@pytest.mark.integration
def test_ppo_trains_1000_steps_without_error():
    """Trainer with PPO config runs model.learn(1000) to completion without exception."""
    trainer = Trainer(_PPO_CONFIG, algorithm="ppo", seed=42, wandb_mode="disabled")
    trainer.model.learn(1000)  # must not raise
