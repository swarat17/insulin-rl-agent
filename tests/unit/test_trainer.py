"""Unit tests for src/training/trainer.py — Phase 3."""

import json
import os

import yaml
from stable_baselines3 import PPO

from src.training.trainer import Trainer

# Minimal config for fast test runs
_FAST_PPO_CONFIG = {
    "algorithm": "ppo",
    "patient": "adult#001",
    "action_type": "discrete",
    "learning_rate": 3e-4,
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 1,
    "n_timesteps": 64,
    "eval_freq": 10_000,  # high enough that no eval fires during short run
}


def test_instantiates_correct_algorithm():
    trainer = Trainer(_FAST_PPO_CONFIG, algorithm="ppo", wandb_mode="disabled")
    assert isinstance(trainer.model, PPO)


def test_checkpoint_saved_after_short_run(tmp_path):
    trainer = Trainer(_FAST_PPO_CONFIG, algorithm="ppo", wandb_mode="disabled")
    checkpoint_path = trainer.run(n_timesteps=64, save_dir=str(tmp_path))

    assert checkpoint_path.endswith(".zip")
    assert os.path.exists(checkpoint_path)


def test_sidecar_json_saved_alongside_checkpoint(tmp_path):
    trainer = Trainer(_FAST_PPO_CONFIG, algorithm="ppo", wandb_mode="disabled")
    checkpoint_path = trainer.run(n_timesteps=64, save_dir=str(tmp_path))

    sidecar_path = checkpoint_path.replace(".zip", ".json")
    assert os.path.exists(sidecar_path)

    with open(sidecar_path) as f:
        data = json.load(f)
    assert "n_timesteps" in data


def test_config_loaded_from_yaml():
    with open("configs/ppo.yaml") as f:
        config = yaml.safe_load(f)
    assert "n_timesteps" in config
