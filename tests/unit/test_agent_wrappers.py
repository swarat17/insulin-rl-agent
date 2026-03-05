"""Unit tests for src/agents/ wrappers — Phase 3."""

import numpy as np
from stable_baselines3 import DQN, PPO, SAC

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sac_agent import SACAgent
from src.env.glucose_env import GlucoseEnv


def _make_obs() -> np.ndarray:
    return np.ones(6, dtype=np.float32) * 0.3


def test_predict_returns_value_in_action_space(tmp_path):
    """Load a saved checkpoint, call predict(obs) — returned value is within action space."""
    env = GlucoseEnv(action_type="discrete")

    # Train a tiny PPO model and save it
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    model.learn(64)
    path = str(tmp_path / "ppo_test")
    model.save(path)

    agent = PPOAgent.load(path, env=env)
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs)

    action_val = int(np.atleast_1d(action).flat[0])
    assert 0 <= action_val < env.action_space.n


def test_uniform_interface_across_algorithms(tmp_path):
    """DQN, PPO, and SAC wrappers all respond to predict(obs) without raising."""
    obs = _make_obs()

    # DQN — discrete
    env_d = GlucoseEnv(action_type="discrete")
    dqn_model = DQN(
        "MlpPolicy", env_d, learning_starts=0, batch_size=32, verbose=0
    )
    dqn_model.learn(64)
    p = str(tmp_path / "dqn_test")
    dqn_model.save(p)
    DQNAgent.load(p, env=env_d).predict(obs)

    # PPO — discrete
    ppo_model = PPO(
        "MlpPolicy", env_d, n_steps=64, batch_size=32, n_epochs=1, verbose=0
    )
    ppo_model.learn(64)
    p = str(tmp_path / "ppo_test")
    ppo_model.save(p)
    PPOAgent.load(p, env=env_d).predict(obs)

    # SAC — continuous
    env_c = GlucoseEnv(action_type="continuous")
    sac_model = SAC("MlpPolicy", env_c, batch_size=32, verbose=0)
    sac_model.learn(200)  # SAC needs learning_starts (100) steps before training
    p = str(tmp_path / "sac_test")
    sac_model.save(p)
    SACAgent.load(p, env=env_c).predict(obs)
