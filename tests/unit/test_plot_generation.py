"""Unit tests for src/evaluation/plots.py — Phase 5."""

import os

import pandas as pd
import pytest

from src.evaluation.plots import plot_tir_comparison, plot_trajectory

_EPISODE_STEPS = 480


def _dummy_results_df():
    return pd.DataFrame(
        [
            {
                "agent_name": "PPO",
                "patient": "adult#001",
                "tir": 72.0,
                "hypo_rate": 3.0,
                "severe_hypo_rate": 0.5,
                "hyperglycemia_rate": 25.0,
                "glucose_variability": 28.0,
                "unsafe_action_fraction": 0.0,
                "mean_episode_reward": 150.0,
            },
            {
                "agent_name": "ClinicianBaseline",
                "patient": "adult#001",
                "tir": 65.0,
                "hypo_rate": 5.0,
                "severe_hypo_rate": 1.0,
                "hyperglycemia_rate": 30.0,
                "glucose_variability": 32.0,
                "unsafe_action_fraction": 0.0,
                "mean_episode_reward": 130.0,
            },
        ]
    )


def test_tir_comparison_plot_file_created(tmp_path):
    """plot_tir_comparison saves a file at the specified path."""
    save_path = str(tmp_path / "tir_comparison.png")
    plot_tir_comparison(_dummy_results_df(), save_path=save_path)
    assert os.path.exists(save_path)


def test_trajectory_plot_requires_full_episode():
    """Passing fewer than 480 CGM readings raises a ValueError."""
    short_cgm = [120.0] * 100  # too short
    full_cgm = [120.0] * _EPISODE_STEPS

    with pytest.raises(ValueError, match="480"):
        plot_trajectory(short_cgm, full_cgm)

    with pytest.raises(ValueError, match="480"):
        plot_trajectory(full_cgm, short_cgm)
