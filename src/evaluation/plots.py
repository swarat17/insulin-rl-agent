"""
Plot generation for evaluation results.

All functions write a PNG to the specified path and return the path.
Use matplotlib with the 'Agg' backend so plots work headlessly (CI / remote).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_EPISODE_STEPS = 480  # expected length for trajectory plots


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# ---------------------------------------------------------------------------
# TIR comparison
# ---------------------------------------------------------------------------


def plot_tir_comparison(
    results_df: pd.DataFrame,
    save_path: str = "results/plots/tir_comparison.png",
) -> str:
    """
    Grouped bar chart — TIR % by agent, bars grouped by patient cohort.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns: agent_name, patient, tir.
    save_path : str
        Output file path.
    """
    _ensure_dir(save_path)

    agents = results_df["agent_name"].unique()
    patients = results_df["patient"].unique()
    x = np.arange(len(patients))
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        subset = results_df[results_df["agent_name"] == agent]
        tirs = [
            (
                float(subset[subset["patient"] == p]["tir"].values[0])
                if p in subset["patient"].values
                else 0.0
            )
            for p in patients
        ]
        ax.bar(x + i * width - 0.4 + width / 2, tirs, width, label=agent)

    ax.axhline(70, color="green", linestyle="--", linewidth=1, label="ADA target (70%)")
    ax.set_xlabel("Patient Cohort")
    ax.set_ylabel("Time-in-Range (%)")
    ax.set_title("Time-in-Range by Agent and Patient Cohort")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Safety comparison
# ---------------------------------------------------------------------------


def plot_safety_comparison(
    results_df: pd.DataFrame,
    save_path: str = "results/plots/safety_comparison.png",
) -> str:
    """
    Grouped bar chart — unsafe action fraction by agent, grouped by patient cohort.
    Constrained agents should be visibly lower than unconstrained variants.
    """
    _ensure_dir(save_path)

    agents = results_df["agent_name"].unique()
    patients = results_df["patient"].unique()
    x = np.arange(len(patients))
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        subset = results_df[results_df["agent_name"] == agent]
        fracs = [
            (
                float(
                    subset[subset["patient"] == p]["unsafe_action_fraction"].values[0]
                )
                if p in subset["patient"].values
                else 0.0
            )
            for p in patients
        ]
        ax.bar(x + i * width - 0.4 + width / 2, fracs, width, label=agent)

    ax.set_xlabel("Patient Cohort")
    ax.set_ylabel("Unsafe Action Fraction")
    ax.set_title("Unsafe Action Fraction by Agent and Patient Cohort")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# CGM trajectory
# ---------------------------------------------------------------------------


def plot_trajectory(
    agent_cgm: list[float],
    baseline_cgm: list[float],
    agent_label: str = "Best RL Agent",
    save_path: str = "results/plots/cgm_trajectory.png",
) -> str:
    """
    Single-episode CGM trajectory for one RL agent vs clinician baseline.

    Parameters
    ----------
    agent_cgm : list of float
        480 CGM readings from the RL agent episode.
    baseline_cgm : list of float
        480 CGM readings from the clinician baseline episode.
    agent_label : str
        Legend label for the RL agent line.
    save_path : str
        Output file path.

    Raises
    ------
    ValueError
        If either CGM array has fewer than 480 readings.
    """
    if len(agent_cgm) < _EPISODE_STEPS:
        raise ValueError(
            f"agent_cgm must have at least {_EPISODE_STEPS} readings, "
            f"got {len(agent_cgm)}"
        )
    if len(baseline_cgm) < _EPISODE_STEPS:
        raise ValueError(
            f"baseline_cgm must have at least {_EPISODE_STEPS} readings, "
            f"got {len(baseline_cgm)}"
        )

    _ensure_dir(save_path)

    steps = np.arange(_EPISODE_STEPS)
    hours = steps * 3 / 60  # 3-minute steps → hours

    fig, ax = plt.subplots(figsize=(14, 5))

    # Background shading
    ax.axhspan(0, 70, alpha=0.12, color="red", label="_nolegend_")
    ax.axhspan(70, 180, alpha=0.10, color="green", label="_nolegend_")
    ax.axhspan(180, 400, alpha=0.08, color="orange", label="_nolegend_")

    # Clinical boundary dashed lines
    ax.axhline(
        70, color="red", linestyle="--", linewidth=1.0, label="Hypo threshold (70)"
    )
    ax.axhline(
        180,
        color="orange",
        linestyle="--",
        linewidth=1.0,
        label="Hyper threshold (180)",
    )

    # Trajectories
    ax.plot(
        hours,
        agent_cgm[:_EPISODE_STEPS],
        color="royalblue",
        linewidth=1.5,
        label=agent_label,
    )
    ax.plot(
        hours,
        baseline_cgm[:_EPISODE_STEPS],
        color="tomato",
        linewidth=1.5,
        linestyle="--",
        label="Clinician Baseline",
    )

    # Legend patches for shading
    green_patch = mpatches.Patch(color="green", alpha=0.3, label="In-range [70–180]")
    red_patch = mpatches.Patch(color="red", alpha=0.3, label="Hypoglycemia (<70)")
    orange_patch = mpatches.Patch(
        color="orange", alpha=0.3, label="Hyperglycemia (>180)"
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [green_patch, red_patch, orange_patch],
        loc="upper right",
        fontsize=8,
    )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("CGM (mg/dL)")
    ax.set_title("CGM Trajectory: RL Agent vs Clinician Baseline (24-hour episode)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 400)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Lambda curve
# ---------------------------------------------------------------------------


def plot_lambda_curve(
    lambda_history: list[float],
    timesteps: list[int] | None = None,
    save_path: str = "results/plots/lambda_curve.png",
) -> str:
    """
    Plot the Lagrangian multiplier λ over training timesteps.

    Parameters
    ----------
    lambda_history : list of float
        λ values recorded at evaluation intervals.
    timesteps : list of int, optional
        Corresponding timestep values. If None, uses episode indices.
    save_path : str
        Output file path.
    """
    _ensure_dir(save_path)

    x = timesteps if timesteps is not None else list(range(len(lambda_history)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, lambda_history, color="purple", linewidth=1.5)
    ax.fill_between(x, 0, lambda_history, alpha=0.15, color="purple")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("λ (Lagrange Multiplier)")
    ax.set_title(
        "Lagrangian Multiplier λ During Constrained PPO Training\n"
        "(rises when agent violates constraints, falls when safe)"
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
