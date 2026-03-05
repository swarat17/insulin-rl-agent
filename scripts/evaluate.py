"""
CLI benchmark runner — evaluates all trained checkpoints vs clinician baseline.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --models-dir models --n-episodes 10

Outputs:
    results/benchmark_results.csv
    results/plots/tir_comparison.png
    results/plots/safety_comparison.png
    results/plots/cgm_trajectory.png
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sac_agent import SACAgent
from src.env.glucose_env import GlucoseEnv
from src.env.patient_configs import PATIENT_CONFIGS
from src.evaluation.evaluator import Evaluator
from src.evaluation.plots import (
    plot_safety_comparison,
    plot_tir_comparison,
    plot_trajectory,
)
from src.safety.constraints import ClinicianBaseline
from src.utils.logger import get_logger

log = get_logger(__name__)

_AGENT_LOADERS = {"dqn": DQNAgent, "ppo": PPOAgent, "sac": SACAgent}


def load_agents(models_dir: str) -> list[tuple[str, object]]:
    """Scan models_dir for .zip checkpoints and load each as an agent wrapper."""
    from pathlib import Path

    agents = []
    for zip_path in sorted(Path(models_dir).glob("*.zip")):
        stem = zip_path.stem  # e.g. ppo_adult001_20240101_120000
        algo = stem.split("_")[0].lower()

        if algo not in _AGENT_LOADERS:
            log.warning(f"Unrecognised algorithm in filename '{stem}' — skipping.")
            continue

        sidecar = zip_path.with_suffix(".json")
        action_type = "discrete"
        constrained = False
        if sidecar.exists():
            with open(sidecar) as f:
                cfg = json.load(f)
            action_type = cfg.get("action_type", "discrete")
            constrained = cfg.get("constrained", False)

        # Build a clean human-readable name, deduplicating constrained variants
        label = algo.upper()
        if constrained:
            label += " (constrained)"

        # If this label already exists, skip (keep first occurrence = earliest timestamp)
        if any(n == label for n, _ in agents):
            log.info(f"Skipping duplicate {label} (keeping earlier checkpoint).")
            continue

        try:
            env = GlucoseEnv(action_type=action_type)
            agent = _AGENT_LOADERS[algo].load(str(zip_path), env=env)
            agents.append((label, agent))
            log.info(f"Loaded: {label} ({zip_path.name})")
        except Exception as exc:
            log.warning(f"Failed to load {zip_path.name}: {exc}")

    return agents


def _print_markdown_table(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    print("\n" + header)
    print(sep)
    for _, row in df.iterrows():
        values = []
        for v in row:
            values.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        print("| " + " | ".join(values) + " |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all trained agents.")
    parser.add_argument("--models-dir", default="models", help="Checkpoint directory.")
    parser.add_argument(
        "--n-episodes", type=int, default=10, help="Eval episodes per agent×patient."
    )
    args = parser.parse_args()

    # Build agent list: trained checkpoints + clinician baseline
    agents = load_agents(args.models_dir)
    agents.append(("ClinicianBaseline", ClinicianBaseline()))

    if not agents:
        log.error("No agents found. Train some models first with scripts/train.py")
        sys.exit(1)

    patient_configs = list(PATIENT_CONFIGS.values())

    evaluator = Evaluator(
        agents=agents,
        patient_configs=patient_configs,
        n_eval_episodes=args.n_episodes,
        save_dir="results",
    )

    log.info(f"Evaluating {len(agents)} agents × {len(patient_configs)} patients "
             f"× {args.n_episodes} episodes…")
    evaluator.run()

    csv_path = "results/benchmark_results.csv"
    log.info(f"Results saved to {csv_path}")
    _print_markdown_table(csv_path)

    # Generate plots
    os.makedirs("results/plots", exist_ok=True)
    df = pd.read_csv(csv_path)
    plot_tir_comparison(df)
    plot_safety_comparison(df)

    # Trajectory plot: best RL agent vs clinician on adult#001
    rl_agents = [(n, a) for n, a in agents if n != "ClinicianBaseline"]
    if rl_agents:
        best_name, best_agent = rl_agents[0]  # first loaded; replace with best by TIR
        from src.evaluation.evaluator import Evaluator as _E

        ev = _E([(best_name, best_agent), ("ClinicianBaseline", ClinicianBaseline())],
                [PATIENT_CONFIGS["adult#001"]], n_eval_episodes=1, save_dir="results")
        env_rl = GlucoseEnv(patient_name="adult#001", action_type=ev._infer_action_type(best_agent))
        env_bl = GlucoseEnv(patient_name="adult#001", action_type="discrete")
        agent_cgm, _, _ = ev._run_episode(best_agent, env_rl, seed=0)
        baseline_cgm, _, _ = ev._run_episode(ClinicianBaseline(), env_bl, seed=0)

        # Pad to 480 if episode terminated early
        agent_cgm = (agent_cgm + [agent_cgm[-1]] * 480)[:480]
        baseline_cgm = (baseline_cgm + [baseline_cgm[-1]] * 480)[:480]

        plot_trajectory(agent_cgm, baseline_cgm, agent_label=best_name)
        log.info("Plots saved to results/plots/")


if __name__ == "__main__":
    main()
