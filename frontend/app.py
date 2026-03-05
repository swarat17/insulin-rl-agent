"""
Streamlit frontend — 3-page app.

Pages:
  1. Live Simulation  — run any agent on any patient, see CGM trajectory
  2. Benchmark Results — compare all agents on all cohorts
  3. About & Methods  — project overview, Lagrangian explanation, links
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.helpers import (
    cgm_color,
    cgm_status,
    format_tir,
    load_benchmark_results,
    rank_agents_by_tir,
)
from src.env.glucose_env import GlucoseEnv
from src.env.patient_configs import PATIENT_CONFIGS
from src.evaluation.metrics import compute_all_metrics
from src.safety.constraints import ClinicianBaseline, DoseConstraintChecker

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RL Insulin Dosing Agent",
    page_icon="💉",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CHECKER = DoseConstraintChecker()
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "benchmark_results.csv",
)


@st.cache_resource
def _load_all_agents():
    """Scan models/ and load all agent checkpoints + clinician baseline."""
    from pathlib import Path

    from src.agents.dqn_agent import DQNAgent
    from src.agents.ppo_agent import PPOAgent
    from src.agents.sac_agent import SACAgent

    _LOADERS = {"dqn": DQNAgent, "ppo": PPOAgent, "sac": SACAgent}
    agents = {}

    for zip_path in sorted(Path(_MODELS_DIR).glob("*.zip")):
        stem = zip_path.stem
        algo = stem.split("_")[0].lower()
        if algo not in _LOADERS:
            continue

        sidecar = zip_path.with_suffix(".json")
        action_type = "discrete"
        constrained = False
        if sidecar.exists():
            with open(sidecar) as f:
                cfg = json.load(f)
            action_type = cfg.get("action_type", "discrete")
            constrained = cfg.get("constrained", False)

        label = algo.upper() + (" (constrained)" if constrained else "")
        if label in agents:
            continue

        try:
            env = GlucoseEnv(action_type=action_type)
            agent = _LOADERS[algo].load(str(zip_path), env=env)
            agents[label] = (agent, action_type)
        except Exception:
            pass

    agents["Clinician Baseline"] = (ClinicianBaseline(), "discrete")
    return agents


def _run_episode(agent, action_type: str, patient_name: str, seed: int = 42):
    """Run one full episode. Returns (cgm_history, dose_history, metrics_dict)."""
    env = GlucoseEnv(patient_name=patient_name, action_type=action_type)
    obs, _ = env.reset(seed=seed)
    cgm_hist, dose_hist, total_reward = [], [], 0.0

    for _ in range(480):
        if hasattr(agent, "predict"):
            action = agent.predict(obs)
        else:
            cgm_raw = obs[0] * 400.0
            tod = obs[3]
            action = agent.recommend(cgm_raw, tod)
            # map dose back to discrete index
            idx = int(round(action / 0.5 * 10))
            action = max(0, min(10, idx))

        # Compute dose from action before stepping (info dict has no "dose" key)
        if action_type == "discrete":
            dose = float(env.DOSE_LEVELS[int(action)])
        else:
            import numpy as np
            dose = float(np.clip(np.atleast_1d(action)[0], 0.0, 1.0)) * env.max_basal_dose

        obs, reward, terminated, truncated, info = env.step(action)
        cgm = info.get("cgm", obs[0] * 400.0)
        cgm_hist.append(cgm)
        dose_hist.append(dose)
        total_reward += reward
        if terminated or truncated:
            break

    # Pad to 480 if early termination
    pad = [cgm_hist[-1]] * (480 - len(cgm_hist))
    cgm_hist += pad
    dose_hist += [0.0] * (480 - len(dose_hist))

    metrics = compute_all_metrics(cgm_hist)
    unsafe = sum(1 for d in dose_hist if not _CHECKER.is_safe(d)) / len(dose_hist)
    metrics["unsafe_action_fraction"] = unsafe
    metrics["mean_episode_reward"] = total_reward

    env.close()
    return cgm_hist, dose_hist, metrics


def _cgm_trajectory_fig(agent_cgm, baseline_cgm, agent_label: str) -> go.Figure:
    steps = list(range(480))
    hours = [s * 3 / 60 for s in steps]

    fig = go.Figure()

    # Dynamic y-axis: 20 mg/dL padding around data, minimum bottom 40, cap top at 350
    all_cgm = agent_cgm + baseline_cgm
    y_min = 0
    y_max = min(350, max(all_cgm) + 20)

    # Shading
    fig.add_hrect(y0=y_min, y1=70, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.07, line_width=0)
    fig.add_hrect(y0=180, y1=y_max, fillcolor="orange", opacity=0.07, line_width=0)

    # Boundary lines
    fig.add_hline(y=70, line_dash="dash", line_color="red",
                  annotation_text="Hypo (70)", annotation_position="right")
    fig.add_hline(y=180, line_dash="dash", line_color="orange",
                  annotation_text="Hyper (180)", annotation_position="right")

    # Trajectories
    fig.add_trace(go.Scatter(x=hours, y=agent_cgm, mode="lines",
                             name=agent_label, line=dict(color="royalblue", width=2)))
    fig.add_trace(go.Scatter(x=hours, y=baseline_cgm, mode="lines",
                             name="Clinician Baseline",
                             line=dict(color="tomato", width=2, dash="dash")))

    fig.update_layout(
        title="CGM Trajectory — 24-hour Episode",
        xaxis_title="Time (hours)",
        yaxis_title="CGM (mg/dL)",
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(range=[0, 24]),
        legend=dict(x=1.01, y=1, xanchor="left"),
        height=420,
    )
    return fig


def _dose_fig(dose_hist: list[float], label: str) -> go.Figure:
    hours = [s * 3 / 60 for s in range(len(dose_hist))]
    fig = go.Figure(go.Scatter(x=hours, y=dose_hist, mode="lines",
                               fill="tozeroy", name=label,
                               line=dict(color="steelblue")))
    fig.update_layout(
        title="Insulin Dose Over Time",
        xaxis_title="Time (hours)",
        yaxis_title="Dose (U/hr)",
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1: Live Simulation
# ---------------------------------------------------------------------------

def page_live_simulation():
    st.header("Live Simulation")
    st.write(
        "Select an agent and patient cohort, then click **Run Episode** to simulate "
        "a full 24-hour insulin dosing episode."
    )

    agents = _load_all_agents()
    patient_names = list(PATIENT_CONFIGS.keys())

    col1, col2, col3 = st.columns(3)
    with col1:
        agent_label = st.selectbox("Agent", [k for k in agents.keys() if k != "Clinician Baseline"])
    with col2:
        patient_name = st.selectbox("Patient Cohort", patient_names)
    with col3:
        seed = st.number_input("Episode Seed", min_value=0, max_value=99, value=42,
                               help="Change seed to see a different episode. "
                                    "Benchmark results are averaged over 10 seeds (0–9).")

    st.caption(
        "⚠️ Single-episode results can differ significantly from the benchmark table, "
        "which averages 10 episodes. High-variance agents like DQN may crash on some seeds."
    )

    if st.button("Run Episode", type="primary"):
        agent, action_type = agents[agent_label]

        with st.spinner("Simulating 24-hour episode…"):
            try:
                agent_cgm, agent_doses, metrics = _run_episode(
                    agent, action_type, patient_name, seed=int(seed)
                )
                baseline_cgm, _, _ = _run_episode(
                    ClinicianBaseline(), "discrete", patient_name, seed=int(seed)
                )
            except Exception as exc:
                st.warning(f"Episode failed: {exc}")
                return

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time-in-Range", format_tir(metrics["time_in_range"]))
        c2.metric("Hypo Rate", f"{metrics['hypoglycemia_rate']:.1f}%")
        c3.metric("Glucose Variability", f"{metrics['glucose_variability']:.1f} mg/dL")
        c4.metric("Unsafe Actions", f"{metrics['unsafe_action_fraction']*100:.1f}%")

        # Trajectory chart
        st.plotly_chart(
            _cgm_trajectory_fig(agent_cgm, baseline_cgm, agent_label),
            use_container_width=True,
        )

        # Dose chart
        st.plotly_chart(_dose_fig(agent_doses, agent_label), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: Benchmark Results
# ---------------------------------------------------------------------------

def page_benchmark():
    st.header("Benchmark Results")

    if not os.path.exists(_RESULTS_CSV):
        st.warning("benchmark_results.csv not found. Run `python scripts/evaluate.py` first.")
        return

    df = pd.read_csv(_RESULTS_CSV)

    # Summary callout — composite score: TIR - hypo_rate (higher is better clinically)
    results = load_benchmark_results(_RESULTS_CSV)
    composite = []
    for agent, patients in results.items():
        tirs = [m["tir"] for m in patients.values() if "tir" in m]
        hypos = [m["hypo_rate"] for m in patients.values() if "hypo_rate" in m]
        if tirs and hypos:
            score = sum(tirs) / len(tirs) - sum(hypos) / len(hypos)
            mean_tir = sum(tirs) / len(tirs)
            mean_hypo = sum(hypos) / len(hypos)
            composite.append((agent, score, mean_tir, mean_hypo))
    composite.sort(key=lambda x: x[1], reverse=True)

    if composite:
        best_name, _, best_tir, best_hypo = composite[0]
        baseline_tirs = [v["tir"] for v in results.get("ClinicianBaseline", {}).values()]
        baseline_mean = sum(baseline_tirs) / len(baseline_tirs) if baseline_tirs else 0.0
        st.info(
            f"**Best agent (TIR − hypo rate):** {best_name} — "
            f"{best_tir:.1f}% avg TIR, {best_hypo:.1f}% hypo rate  "
            f"vs {baseline_mean:.1f}% clinician baseline TIR"
        )

    # Full results table
    st.subheader("Full Results Table")
    st.dataframe(df.style.format({c: "{:.2f}" for c in df.select_dtypes("float").columns}),
                 use_container_width=True)

    # TIR grouped bar chart
    st.subheader("Time-in-Range by Agent and Patient Cohort")
    agents = df["agent_name"].unique()
    patients = df["patient"].unique()

    tir_fig = go.Figure()
    for agent in agents:
        sub = df[df["agent_name"] == agent]
        tirs = [
            float(sub[sub["patient"] == p]["tir"].values[0])
            if p in sub["patient"].values else 0.0
            for p in patients
        ]
        tir_fig.add_trace(go.Bar(name=agent, x=list(patients), y=tirs))

    tir_fig.add_hline(y=70, line_dash="dash", line_color="green",
                      annotation_text="ADA target (70%)")
    tir_fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 105], title="TIR (%)"),
        xaxis_title="Patient Cohort",
        height=420,
    )
    st.plotly_chart(tir_fig, use_container_width=True)

    # Glucose variability chart
    st.subheader("Glucose Variability (Std Dev) by Agent and Patient Cohort")
    st.caption(
        "Lower is better. High variability means erratic glucose swings, which is clinically "
        "dangerous even when the mean is in range. PPO unconstrained is a clear outlier — its "
        "policy chases hyperglycemia, causing wild oscillations."
    )
    var_fig = go.Figure()
    for agent in agents:
        sub = df[df["agent_name"] == agent]
        vals = [
            float(sub[sub["patient"] == p]["glucose_variability"].values[0])
            if p in sub["patient"].values else 0.0
            for p in patients
        ]
        var_fig.add_trace(go.Bar(name=agent, x=list(patients), y=vals))

    var_fig.update_layout(
        barmode="group",
        yaxis=dict(title="Glucose Variability (mg/dL std dev)"),
        xaxis_title="Patient Cohort",
        height=400,
    )
    st.plotly_chart(var_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3: About & Methods
# ---------------------------------------------------------------------------

def page_about():
    st.header("About & Methods")

    st.markdown("""
### The Clinical Problem

Type 1 Diabetes (T1D) patients lack the ability to produce insulin, requiring exogenous
insulin delivery to control blood glucose. Dosing is extraordinarily difficult: too little
insulin leaves blood glucose dangerously high (hyperglycemia → long-term organ damage),
while too much causes hypoglycemia — blood glucose crashes that can cause seizures, loss of
consciousness, or death within minutes. The complexity is compounded by variable insulin
absorption, meals, exercise, sleep, and stress hormones — making optimal dosing a
non-stationary, partially observable control problem. Reinforcement Learning is a natural
fit: it can learn personalized policies from interaction with a patient model without
explicit programming of every clinical scenario.

### Environment: Simglucose

We use **Simglucose**, an FDA-validated T1D simulator based on the UVA/Padova mathematical
model. It simulates glucose-insulin dynamics in 30 virtual patients across three cohorts
(adolescent, adult, child) using differential equations derived from clinical trial data.
Each episode is 480 steps of 3-minute intervals (1 simulated day). The agent observes
a 6-dimensional state: recent CGM readings, time of day, steps since last meal, and the
last dose taken. Actions are insulin basal rates in U/hr, capped at 0.5 U/hr.

### Lagrangian Safety Constraint

The Lagrangian approach solves a constrained MDP:
> **maximize** expected reward **subject to** expected constraint violation ≤ ε

Rather than hard-clipping unsafe actions (which produces a discontinuous reward surface),
we augment the reward with a penalty: `reward' = reward - λ × violation`. The multiplier
λ is updated via dual gradient descent — rising when the agent violates constraints,
falling when it behaves safely. This gives the policy a smooth, informative gradient signal
and provably converges to a constrained-optimal solution.
""")

    # Lambda curve image
    lambda_img = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "plots", "lambda_curve.png"
    )
    if os.path.exists(lambda_img):
        st.image(lambda_img, width=600, caption=(
            "Lagrangian multiplier λ during constrained PPO training. "
            "λ rises when the agent violates safety constraints and falls as it learns safe behavior."
        ))
    else:
        st.info("Lambda curve not available (train a constrained agent and export λ history from W&B).")

    st.markdown("""
### Training Details

| Agent | Timesteps | Action Space | Constrained |
|-------|-----------|-------------|-------------|
| DQN   | 500,000   | Discrete (11) | No |
| PPO   | 500,000   | Discrete (11) | No |
| PPO ⚡ | 500,000  | Discrete (11) | Yes (Lagrangian) |
| SAC   | 500,000   | Continuous   | No |
| SAC ⚡ | 500,000  | Continuous   | Yes (Lagrangian) |

Hardware: CPU (Windows 11). All training via Stable-Baselines3.

### Links

- **GitHub:** [github.com/swarat17/insulin-rl-agent](https://github.com/swarat17/insulin-rl-agent)
- **W&B Dashboard:** [wandb.ai/swaratsarkar-university-at-buffalo/insulin-rl-agent](https://wandb.ai/swaratsarkar-university-at-buffalo/insulin-rl-agent)
""")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PAGES = {
    "Live Simulation": page_live_simulation,
    "Benchmark Results": page_benchmark,
    "About & Methods": page_about,
}

st.sidebar.title("💉 Insulin Dosing RL Agent")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", list(_PAGES.keys()))
_PAGES[page]()
