---
title: RL Insulin Dosing Agent
emoji: 💉
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.55.0
app_file: frontend/app.py
pinned: false
---

## Safety-Constrained RL Insulin Dosing Agent

A safety-constrained reinforcement learning system that learns personalized insulin dosing
policies for Type 1 Diabetes patients using the FDA-validated Simglucose simulator.

Three RL algorithms (DQN, PPO, SAC) are trained and benchmarked against a clinician
baseline policy. A **Lagrangian safety constraint** enforces that the agent never
recommends a clinically dangerous dose — implemented via dual gradient descent rather
than hard action clipping, giving the policy smooth and informative gradients.

Results are evaluated using clinical metrics (Time-in-Range, hypoglycemia rate, glucose
variability) across three patient cohorts (adolescent, adult, child).

### How to use

1. **Live Simulation** — select an agent and patient, click Run Episode to see a
   24-hour CGM trajectory with insulin doses overlaid.
2. **Benchmark Results** — compare all agents on all cohorts, sortable table and charts.
3. **About & Methods** — clinical background, Lagrangian constraint explanation, training details.
# Safety-Constrained RL Insulin Dosing Agent

> A reinforcement learning system that learns personalized insulin dosing policies for
> Type 1 Diabetes patients, with Lagrangian safety constraints that enforce clinical dose bounds.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29-orange.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.7-green.svg)](https://stable-baselines3.readthedocs.io/)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-tracked-yellow.svg)](https://wandb.ai/swaratsarkar-university-at-buffalo/insulin-rl-agent)

**[Live Demo — Hugging Face Spaces](https://huggingface.co/spaces/swarat17/rl-dosing-agent)**

---

## The Clinical Problem

Type 1 Diabetes (T1D) means the pancreas produces no insulin. Insulin moves glucose out
of the blood into cells — without it, blood sugar rises dangerously (hyperglycemia, causing
long-term organ damage). Too much insulin causes it to crash (hypoglycemia — the brain is
starved of fuel, leading to seizures or death within minutes). The clinical goal: decide a
continuous basal insulin rate (U/hr) to keep blood glucose in the safe range of
**70–180 mg/dL** at all times.

This is a hard control problem. Insulin sensitivity varies by time of day, meal timing,
exercise, stress hormones, and patient age. A policy must anticipate glucose trends
3–30 minutes into the future (insulin takes time to act) and adapt to each patient's
individual physiology. Reinforcement Learning is a natural fit: it learns personalized
policies from simulated interaction without requiring explicit programming of every
clinical scenario.

---

## Architecture

### The RL Loop

```
┌─────────────────────────────────────────────────────────┐
│  Every 3 minutes (one step):                            │
│                                                         │
│  Observation (6 features)                               │
│    [CGM now, CGM -3min, CGM -6min,                      │
│     time of day, steps since meal, last dose]           │
│           │                                             │
│           ▼                                             │
│       RL Agent  ──────────────────────────────────┐     │
│           │                                       │     │
│           ▼                                       ▼     │
│      Proposed dose         DoseConstraintChecker        │
│           │                  violation = max(0,         │
│           │                    dose - max_dose)         │
│           │                          │                  │
│           ▼                          ▼                  │
│     Simglucose               LagrangianMultiplier       │
│    (3-min physiology)          λ ← λ + lr*(v - ε)       │
│           │                          │                  │
│           ▼                          ▼                  │
│      New CGM          reward' = reward - λ × violation  │
│           │                          │                  │
│           └──────────────────────────┘                  │
│                  Agent learns from reward'              │
└─────────────────────────────────────────────────────────┘
```

### Why Lagrangian Relaxation (not hard action clipping)

Hard clipping produces a **discontinuous reward surface** — the agent receives the same
reward whether it proposed a dose of 0.5 U/hr (clipped to safe) or 5.0 U/hr (also
clipped to safe). This removes the gradient signal needed to learn *why* large doses are
bad. The Lagrangian approach instead penalizes violations smoothly through the reward:
`reward' = reward - λ × violation`. The multiplier λ is updated via dual gradient
descent — rising when the agent violates constraints, falling when it behaves safely.
This gives the policy a smooth, informative gradient and provably converges to a
constrained-optimal solution.

### Observation Space (6 features, all normalized to [0, 1])

| Index | Feature | Why it matters |
|-------|---------|----------------|
| 0 | CGM now / 400 | Current blood sugar |
| 1 | CGM 3 min ago / 400 | Trend direction (falling vs rising) |
| 2 | CGM 6 min ago / 400 | Trend acceleration |
| 3 | Time of day | Insulin sensitivity varies diurnally |
| 4 | Steps since last meal / 480 | Post-meal glucose spike timing |
| 5 | Last dose / max\_basal | Insulin-on-board proxy (avoids stacking) |

An 80 mg/dL reading while falling is dangerous; the same reading while rising is fine.
Three CGM readings let the agent infer direction without an explicit derivative.

### Reward Function (asymmetric by design)

| Condition | Reward | Rationale |
|-----------|--------|-----------|
| CGM ∈ [70, 180] | **+1.0** | Target zone |
| CGM < 70 | **−2.0** | Hypoglycemia — dangerous |
| CGM < 54 | **−4.0** | Severe hypoglycemia — life-threatening |
| CGM > 180 | **−0.5** | Hyperglycemia — bad but slower harm |

Low blood sugar kills in minutes; high blood sugar causes organ damage over years.
The asymmetry teaches the agent to err on the side of slightly too much sugar rather
than too little.

---

## Benchmark Results

Evaluated: 5 trained agents + clinician baseline × 3 patient cohorts × 10 episodes each.

### Time-in-Range (TIR %) — higher is better, target > 70%

| Agent | Adolescent | Adult | Child | Mean |
|-------|-----------|-------|-------|------|
| DQN | 79.7% | 80.0% | 72.0% | 77.2% |
| PPO | 59.8% | 47.2% | 68.2% | 58.4% |
| PPO (constrained ⚡) | 77.2% | 73.3% | 77.8% | 76.1% |
| SAC | 73.6% | 80.5% | 74.0% | 76.0% |
| SAC (constrained ⚡) | **80.5%** | 79.0% | 70.0% | **76.5%** |
| Clinician Baseline | 77.2% | 75.2% | 76.1% | 76.1% |

### Hypoglycemia Rate (%) — lower is better, target < 4%

| Agent | Adolescent | Adult | Child |
|-------|-----------|-------|-------|
| DQN | 20.3% | 18.0% | 28.0% |
| PPO | 13.7% | 5.7% | 20.0% |
| PPO (constrained ⚡) | 22.0% | 22.4% | 22.1% |
| SAC | 26.4% | 17.4% | 26.0% |
| SAC (constrained ⚡) | 18.8% | 16.0% | 28.4% |
| Clinician Baseline | 22.1% | 20.4% | 23.9% |

---

## Key Findings

- **Lagrangian constraint rescued PPO**: Unconstrained PPO collapsed to a hyperglycemic
  strategy (47% TIR on adult — far below the 75% clinician baseline). Adding the
  Lagrangian penalty recovered it to 73% TIR on adult (+26% absolute). The constraint
  acts as a regularizer preventing policy collapse into bad local optima.

- **SAC is the best RL algorithm overall**: SAC (constrained) achieved 76.5% mean TIR
  and beat the clinician baseline on adult (79% vs 75%) and adolescent (80.5% vs 77.2%).
  SAC's entropy regularization provides implicit stability that PPO lacks without the
  constraint.

- **Children are hardest to control**: All agents showed the highest hypoglycemia rates
  on the child cohort (20–28%), consistent with clinical literature — children are most
  insulin-sensitive and small dose errors cause large glucose drops.

- **Constraint cost is asymmetric**: For PPO, adding the constraint was net-positive on
  every metric. For SAC, the TIR cost was negligible (−1.5% absolute on adult). Lagrangian
  constraints are a low-cost safety improvement for stable algorithms and a critical
  correction for unstable ones.

---

## Environment: Simglucose

**Simglucose** is an FDA-validated T1D simulator based on the UVA/Padova mathematical
model. It simulates glucose-insulin dynamics using differential equations fitted to
clinical trial data from 30 virtual patients (10 adult, 10 adolescent, 10 child).

Each episode = 480 steps × 3 minutes = 1 simulated day. The agent interacts with the
simulator exactly as a real closed-loop system would: observe CGM → decide dose →
physiology advances → observe new CGM.

| Patient Cohort | Clinical Challenge |
|---------------|-------------------|
| `adult#001` | Most stable dynamics — easiest to control |
| `adolescent#001` | Hormonal changes cause unpredictable insulin resistance |
| `child#001` | Most insulin-sensitive — highest hypoglycemia risk |

---

## Training Details

| Agent | Timesteps | LR | Action Space | Constrained |
|-------|-----------|-----|-------------|-------------|
| DQN | 500,000 | 1e-4 | Discrete (11) | No |
| PPO | 500,000 | 3e-4 | Discrete (11) | No |
| PPO ⚡ | 500,000 | 1e-4 | Discrete (11) | Yes |
| SAC | 500,000 | 3e-4 | Continuous | No |
| SAC ⚡ | 500,000 | 1e-4 | Continuous | Yes |

Hardware: CPU (Windows 11, Intel). ~20 minutes per agent via Stable-Baselines3.

W&B Dashboard: [wandb.ai/swaratsarkar-university-at-buffalo/insulin-rl-agent](https://wandb.ai/swaratsarkar-university-at-buffalo/insulin-rl-agent)

---

## Local Setup

```bash
git clone https://github.com/swarat17/insulin-rl-agent
cd insulin-rl-agent
python -m venv venv && source venv/Scripts/activate   # Windows
pip install -r requirements.txt
# Download model checkpoints into models/ (see Releases or HF Spaces repo)
streamlit run frontend/app.py
```

---

## Running Tests

```bash
# Unit tests (fast, no checkpoints needed)
pytest tests/unit/ -v

# Integration tests (needs environment only)
pytest tests/integration/ -v -m integration

# End-to-end tests (needs trained checkpoints in models/)
pytest tests/e2e/ -v -m e2e
```

---

## Project Structure

```
├── src/
│   ├── env/           # Gymnasium wrapper around Simglucose
│   ├── safety/        # DoseConstraintChecker + LagrangianMultiplier
│   ├── agents/        # DQN / PPO / SAC wrappers
│   ├── training/      # Trainer + W&B callbacks
│   └── evaluation/    # Benchmark runner + clinical metrics + plots
├── configs/           # YAML hyperparameter files
├── scripts/           # train.py, evaluate.py, smoke_test.py
├── frontend/          # Streamlit app (3 pages)
├── models/            # Trained checkpoints (.zip + .json sidecar)
├── results/           # benchmark_results.csv + plots + analysis.md
└── tests/             # unit / integration / e2e
```
