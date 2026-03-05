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
