# Benchmark Analysis — Insulin RL Dosing Agent

> Fill this file in after running `python scripts/evaluate.py` with all 5 trained checkpoints.
> Replace every `X%` placeholder with the actual number from `results/benchmark_results.csv`.

---

## Benchmark Results

| Agent                  | TIR ↑  | Hypo Rate ↓ | Unsafe Actions ↓ | Glucose Variability ↓ |
|------------------------|--------|-------------|------------------|----------------------|
| Clinician Baseline     |  X%    |     X%      |       0%         |       X mg/dL        |
| DQN (unconstrained)    |  X%    |     X%      |      X%          |       X mg/dL        |
| PPO (unconstrained)    |  X%    |     X%      |      X%          |       X mg/dL        |
| PPO (constrained ⚡)   |  X%    |     X%      |      X%          |       X mg/dL        |
| SAC (constrained ⚡)   |  X%    |     X%      |      X%          |       X mg/dL        |

---

## Key Findings

*(Complete after running evaluate.py with trained checkpoints.)*

1. **Lagrangian constraint effectiveness**: Constrained PPO/SAC reduced unsafe dose events
   by X% relative to their unconstrained counterparts, with only a X% reduction in TIR.

2. **Best RL agent vs clinician baseline**: The best RL agent achieved X% TIR vs X% for
   the clinician baseline on the adult cohort. [Note here if the RL agent beat/matched/fell
   short of the baseline and why.]

3. **Hardest patient cohort**: The [adolescent/child/adult] cohort showed the highest
   glucose variability (X mg/dL), consistent with the clinical literature. Children
   are typically hardest to control due to high insulin sensitivity.

4. **TIR vs safety trade-off**: Adding the Lagrangian constraint reduced TIR by approximately
   X% (absolute), suggesting [a modest / a negligible / a significant] safety–performance
   trade-off. This is quantified as the cost of safety in this constrained MDP formulation.

---

## Methodology Notes

- Each agent evaluated over 10 episodes per patient cohort (30 episodes total per agent).
- `unsafe_action_fraction` counts steps where dose exceeds DoseConstraintChecker bounds
  [0.0, 0.5] U/hr. ClinicianBaseline is 0% by construction.
- Clinician baseline does not learn; it uses a fixed rule-based schedule. A lower TIR
  than a well-trained RL agent is expected.
- If DQN underperforms the clinician baseline, note that DQN is a value-based method
  designed for discrete actions and may need more timesteps to converge than PPO/SAC.

---

## Per-Cohort Notes

### Adult (adult#001)
- Most stable dynamics; expected highest TIR across all agents.
- DQN best suited here due to predictable insulin response.

### Adolescent (adolescent#001)
- High variability from puberty-driven insulin resistance changes.
- PPO/SAC may outperform DQN due to smoother policy gradients.

### Child (child#001)
- Most insulin-sensitive; small dose errors cause large excursions.
- Hypoglycemia rate will be highest here for all agents.
- Lagrangian constraint most valuable here — the downside risk is greatest.
