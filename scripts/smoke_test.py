"""
Smoke test for Phases 1 & 2 — runs one complete episode and prints results.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.glucose_env import GlucoseEnv
from src.evaluation.metrics import compute_all_metrics

PATIENT = "adult#001"
FIXED_ACTION = 2  # discrete level 2 ≈ 0.1 U/hr


def main():
    env = GlucoseEnv(patient_name=PATIENT, action_type="discrete")
    print(f"Patient      : {PATIENT}")
    print(f"Fixed action : {FIXED_ACTION}  ({env.DOSE_LEVELS[FIXED_ACTION]:.3f} U/hr)")
    print(f"Max steps    : {env.max_steps}")

    obs, info = env.reset(seed=42)

    total_reward = 0.0
    cgm_history = []
    steps = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(FIXED_ACTION)
        total_reward += reward
        cgm_history.append(info["cgm"])
        steps += 1

    final_cgm = cgm_history[-1]
    metrics = compute_all_metrics(cgm_history)

    print("\n--- Episode Results ---")
    print(f"Total steps      : {steps}")
    print(f"Total reward     : {total_reward:.2f}")
    print(f"Final CGM        : {final_cgm:.1f} mg/dL")
    print(f"Terminated early : {terminated}")
    print(f"Truncated        : {truncated}")
    print("\n--- Clinical Metrics ---")
    for key, val in metrics.items():
        unit = " mg/dL" if key == "glucose_variability" else "%"
        print(f"  {key:<28}: {val:.2f}{unit}")


if __name__ == "__main__":
    main()
