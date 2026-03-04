"""
Smoke test for Phase 1 — runs one complete 480-step episode and prints results.

Usage:
    python scripts/smoke_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.glucose_env import GlucoseEnv

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
    tir = (
        sum(1 for g in cgm_history if 70.0 <= g <= 180.0) / len(cgm_history) * 100.0
    )

    print("\n--- Episode Results ---")
    print(f"Total steps      : {steps}")
    print(f"Total reward     : {total_reward:.2f}")
    print(f"Final CGM        : {final_cgm:.1f} mg/dL")
    print(f"Time-in-Range    : {tir:.1f}%")
    print(f"Terminated early : {terminated}")
    print(f"Truncated        : {truncated}")


if __name__ == "__main__":
    main()
