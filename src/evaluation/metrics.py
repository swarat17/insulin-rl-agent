"""
Clinical evaluation metrics for glucose control.

All functions accept a list or numpy array of CGM readings in mg/dL and
return floats. These are the primary evaluation language for the entire
project — every agent comparison uses them.

International consensus targets (ADA / ATTD 2019):
  Time-in-Range        > 70%
  Hypoglycemia rate    < 4%
  Severe hypo rate     < 1%
"""

from __future__ import annotations

import numpy as np


def _arr(cgm_readings) -> np.ndarray:
    return np.asarray(cgm_readings, dtype=float)


def time_in_range(cgm_readings) -> float:
    """Percentage of readings in [70, 180] mg/dL. Target: > 70%."""
    a = _arr(cgm_readings)
    return float(np.mean((a >= 70.0) & (a <= 180.0)) * 100.0)


def hypoglycemia_rate(cgm_readings) -> float:
    """Percentage of readings below 70 mg/dL. Target: < 4%."""
    a = _arr(cgm_readings)
    return float(np.mean(a < 70.0) * 100.0)


def severe_hypoglycemia_rate(cgm_readings) -> float:
    """Percentage of readings below 54 mg/dL. Target: < 1%."""
    a = _arr(cgm_readings)
    return float(np.mean(a < 54.0) * 100.0)


def hyperglycemia_rate(cgm_readings) -> float:
    """Percentage of readings above 180 mg/dL."""
    a = _arr(cgm_readings)
    return float(np.mean(a > 180.0) * 100.0)


def time_above_250(cgm_readings) -> float:
    """Percentage of readings above 250 mg/dL (severe hyperglycemia)."""
    a = _arr(cgm_readings)
    return float(np.mean(a > 250.0) * 100.0)


def glucose_variability(cgm_readings) -> float:
    """
    Standard deviation of CGM readings (mg/dL).
    Lower is better — high variability is clinically dangerous
    independent of mean glucose level.
    """
    a = _arr(cgm_readings)
    return float(np.std(a))


def compute_all_metrics(cgm_readings) -> dict:
    """
    Run all clinical metrics and return a dict with consistent key names.

    This is the format logged to W&B and displayed in the Streamlit app.

    Keys
    ----
    time_in_range, hypoglycemia_rate, severe_hypoglycemia_rate,
    hyperglycemia_rate, time_above_250, glucose_variability
    """
    return {
        "time_in_range": time_in_range(cgm_readings),
        "hypoglycemia_rate": hypoglycemia_rate(cgm_readings),
        "severe_hypoglycemia_rate": severe_hypoglycemia_rate(cgm_readings),
        "hyperglycemia_rate": hyperglycemia_rate(cgm_readings),
        "time_above_250": time_above_250(cgm_readings),
        "glucose_variability": glucose_variability(cgm_readings),
    }
