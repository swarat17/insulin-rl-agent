"""
Pure helper functions for the Streamlit frontend.

All functions are stateless and testable without Streamlit.
"""

from __future__ import annotations

import csv


# ---------------------------------------------------------------------------
# CGM status classification
# ---------------------------------------------------------------------------

_SEVERE_HYPO_THRESHOLD = 54.0
_HYPO_THRESHOLD = 70.0
_HYPER_THRESHOLD = 180.0
_SEVERE_HYPER_THRESHOLD = 250.0


def cgm_status(cgm: float) -> str:
    """
    Classify a CGM reading into a clinical status string.

    Returns
    -------
    str
        One of: "severe_hypo" | "hypo" | "in_range" | "hyper" | "severe_hyper"
    """
    if cgm < _SEVERE_HYPO_THRESHOLD:
        return "severe_hypo"
    if cgm < _HYPO_THRESHOLD:
        return "hypo"
    if cgm <= _HYPER_THRESHOLD:
        return "in_range"
    if cgm <= _SEVERE_HYPER_THRESHOLD:
        return "hyper"
    return "severe_hyper"


def cgm_color(status: str) -> str:
    """
    Return a hex color string for a CGM status.

    Parameters
    ----------
    status : str
        One of the strings returned by cgm_status().

    Returns
    -------
    str
        Hex color code.
    """
    _COLORS = {
        "severe_hypo": "#8B0000",   # dark red
        "hypo": "#FF4444",           # red
        "in_range": "#2ECC71",       # green
        "hyper": "#F39C12",          # orange
        "severe_hyper": "#E67E22",   # dark orange
    }
    return _COLORS.get(status, "#888888")


# ---------------------------------------------------------------------------
# TIR formatting
# ---------------------------------------------------------------------------


def format_tir(tir_float: float) -> str:
    """
    Format a TIR percentage with a clinical status emoji.

    Parameters
    ----------
    tir_float : float
        TIR as a percentage, e.g. 73.2.

    Returns
    -------
    str
        E.g. "73.2% ✅", "62.1% ⚠️", "44.0% ❌"
    """
    if tir_float >= 70.0:
        badge = "✅"
    elif tir_float >= 50.0:
        badge = "⚠️"
    else:
        badge = "❌"
    return f"{tir_float:.1f}% {badge}"


# ---------------------------------------------------------------------------
# Benchmark results loading
# ---------------------------------------------------------------------------


def load_benchmark_results(csv_path: str) -> dict:
    """
    Load benchmark_results.csv into a nested dict.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. "results/benchmark_results.csv").

    Returns
    -------
    dict
        Structure: {agent_name: {patient: {metric: value}}}
        Numeric values are cast to float; string values kept as str.
    """
    results: dict = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            agent = row["agent_name"]
            patient = row["patient"]
            metrics = {}
            for key, val in row.items():
                if key in ("agent_name", "patient"):
                    continue
                try:
                    metrics[key] = float(val)
                except (ValueError, TypeError):
                    metrics[key] = val
            results.setdefault(agent, {})[patient] = metrics
    return results


# ---------------------------------------------------------------------------
# Agent ranking
# ---------------------------------------------------------------------------


def rank_agents_by_tir(results: dict) -> list[tuple[str, float]]:
    """
    Rank agents by mean TIR across all patients, descending.

    Parameters
    ----------
    results : dict
        Output of load_benchmark_results().

    Returns
    -------
    list of (agent_name, mean_tir) tuples, sorted highest TIR first.
    """
    ranked = []
    for agent, patients in results.items():
        tirs = [metrics["tir"] for metrics in patients.values() if "tir" in metrics]
        if tirs:
            mean_tir = sum(tirs) / len(tirs)
            ranked.append((agent, mean_tir))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
