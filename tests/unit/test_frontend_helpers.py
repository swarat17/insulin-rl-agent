"""Unit tests for frontend/helpers.py — Phase 6."""

import csv
import os

import pytest

from frontend.helpers import (
    cgm_color,
    cgm_status,
    format_tir,
    load_benchmark_results,
    rank_agents_by_tir,
)


def test_cgm_status_in_range():
    assert cgm_status(120.0) == "in_range"


def test_cgm_status_hypo():
    assert cgm_status(65.0) == "hypo"


def test_cgm_status_severe_hypo():
    assert cgm_status(50.0) == "severe_hypo"


def test_cgm_status_hyper():
    assert cgm_status(200.0) == "hyper"


def test_format_tir_good_shows_checkmark():
    assert "✅" in format_tir(75.0)


def test_format_tir_poor_shows_cross():
    assert "❌" in format_tir(45.0)


def test_rank_agents_sorted_descending():
    results = {
        "AgentA": {"adult#001": {"tir": 60.0}},
        "AgentB": {"adult#001": {"tir": 80.0}},
        "AgentC": {"adult#001": {"tir": 70.0}},
    }
    ranked = rank_agents_by_tir(results)
    tirs = [t for _, t in ranked]
    assert tirs == sorted(tirs, reverse=True)


def test_load_benchmark_results_top_level_keys(tmp_path):
    csv_path = tmp_path / "benchmark_results.csv"
    fieldnames = ["agent_name", "patient", "tir", "hypo_rate"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"agent_name": "PPO", "patient": "adult#001", "tir": "72.0", "hypo_rate": "3.0"})
        writer.writerow({"agent_name": "DQN", "patient": "adult#001", "tir": "77.0", "hypo_rate": "18.0"})

    results = load_benchmark_results(str(csv_path))
    assert "PPO" in results
    assert "DQN" in results
