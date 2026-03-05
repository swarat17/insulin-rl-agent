"""Unit tests for src/evaluation/metrics.py — Phase 2."""

import pytest
from src.evaluation.metrics import (
    time_in_range,
    hypoglycemia_rate,
    severe_hypoglycemia_rate,
    compute_all_metrics,
    glucose_variability,
)

_ALL_KEYS = {
    "time_in_range",
    "hypoglycemia_rate",
    "severe_hypoglycemia_rate",
    "hyperglycemia_rate",
    "time_above_250",
    "glucose_variability",
}


def test_perfect_tir():
    readings = [100.0] * 480  # all in-range
    assert time_in_range(readings) == pytest.approx(100.0)


def test_zero_tir():
    readings = [300.0] * 480  # all hyperglycemic
    assert time_in_range(readings) == pytest.approx(0.0)


def test_hypo_rate_counts_correctly():
    readings = [60.0] * 3 + [120.0] * 7  # 3 of 10 below 70
    assert hypoglycemia_rate(readings) == pytest.approx(30.0)


def test_severe_hypo_is_subset_of_hypo():
    readings = [40.0, 55.0, 65.0, 80.0, 120.0, 200.0]
    assert severe_hypoglycemia_rate(readings) <= hypoglycemia_rate(readings)


def test_compute_all_metrics_has_all_keys():
    readings = [120.0] * 100
    result = compute_all_metrics(readings)
    assert set(result.keys()) == _ALL_KEYS


def test_variability_zero_for_constant_readings():
    readings = [150.0] * 100
    assert glucose_variability(readings) == pytest.approx(0.0)
