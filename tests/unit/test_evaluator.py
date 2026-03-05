"""Unit tests for src/evaluation/evaluator.py — Phase 5."""

from unittest.mock import patch


from src.evaluation.evaluator import Evaluator, _CSV_COLS

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PATIENT_CFG = {"patient_id": "adult#001"}
_MOCK_EPISODE = ([120.0] * 480, [0.1] * 480, 50.0)  # cgm_hist, doses, reward


class _ConstantAgent:
    """Always picks action 0 — satisfies the predict(obs) interface."""

    def predict(self, obs):
        return 0


def _make_evaluator(n_eval_episodes=1, tmp_path=None, **kwargs):
    save_dir = str(tmp_path) if tmp_path else "results"
    return Evaluator(
        agents=[("test_agent", _ConstantAgent())],
        patient_configs=[_PATIENT_CFG],
        n_eval_episodes=n_eval_episodes,
        save_dir=save_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_contains_all_metric_keys(tmp_path):
    """Result dict for one agent × patient run contains all expected CSV column names."""
    evaluator = _make_evaluator(n_eval_episodes=1, tmp_path=tmp_path)

    with patch.object(evaluator, "_run_episode", return_value=_MOCK_EPISODE):
        results = evaluator.run()

    metrics = results["test_agent"]["adult#001"]
    expected = set(_CSV_COLS) - {"agent_name", "patient"}
    assert expected == set(metrics.keys())


def test_n_eval_episodes_respected(tmp_path):
    """With n_eval_episodes=3, _run_episode is called exactly 3 times."""
    evaluator = _make_evaluator(n_eval_episodes=3, tmp_path=tmp_path)

    with patch.object(
        evaluator, "_run_episode", return_value=_MOCK_EPISODE
    ) as mock_run:
        evaluator.run()

    assert mock_run.call_count == 3


def test_csv_saved_to_results_directory(tmp_path):
    """After evaluator.run(), benchmark_results.csv exists in save_dir."""
    evaluator = _make_evaluator(n_eval_episodes=1, tmp_path=tmp_path)

    with patch.object(evaluator, "_run_episode", return_value=_MOCK_EPISODE):
        evaluator.run()

    assert (tmp_path / "benchmark_results.csv").exists()


def test_unsafe_fraction_is_in_unit_interval(tmp_path):
    """unsafe_action_fraction is always in [0.0, 1.0]."""
    # Mix of safe and unsafe doses
    doses = [0.1] * 400 + [0.9] * 80  # some above DoseConstraintChecker max (0.5)
    mock_episode = ([120.0] * 480, doses, 50.0)

    evaluator = _make_evaluator(n_eval_episodes=1, tmp_path=tmp_path)

    with patch.object(evaluator, "_run_episode", return_value=mock_episode):
        results = evaluator.run()

    frac = results["test_agent"]["adult#001"]["unsafe_action_fraction"]
    assert 0.0 <= frac <= 1.0
