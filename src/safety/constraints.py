"""
Dose safety constraints and clinician baseline policy.

DoseConstraintChecker
---------------------
Validates and clips insulin doses against configurable clinical bounds.
Used by the Lagrangian safety layer (Phase 4) and the evaluator (Phase 5).

ClinicianBaseline
-----------------
A rule-based (non-ML) policy that mirrors typical clinical basal insulin
programming: a low constant basal with brief post-meal elevations around
standard breakfast (08:00), lunch (12:00), and dinner (18:00) windows.
This sets the performance bar that RL agents must beat.
"""

from __future__ import annotations


class DoseConstraintChecker:
    """
    Checks whether an insulin dose (U/hr) lies within clinical bounds.

    Parameters
    ----------
    min_dose : float
        Minimum allowable basal rate (U/hr). Default 0.0.
    max_dose : float
        Maximum allowable basal rate (U/hr). Default 0.5.
    """

    def __init__(self, min_dose: float = 0.0, max_dose: float = 0.5) -> None:
        assert min_dose <= max_dose, "min_dose must be <= max_dose"
        self.min_dose = min_dose
        self.max_dose = max_dose

    def is_safe(self, action_uhr: float) -> bool:
        """Return True if dose is within [min_dose, max_dose]."""
        return self.min_dose <= action_uhr <= self.max_dose

    def clip_to_safe(self, action_uhr: float) -> float:
        """Clip dose to [min_dose, max_dose] without raising."""
        return float(max(self.min_dose, min(self.max_dose, action_uhr)))

    def constraint_violation(self, action_uhr: float) -> float:
        """
        Magnitude of bounds exceedance. Returns 0.0 when safe.

        The Lagrangian multiplier (Phase 4) consumes this signal to
        penalise unsafe actions through the reward rather than blocking them.
        """
        if action_uhr < self.min_dose:
            return float(self.min_dose - action_uhr)
        if action_uhr > self.max_dose:
            return float(action_uhr - self.max_dose)
        return 0.0

    def batch_violations(self, actions: list[float]) -> list[float]:
        """Vectorised constraint_violation over a full episode trajectory."""
        return [self.constraint_violation(a) for a in actions]


# ---------------------------------------------------------------------------
# Clinician baseline
# ---------------------------------------------------------------------------

# Post-meal windows (normalized time-of-day in [0, 1]; 1.0 == 24 h)
# Breakfast  08:00–10:00  →  0.333–0.417
# Lunch      12:00–14:00  →  0.500–0.583
# Dinner     18:00–20:00  →  0.750–0.833
_MEAL_WINDOWS = [
    (8 / 24, 10 / 24),   # breakfast
    (12 / 24, 14 / 24),  # lunch
    (18 / 24, 20 / 24),  # dinner
]

_BASE_BASAL = 0.1    # U/hr — overnight / inter-meal rate
_MEAL_BASAL = 0.25   # U/hr — elevated rate in post-meal windows


class ClinicianBaseline:
    """
    Rule-based insulin policy that mimics a simple clinical basal programme.

    Logic:
        - Constant low basal rate (_BASE_BASAL) at most hours.
        - Slightly elevated rate (_MEAL_BASAL) in the two-hour windows
          after standard breakfast, lunch, and dinner.

    This is the performance bar RL agents must beat on TIR and safety metrics.
    """

    def recommend(self, cgm: float, time_of_day: float) -> float:
        """
        Parameters
        ----------
        cgm : float
            Current CGM reading (mg/dL). Unused in this simple rule; included
            for interface symmetry with future adaptive baselines.
        time_of_day : float
            Normalised time in [0, 1] (0.0 = midnight, 1.0 = next midnight).

        Returns
        -------
        float
            Recommended basal rate in U/hr.
        """
        for start, end in _MEAL_WINDOWS:
            if start <= time_of_day < end:
                return _MEAL_BASAL
        return _BASE_BASAL
