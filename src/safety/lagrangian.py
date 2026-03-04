"""
Lagrangian multiplier for safety-constrained RL.

The constrained MDP formulation:
    maximise  E[reward]
    subject to  E[constraint_violation] ≤ epsilon

The Lagrangian relaxation solves this by adding -λ * violation to the reward,
where λ is learned via dual gradient descent:

    λ ← clamp(λ + lr_lambda * (mean_violation - epsilon), 0, lambda_max)

When violation > epsilon, λ increases → the penalty grows → the agent is
discouraged from unsafe actions through the reward signal (not by blocking them).
When violation ≤ epsilon, λ decreases toward zero → penalty fades.

This gives the policy a smooth, informative gradient rather than the
discontinuous reward surface produced by hard action masking.
"""

from __future__ import annotations

import numpy as np


class LagrangianMultiplier:
    """
    Maintains a scalar Lagrange multiplier λ and updates it via dual
    gradient descent based on observed constraint violations.

    Parameters
    ----------
    lambda_init : float
        Initial value of λ. Default 0.01.
    lr_lambda : float
        Dual learning rate. Default 0.001.
    epsilon : float
        Acceptable constraint budget (mean violation threshold). Default 0.05.
    lambda_max : float
        Hard upper cap on λ to prevent runaway penalisation. Default 10.0.
    """

    def __init__(
        self,
        lambda_init: float = 0.01,
        lr_lambda: float = 0.001,
        epsilon: float = 0.05,
        lambda_max: float = 10.0,
    ) -> None:
        self._lambda = float(lambda_init)
        self.lr_lambda = lr_lambda
        self.epsilon = epsilon
        self.lambda_max = lambda_max

    def update(self, constraint_violations: list[float]) -> None:
        """
        Update λ using the mean violation of a batch.

        Parameters
        ----------
        constraint_violations : list[float]
            Violation magnitudes for a batch of actions (e.g. one full episode).
            Zero for safe actions, positive for unsafe ones.
        """
        mean_violation = float(np.mean(constraint_violations))
        new_lambda = self._lambda + self.lr_lambda * (mean_violation - self.epsilon)
        self._lambda = float(np.clip(new_lambda, 0.0, self.lambda_max))

        # Log to W&B if a run is active (no-op when wandb_mode="disabled")
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"lagrangian/lambda": self._lambda})
        except Exception:
            pass

    def augment_reward(self, reward: float, violation: float) -> float:
        """
        Subtract the Lagrangian penalty from the base reward.

        Returns
        -------
        float
            `reward - λ * violation`. When violation is 0.0, reward is unchanged.
        """
        return reward - self._lambda * violation

    def get_lambda(self) -> float:
        """Return the current value of λ."""
        return self._lambda
