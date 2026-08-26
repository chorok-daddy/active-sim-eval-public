"""RankSplit scoring for three-policy Beta posteriors.

The public implementation is intentionally small: it measures the posterior
separation of the observed policy from its closest competitor, then uses that
clarity to adapt the preference weight in the exact ranking information
calculation.
"""

from __future__ import annotations

import math
from typing import Sequence

if __package__:
    from .fixed_preference import BetaPosterior, ranking_pd_eig
else:
    from fixed_preference import BetaPosterior, ranking_pd_eig


def _validate_lambda(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("preference_lambda must lie within [0, 1]")
    return value


def _beta_variance(state: BetaPosterior) -> float:
    total = state.alpha + state.beta
    return state.alpha * state.beta / (total * total * (total + 1.0))


def boundary_clarity(
    posteriors: Sequence[BetaPosterior],
    observed_policy: int,
) -> float:
    """Return the weakest normalized posterior separation from a competitor."""

    if not 0 <= observed_policy < len(posteriors):
        raise ValueError("observed_policy is outside the policy set")
    values = []
    for competitor in range(len(posteriors)):
        if competitor == observed_policy:
            continue
        scale = math.sqrt(
            _beta_variance(posteriors[observed_policy])
            + _beta_variance(posteriors[competitor])
        )
        gap = abs(
            posteriors[observed_policy].mean - posteriors[competitor].mean
        )
        values.append(gap / (gap + scale))
    if not values:
        raise ValueError("boundary clarity requires at least two policies")
    return min(values)


def ranksplit_score(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> float:
    """Return the clarity-adaptive RankSplit acquisition score."""

    lam = _validate_lambda(preference_lambda)
    clarity = boundary_clarity(posteriors, observed_policy)
    effective_lambda = lam * clarity
    return ranking_pd_eig(
        posteriors,
        observed_policy=observed_policy,
        preference_lambda=effective_lambda,
    ).score


__all__ = ["boundary_clarity", "ranksplit_score"]
