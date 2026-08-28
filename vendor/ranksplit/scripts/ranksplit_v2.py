"""Post-confirmation RankSplit-v2 development objectives.

These functions are not part of the frozen 2026-07-23 confirmation.  They
test three result-bound compositions of the frozen exact information split.
The leading continuation computes posterior boundary clarity directly; the
older ambiguity-weighted objectives remain development comparators only.
"""

from __future__ import annotations

import math
from typing import Sequence

from preference_decomposed_eig import BetaPosterior, ranking_pd_eig
from screen_preference_decomposed_eig import ambiguity_weight


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
    """Return weakest normalized posterior separation from a competitor."""

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


def boundary_endpoint_score(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> float:
    """Interpolate from total information to boundary-gated ranking information."""

    lam = _validate_lambda(preference_lambda)
    decomposition = ranking_pd_eig(
        posteriors,
        observed_policy=observed_policy,
        preference_lambda=lam,
    )
    ambiguity = ambiguity_weight(tuple(posteriors), observed_policy)
    return (
        (1.0 - lam) * decomposition.total_information
        + lam * ambiguity * decomposition.decision_information
    )


def serial_gated_score(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> float:
    """Apply boundary gating and the exact RankSplit interpolation in series."""

    lam = _validate_lambda(preference_lambda)
    decomposition = ranking_pd_eig(
        posteriors,
        observed_policy=observed_policy,
        preference_lambda=lam,
    )
    ambiguity = ambiguity_weight(tuple(posteriors), observed_policy)
    gate = (1.0 - lam) + lam * ambiguity
    return gate * decomposition.score


def clarity_continuation_score(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> float:
    """Apply RankSplit through a posterior boundary-clarity continuation."""

    lam = _validate_lambda(preference_lambda)
    clarity = boundary_clarity(posteriors, observed_policy)
    effective_lambda = lam * clarity
    return ranking_pd_eig(
        posteriors,
        observed_policy=observed_policy,
        preference_lambda=effective_lambda,
    ).score


def score(
    family: str,
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> float:
    """Dispatch one frozen development family."""

    if family == "boundary-endpoint":
        return boundary_endpoint_score(
            posteriors,
            observed_policy=observed_policy,
            preference_lambda=preference_lambda,
        )
    if family == "serial-gated":
        return serial_gated_score(
            posteriors,
            observed_policy=observed_policy,
            preference_lambda=preference_lambda,
        )
    if family == "clarity-continuation":
        return clarity_continuation_score(
            posteriors,
            observed_policy=observed_policy,
            preference_lambda=preference_lambda,
        )
    raise ValueError(f"unknown RankSplit-v2 family: {family}")
