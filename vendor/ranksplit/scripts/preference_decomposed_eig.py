"""Exact discrete-decision information for Beta--Bernoulli evaluation.

The module keeps the user preference inside the information objective.  For a
latent performance state ``Theta``, a deterministic terminal decision
``Z = g(Theta)``, and the next Bernoulli outcome ``Y``, it reports

    G_lambda = (1 - lambda) I(Y; Theta) + lambda I(Y; Z).

All Beta shape parameters are positive integers.  This is the conjugate state
reached from an integer Beta prior after binary observations.  Pairwise and
winner probabilities are evaluated as exact rational polynomial integrals;
only the final entropy calculation is floating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import itertools
import math
from typing import Sequence


_NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True, order=True)
class BetaPosterior:
    """Integer-shape Beta posterior for one policy's success probability."""

    alpha: int
    beta: int

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, int):
            raise TypeError("alpha must be a positive integer")
        if isinstance(self.beta, bool) or not isinstance(self.beta, int):
            raise TypeError("beta must be a positive integer")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta shape parameters must be positive")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def after(self, outcome: int) -> "BetaPosterior":
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        return BetaPosterior(
            self.alpha + outcome,
            self.beta + (1 - outcome),
        )


@dataclass(frozen=True)
class InformationDecomposition:
    """One acquisition's total, decision, residual, and preferred information."""

    total_information: float
    decision_information: float
    residual_information: float
    preference_lambda: float
    score: float


def _validate_preference_lambda(preference_lambda: float) -> None:
    if not math.isfinite(preference_lambda) or not 0.0 <= preference_lambda <= 1.0:
        raise ValueError("preference_lambda must lie within [0, 1]")


def _digamma(value: float) -> float:
    """Positive-domain digamma approximation used by the existing simulator."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError("digamma input must be finite and positive")
    result = 0.0
    while value < 8.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    inverse2 = inverse * inverse
    result += (
        math.log(value)
        - 0.5 * inverse
        - inverse2
        * (
            1.0 / 12.0
            - inverse2
            * (1.0 / 120.0 - inverse2 * (1.0 / 252.0 - inverse2 / 240.0))
        )
    )
    return result


def binary_entropy(probability: float) -> float:
    """Bernoulli entropy in nats, including the exact deterministic endpoints."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie within [0, 1]")
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log1p(-probability)


def categorical_entropy(probabilities: Sequence[float]) -> float:
    """Categorical entropy in nats."""

    if not probabilities:
        raise ValueError("at least one probability is required")
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("probabilities must be finite and nonnegative")
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"probabilities must sum to one, got {total}")
    return -math.fsum(value * math.log(value) for value in probabilities if value)


def beta_bernoulli_information(posterior: BetaPosterior) -> float:
    """Return ``I(Y; theta)`` for one posterior-predictive Bernoulli outcome."""

    alpha = float(posterior.alpha)
    beta = float(posterior.beta)
    total = alpha + beta
    mean = alpha / total
    expected_theta_log_theta = mean * (
        _digamma(alpha + 1.0) - _digamma(total + 1.0)
    )
    expected_one_minus_log = (beta / total) * (
        _digamma(beta + 1.0) - _digamma(total + 1.0)
    )
    expected_conditional_entropy = -(
        expected_theta_log_theta + expected_one_minus_log
    )
    return max(0.0, binary_entropy(mean) - expected_conditional_entropy)


def _multiply_polynomials(
    first: tuple[Fraction, ...],
    second: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    result = [Fraction(0) for _ in range(len(first) + len(second) - 1)]
    for left_degree, left_value in enumerate(first):
        if not left_value:
            continue
        for right_degree, right_value in enumerate(second):
            if right_value:
                result[left_degree + right_degree] += left_value * right_value
    return tuple(result)


@lru_cache(maxsize=None)
def _integer_beta_cdf_polynomial(posterior: BetaPosterior) -> tuple[Fraction, ...]:
    """Power-basis coefficients of an integer-shape regularized Beta CDF."""

    degree = posterior.alpha + posterior.beta - 1
    coefficients = [Fraction(0) for _ in range(degree + 1)]
    # I_x(a,b) is the upper binomial tail with n=a+b-1.
    for successes in range(posterior.alpha, degree + 1):
        outer = math.comb(degree, successes)
        remaining = degree - successes
        for expansion in range(remaining + 1):
            coefficients[successes + expansion] += Fraction(
                outer * math.comb(remaining, expansion) * ((-1) ** expansion)
            )
    return tuple(coefficients)


def _beta_moment_ratio(posterior: BetaPosterior, power: int) -> Fraction:
    """Return ``B(alpha+power,beta) / B(alpha,beta)`` exactly."""

    if power < 0:
        raise ValueError("power must be nonnegative")
    ratio = Fraction(1)
    total = posterior.alpha + posterior.beta
    for offset in range(power):
        ratio *= Fraction(posterior.alpha + offset, total + offset)
    return ratio


@lru_cache(maxsize=None)
def _winner_probabilities_cached(
    posteriors: tuple[BetaPosterior, ...],
) -> tuple[float, ...]:
    if len(posteriors) < 2:
        raise ValueError("winner identity requires at least two policies")
    probabilities: list[Fraction] = []
    for focal, posterior in enumerate(posteriors):
        other_cdfs = (Fraction(1),)
        for index, competitor in enumerate(posteriors):
            if index != focal:
                other_cdfs = _multiply_polynomials(
                    other_cdfs,
                    _integer_beta_cdf_polynomial(competitor),
                )
        probability = sum(
            coefficient * _beta_moment_ratio(posterior, power)
            for power, coefficient in enumerate(other_cdfs)
        )
        probabilities.append(probability)
    exact_total = sum(probabilities, Fraction(0))
    if exact_total != 1:
        raise ArithmeticError(f"winner probabilities do not sum to one: {exact_total}")
    return tuple(float(value) for value in probabilities)


def winner_probabilities(
    posteriors: Sequence[BetaPosterior],
) -> tuple[float, ...]:
    """Exact winner-identity probabilities for independent Beta posteriors."""

    return _winner_probabilities_cached(tuple(posteriors))


def exact_probability_cache_info() -> dict[str, int]:
    """Expose compact cache telemetry for reproducibility and runtime audits."""

    winner = _winner_probabilities_cached.cache_info()
    cdf = _integer_beta_cdf_polynomial.cache_info()
    return {
        "winner_hits": winner.hits,
        "winner_misses": winner.misses,
        "winner_size": winner.currsize,
        "cdf_hits": cdf.hits,
        "cdf_misses": cdf.misses,
        "cdf_size": cdf.currsize,
    }


def pairwise_preference_probability(
    first: BetaPosterior,
    second: BetaPosterior,
) -> float:
    """Return the exact posterior probability that the first policy is better."""

    return winner_probabilities((first, second))[0]


@lru_cache(maxsize=None)
def _three_policy_ranking_probabilities_cached(
    posteriors: tuple[BetaPosterior, BetaPosterior, BetaPosterior],
) -> tuple[float, ...]:
    """Exact probabilities of all best-to-worst orders for three policies."""

    exact_probabilities: list[Fraction] = []
    for best, middle, worst in itertools.permutations(range(3)):
        worst_cdf = _integer_beta_cdf_polynomial(posteriors[worst])
        best_cdf = _integer_beta_cdf_polynomial(posteriors[best])
        best_survival = tuple(
            (Fraction(1) if degree == 0 else Fraction(0)) - coefficient
            for degree, coefficient in enumerate(best_cdf)
        )
        order_polynomial = _multiply_polynomials(worst_cdf, best_survival)
        probability = sum(
            coefficient * _beta_moment_ratio(posteriors[middle], power)
            for power, coefficient in enumerate(order_polynomial)
        )
        exact_probabilities.append(probability)
    exact_total = sum(exact_probabilities, Fraction(0))
    if exact_total != 1:
        raise ArithmeticError(f"ranking probabilities do not sum to one: {exact_total}")
    return tuple(float(value) for value in exact_probabilities)


def three_policy_ranking_probabilities(
    posteriors: Sequence[BetaPosterior],
) -> tuple[float, ...]:
    """Return six exact probabilities in ``itertools.permutations(range(3))`` order."""

    states = tuple(posteriors)
    if len(states) != 3:
        raise ValueError("the exact ranking calculator currently requires three policies")
    return _three_policy_ranking_probabilities_cached(states)


def _decision_information_from_states(
    prior_probabilities: Sequence[float],
    success_probabilities: Sequence[float],
    failure_probabilities: Sequence[float],
    predictive_success: float,
) -> float:
    prior_entropy = categorical_entropy(prior_probabilities)
    posterior_entropy = (
        predictive_success * categorical_entropy(success_probabilities)
        + (1.0 - predictive_success) * categorical_entropy(failure_probabilities)
    )
    information = prior_entropy - posterior_entropy
    if information < -_NUMERICAL_TOLERANCE:
        raise ArithmeticError(f"negative decision information: {information}")
    return max(0.0, information)


def pairwise_decision_information(
    first: BetaPosterior,
    second: BetaPosterior,
    *,
    observed_policy: int,
) -> float:
    """Return ``I(Y; 1[theta_first > theta_second])`` in nats."""

    if observed_policy not in (0, 1):
        raise ValueError("observed_policy must be 0 or 1")
    states = (first, second)
    observed = states[observed_policy]
    success = list(states)
    failure = list(states)
    success[observed_policy] = observed.after(1)
    failure[observed_policy] = observed.after(0)
    prior_first = pairwise_preference_probability(*states)
    success_first = pairwise_preference_probability(*success)
    failure_first = pairwise_preference_probability(*failure)
    return _decision_information_from_states(
        (prior_first, 1.0 - prior_first),
        (success_first, 1.0 - success_first),
        (failure_first, 1.0 - failure_first),
        observed.mean,
    )


def winner_decision_information(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
) -> float:
    """Return mutual information between one outcome and winner identity."""

    states = tuple(posteriors)
    if not 0 <= observed_policy < len(states):
        raise ValueError("observed_policy is outside the policy set")
    observed = states[observed_policy]
    success = list(states)
    failure = list(states)
    success[observed_policy] = observed.after(1)
    failure[observed_policy] = observed.after(0)
    return _decision_information_from_states(
        winner_probabilities(states),
        winner_probabilities(success),
        winner_probabilities(failure),
        observed.mean,
    )


def ranking_decision_information(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
) -> float:
    """Return information about the joint three-policy pairwise-sign vector.

    With continuous Beta variables, the three pairwise signs are equivalent to
    one of the six strict policy rankings.  Computing categorical information
    over rankings avoids double-counting the marginal pairwise decisions.
    """

    states = tuple(posteriors)
    if len(states) != 3:
        raise ValueError("ranking decision information currently requires three policies")
    if not 0 <= observed_policy < 3:
        raise ValueError("observed_policy is outside the policy set")
    observed = states[observed_policy]
    success = list(states)
    failure = list(states)
    success[observed_policy] = observed.after(1)
    failure[observed_policy] = observed.after(0)
    return _decision_information_from_states(
        three_policy_ranking_probabilities(states),
        three_policy_ranking_probabilities(success),
        three_policy_ranking_probabilities(failure),
        observed.mean,
    )


def decompose_information(
    total_information: float,
    decision_information: float,
    *,
    preference_lambda: float,
) -> InformationDecomposition:
    """Apply the PD-EIG chain-rule weighting to compatible information terms."""

    _validate_preference_lambda(preference_lambda)
    if not math.isfinite(total_information) or total_information < 0.0:
        raise ValueError("total_information must be finite and nonnegative")
    if not math.isfinite(decision_information) or decision_information < 0.0:
        raise ValueError("decision_information must be finite and nonnegative")
    if decision_information > total_information + _NUMERICAL_TOLERANCE:
        raise ValueError("decision information cannot exceed total information")
    decision_information = min(decision_information, total_information)
    residual = total_information - decision_information
    score = decision_information + (1.0 - preference_lambda) * residual
    equivalent = (
        (1.0 - preference_lambda) * total_information
        + preference_lambda * decision_information
    )
    if not math.isclose(score, equivalent, rel_tol=0.0, abs_tol=1e-14):
        raise ArithmeticError("PD-EIG equivalent forms disagree")
    return InformationDecomposition(
        total_information=total_information,
        decision_information=decision_information,
        residual_information=residual,
        preference_lambda=preference_lambda,
        score=score,
    )


def pairwise_pd_eig(
    first: BetaPosterior,
    second: BetaPosterior,
    *,
    observed_policy: int,
    preference_lambda: float,
) -> InformationDecomposition:
    """Exact pairwise-decision PD-EIG for observing either of two policies."""

    states = (first, second)
    if observed_policy not in (0, 1):
        raise ValueError("observed_policy must be 0 or 1")
    total = beta_bernoulli_information(states[observed_policy])
    decision = pairwise_decision_information(
        first,
        second,
        observed_policy=observed_policy,
    )
    return decompose_information(
        total,
        decision,
        preference_lambda=preference_lambda,
    )


def winner_pd_eig(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> InformationDecomposition:
    """Exact winner-decision PD-EIG for observing one policy in a set."""

    states = tuple(posteriors)
    if not 0 <= observed_policy < len(states):
        raise ValueError("observed_policy is outside the policy set")
    total = beta_bernoulli_information(states[observed_policy])
    decision = winner_decision_information(states, observed_policy=observed_policy)
    return decompose_information(
        total,
        decision,
        preference_lambda=preference_lambda,
    )


def ranking_pd_eig(
    posteriors: Sequence[BetaPosterior],
    *,
    observed_policy: int,
    preference_lambda: float,
) -> InformationDecomposition:
    """Exact PD-EIG for the joint three-policy pairwise-sign decision."""

    states = tuple(posteriors)
    if len(states) != 3:
        raise ValueError("ranking PD-EIG currently requires three policies")
    if not 0 <= observed_policy < 3:
        raise ValueError("observed_policy is outside the policy set")
    total = beta_bernoulli_information(states[observed_policy])
    decision = ranking_decision_information(states, observed_policy=observed_policy)
    return decompose_information(
        total,
        decision,
        preference_lambda=preference_lambda,
    )
