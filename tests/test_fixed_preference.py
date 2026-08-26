from __future__ import annotations

import math
import itertools
from pathlib import Path
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fixed_preference import (  # noqa: E402
    BetaPosterior,
    decompose_information,
    exact_probability_cache_info,
    pairwise_decision_information,
    pairwise_pd_eig,
    pairwise_preference_probability,
    ranking_decision_information,
    ranking_pd_eig,
    three_policy_ranking_probabilities,
    winner_decision_information,
    winner_pd_eig,
    winner_probabilities,
)


def empirical_mutual_information(samples: list[tuple[int, int]], classes: int) -> float:
    joint = [[0 for _ in range(classes)] for _ in range(2)]
    for outcome, decision in samples:
        joint[outcome][decision] += 1
    count = len(samples)
    outcome_counts = [sum(row) for row in joint]
    decision_counts = [sum(joint[y][z] for y in range(2)) for z in range(classes)]
    information = 0.0
    for outcome in range(2):
        for decision in range(classes):
            frequency = joint[outcome][decision]
            if frequency:
                information += (frequency / count) * math.log(
                    frequency * count
                    / (outcome_counts[outcome] * decision_counts[decision])
                )
    return information


class ExactProbabilityTests(unittest.TestCase):
    def test_pairwise_known_probability(self) -> None:
        self.assertAlmostEqual(
            pairwise_preference_probability(BetaPosterior(2, 1), BetaPosterior(1, 1)),
            2.0 / 3.0,
            places=14,
        )

    def test_symmetric_winner_probabilities(self) -> None:
        probabilities = winner_probabilities([BetaPosterior(3, 4)] * 3)
        for probability in probabilities:
            self.assertAlmostEqual(probability, 1.0 / 3.0, places=14)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=14)

    def test_symmetric_three_policy_rankings(self) -> None:
        probabilities = three_policy_ranking_probabilities([BetaPosterior(3, 4)] * 3)
        for probability in probabilities:
            self.assertAlmostEqual(probability, 1.0 / 6.0, places=14)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=14)

    def test_winner_probabilities_permute_with_labels(self) -> None:
        states = [BetaPosterior(2, 5), BetaPosterior(4, 3), BetaPosterior(7, 2)]
        original = winner_probabilities(states)
        permutation = (2, 0, 1)
        permuted = winner_probabilities([states[index] for index in permutation])
        self.assertEqual(
            tuple(original[index] for index in permutation),
            permuted,
        )

    def test_ranking_probabilities_permute_with_labels(self) -> None:
        states = [BetaPosterior(2, 5), BetaPosterior(4, 3), BetaPosterior(7, 2)]
        orders = list(itertools.permutations(range(3)))
        original = dict(zip(orders, three_policy_ranking_probabilities(states)))
        permutation = (2, 0, 1)
        permuted = three_policy_ranking_probabilities(
            [states[index] for index in permutation]
        )
        for new_order, probability in zip(orders, permuted):
            original_labels = tuple(permutation[index] for index in new_order)
            self.assertAlmostEqual(probability, original[original_labels], places=14)

    def test_repeated_winner_query_uses_exact_probability_cache(self) -> None:
        states = (BetaPosterior(11, 7), BetaPosterior(6, 12), BetaPosterior(9, 9))
        winner_probabilities(states)
        before = exact_probability_cache_info()
        repeated = winner_probabilities(states)
        after = exact_probability_cache_info()
        self.assertAlmostEqual(sum(repeated), 1.0, places=14)
        self.assertEqual(after["winner_hits"], before["winner_hits"] + 1)
        self.assertEqual(after["winner_misses"], before["winner_misses"])


class InformationConstitutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = BetaPosterior(5, 3)
        self.second = BetaPosterior(3, 5)

    def test_endpoints_and_chain_rule_identity(self) -> None:
        total_endpoint = pairwise_pd_eig(
            self.first,
            self.second,
            observed_policy=0,
            preference_lambda=0.0,
        )
        decision_endpoint = pairwise_pd_eig(
            self.first,
            self.second,
            observed_policy=0,
            preference_lambda=1.0,
        )
        self.assertAlmostEqual(total_endpoint.score, total_endpoint.total_information, places=14)
        self.assertAlmostEqual(
            decision_endpoint.score,
            decision_endpoint.decision_information,
            places=14,
        )
        self.assertAlmostEqual(
            total_endpoint.total_information,
            total_endpoint.decision_information + total_endpoint.residual_information,
            places=14,
        )

    def test_information_order_and_interior_convexity(self) -> None:
        result = pairwise_pd_eig(
            self.first,
            self.second,
            observed_policy=0,
            preference_lambda=0.4,
        )
        self.assertGreaterEqual(result.decision_information, 0.0)
        self.assertLessEqual(result.decision_information, result.total_information)
        self.assertGreaterEqual(result.score, result.decision_information)
        self.assertLessEqual(result.score, result.total_information)

    def test_policy_label_permutation_invariance(self) -> None:
        first = pairwise_pd_eig(
            self.first,
            self.second,
            observed_policy=0,
            preference_lambda=0.65,
        )
        relabeled = pairwise_pd_eig(
            self.second,
            self.first,
            observed_policy=1,
            preference_lambda=0.65,
        )
        self.assertAlmostEqual(first.total_information, relabeled.total_information, places=14)
        self.assertAlmostEqual(
            first.decision_information,
            relabeled.decision_information,
            places=14,
        )
        self.assertAlmostEqual(first.residual_information, relabeled.residual_information, places=14)
        self.assertAlmostEqual(first.score, relabeled.score, places=14)

    def test_winner_information_permutation_invariance(self) -> None:
        states = (BetaPosterior(2, 5), BetaPosterior(4, 3), BetaPosterior(7, 2))
        original = winner_pd_eig(
            states,
            observed_policy=1,
            preference_lambda=0.5,
        )
        permutation = (2, 0, 1)
        permuted = winner_pd_eig(
            [states[index] for index in permutation],
            observed_policy=permutation.index(1),
            preference_lambda=0.5,
        )
        self.assertAlmostEqual(original.total_information, permuted.total_information, places=14)
        self.assertAlmostEqual(
            original.decision_information,
            permuted.decision_information,
            places=14,
        )
        self.assertAlmostEqual(original.score, permuted.score, places=14)

    def test_joint_ranking_information_permutation_invariance(self) -> None:
        states = (BetaPosterior(2, 5), BetaPosterior(4, 3), BetaPosterior(7, 2))
        original = ranking_pd_eig(
            states,
            observed_policy=1,
            preference_lambda=0.65,
        )
        permutation = (2, 0, 1)
        permuted = ranking_pd_eig(
            [states[index] for index in permutation],
            observed_policy=permutation.index(1),
            preference_lambda=0.65,
        )
        self.assertAlmostEqual(original.total_information, permuted.total_information, places=14)
        self.assertAlmostEqual(
            original.decision_information,
            permuted.decision_information,
            places=14,
        )
        self.assertAlmostEqual(
            original.residual_information,
            permuted.residual_information,
            places=14,
        )
        self.assertAlmostEqual(original.score, permuted.score, places=14)

    def test_ranking_information_is_at_least_winner_information(self) -> None:
        states = (BetaPosterior(2, 5), BetaPosterior(4, 3), BetaPosterior(7, 2))
        ranking = ranking_pd_eig(
            states,
            observed_policy=1,
            preference_lambda=1.0,
        )
        winner = winner_pd_eig(
            states,
            observed_policy=1,
            preference_lambda=1.0,
        )
        self.assertGreaterEqual(
            ranking.decision_information + 1e-14,
            winner.decision_information,
        )
        self.assertLessEqual(ranking.decision_information, ranking.total_information)

    def test_invalid_preference_or_information_order_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_pd_eig(
                self.first,
                self.second,
                observed_policy=0,
                preference_lambda=1.01,
            )
        with self.assertRaises(ValueError):
            decompose_information(0.1, 0.2, preference_lambda=0.5)


class PreferenceGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.posterior_sets = (
            (BetaPosterior(2, 8), BetaPosterior(5, 5), BetaPosterior(8, 2)),
            (BetaPosterior(7, 4), BetaPosterior(4, 7), BetaPosterior(5, 5)),
            (BetaPosterior(12, 9), BetaPosterior(9, 12), BetaPosterior(6, 6)),
        )
        self.lambdas = tuple(index / 20.0 for index in range(21))

    @staticmethod
    def _actions(states, preference_lambda):
        return tuple(
            ranking_pd_eig(
                states,
                observed_policy=policy,
                preference_lambda=preference_lambda,
            )
            for policy in range(3)
        )

    def test_selected_action_follows_revealed_preference_monotonicity(self) -> None:
        for states in self.posterior_sets:
            selected = []
            for preference_lambda in self.lambdas:
                actions = self._actions(states, preference_lambda)
                index = max(range(3), key=lambda policy: (actions[policy].score, -policy))
                selected.append(actions[index])
            for first, second in zip(selected, selected[1:]):
                self.assertGreaterEqual(
                    second.decision_information + 1e-14,
                    first.decision_information,
                )
                self.assertLessEqual(
                    second.residual_information,
                    first.residual_information + 1e-14,
                )

    def test_interior_selection_is_pareto_efficient_in_total_and_ranking(self) -> None:
        for states in self.posterior_sets:
            for preference_lambda in self.lambdas[1:-1]:
                actions = self._actions(states, preference_lambda)
                selected = max(actions, key=lambda action: action.score)
                for action in actions:
                    dominates = (
                        action.total_information >= selected.total_information - 1e-14
                        and action.decision_information
                        >= selected.decision_information - 1e-14
                        and (
                            action.total_information > selected.total_information + 1e-14
                            or action.decision_information
                            > selected.decision_information + 1e-14
                        )
                    )
                    self.assertFalse(dominates)

    def test_preference_envelope_is_nonincreasing_and_convex(self) -> None:
        for states in self.posterior_sets:
            values = [
                max(
                    action.score
                    for action in self._actions(states, preference_lambda)
                )
                for preference_lambda in self.lambdas
            ]
            for first, second in zip(values, values[1:]):
                self.assertLessEqual(second, first + 1e-14)
            for left, middle, right in zip(values, values[1:], values[2:]):
                self.assertGreaterEqual(left - 2.0 * middle + right, -1e-13)


class MonteCarloAgreementTests(unittest.TestCase):
    def test_pairwise_decision_information_agrees_with_monte_carlo(self) -> None:
        first = BetaPosterior(4, 3)
        second = BetaPosterior(3, 5)
        exact = pairwise_decision_information(first, second, observed_policy=0)
        rng = random.Random(9147)
        samples = []
        for _ in range(120_000):
            theta_first = rng.betavariate(first.alpha, first.beta)
            theta_second = rng.betavariate(second.alpha, second.beta)
            outcome = int(rng.random() < theta_first)
            samples.append((outcome, int(theta_first > theta_second)))
        empirical = empirical_mutual_information(samples, classes=2)
        self.assertAlmostEqual(exact, empirical, delta=0.003)

    def test_winner_decision_information_agrees_with_monte_carlo(self) -> None:
        states = (BetaPosterior(3, 4), BetaPosterior(5, 3), BetaPosterior(2, 5))
        exact = winner_decision_information(states, observed_policy=1)
        rng = random.Random(2701)
        samples = []
        for _ in range(160_000):
            theta = [rng.betavariate(state.alpha, state.beta) for state in states]
            outcome = int(rng.random() < theta[1])
            winner = max(range(len(theta)), key=theta.__getitem__)
            samples.append((outcome, winner))
        empirical = empirical_mutual_information(samples, classes=len(states))
        self.assertAlmostEqual(exact, empirical, delta=0.004)

    def test_ranking_decision_information_agrees_with_monte_carlo(self) -> None:
        states = (BetaPosterior(3, 4), BetaPosterior(5, 3), BetaPosterior(2, 5))
        exact = ranking_decision_information(states, observed_policy=1)
        permutations = list(__import__("itertools").permutations(range(3)))
        ranking_index = {order: index for index, order in enumerate(permutations)}
        rng = random.Random(7219)
        samples = []
        for _ in range(180_000):
            theta = [rng.betavariate(state.alpha, state.beta) for state in states]
            outcome = int(rng.random() < theta[1])
            order = tuple(sorted(range(3), key=theta.__getitem__, reverse=True))
            samples.append((outcome, ranking_index[order]))
        empirical = empirical_mutual_information(samples, classes=6)
        self.assertAlmostEqual(exact, empirical, delta=0.004)


if __name__ == "__main__":
    unittest.main()
