from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fixed_preference import BetaPosterior, ranking_pd_eig  # noqa: E402
from ranksplit import boundary_clarity, ranksplit_score  # noqa: E402


STATES = (
    BetaPosterior(3, 4),
    BetaPosterior(5, 3),
    BetaPosterior(2, 6),
)


class RankSplitTests(unittest.TestCase):
    def test_zero_preference_matches_conventional_information(self) -> None:
        expected = ranking_pd_eig(
            STATES, observed_policy=1, preference_lambda=0.0
        ).total_information
        self.assertAlmostEqual(
            ranksplit_score(
                STATES, observed_policy=1, preference_lambda=0.0
            ),
            expected,
            places=14,
        )

    def test_score_is_finite_and_nonnegative(self) -> None:
        for policy in range(len(STATES)):
            value = ranksplit_score(
                STATES, observed_policy=policy, preference_lambda=0.5
            )
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_clarity_adapts_the_preference_weight(self) -> None:
        lam = 0.75
        clarity = boundary_clarity(STATES, 2)
        expected = ranking_pd_eig(
            STATES,
            observed_policy=2,
            preference_lambda=lam * clarity,
        ).score
        self.assertAlmostEqual(
            ranksplit_score(
                STATES, observed_policy=2, preference_lambda=lam
            ),
            expected,
            places=14,
        )

    def test_clarity_is_between_zero_and_one(self) -> None:
        for policy in range(len(STATES)):
            clarity = boundary_clarity(STATES, policy)
            self.assertGreaterEqual(clarity, 0.0)
            self.assertLessEqual(clarity, 1.0)

    def test_invalid_lambda_fails(self) -> None:
        for invalid in (-0.1, 1.1, math.inf):
            with self.assertRaises(ValueError):
                ranksplit_score(
                    STATES, observed_policy=0, preference_lambda=invalid
                )

    def test_invalid_policy_fails(self) -> None:
        with self.assertRaises(ValueError):
            boundary_clarity(STATES, 3)


if __name__ == "__main__":
    unittest.main()
