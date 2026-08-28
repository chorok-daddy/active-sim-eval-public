"""Outcome-blind task-local adapters for established ranking allocators.

The study exposes two ranking-specific reference allocations through the same
one-cell interface used by RankSplit-v2:

* ``SaadTaskwiseState`` is the disclosed d=1, task-local specialization of
  Saad, Verzelen, and Carpentier (ICML 2023).  It keeps the paper's binary
  insertion shape, but uses a frozen finite-sample confidence rule because the
  paper's native procedure is fixed-confidence rather than fixed-budget.
* ``SRankSingletonState`` is the singleton-cluster, fixed-budget specialization
  of Karpov and Zhang's SRank (NeurIPS 2020).

Neither class computes an endpoint estimate.  The shared evaluator owns the
weak-order estimate; these classes only choose the next policy within one
already-selected task.  This keeps the comparator contract auditable and makes
it impossible for a selector to consume another task's cell.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field


def stable_seed(master: int, *labels: object) -> int:
    """Return the frozen deterministic tie-break integer."""

    payload = "|".join(map(str, (master, *labels))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _least_sampled(
    candidates: list[int],
    counts: list[int],
    *,
    master: int,
    labels: tuple[object, ...],
) -> int:
    if not candidates:
        raise RuntimeError("selector has no available policy")
    minimum = min(counts[policy] for policy in candidates)
    tied = [policy for policy in candidates if counts[policy] == minimum]
    return max(
        tied,
        key=lambda policy: stable_seed(master, *labels, "policy", policy),
    )


@dataclass
class SaadTaskwiseState:
    """Finite-budget task-local binary-insertion adapter for Active Ranking.

    ``counts`` and ``successes`` include the shared one-observation census.
    The confidence radius is a fixed Hoeffding radius with a predeclared
    familywise union over every policy, task, and possible per-arm sample
    count.  A comparison that is not strictly separated remains unresolved;
    no strict relation is fabricated at a budget cap or exhausted endpoint.
    """

    task: int
    repetition: int
    master_seed: int
    capacities: tuple[int, ...]
    counts: list[int]
    successes: list[int]
    task_count: int = 4
    delta: float = 0.05
    ordered: list[int] = field(default_factory=lambda: [0])
    next_policy: int = 1
    low: int = 0
    high: int = 1
    active_pair: tuple[int, int] | None = None
    completed: bool = False
    stalled: bool = False
    certified_relations: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.capacities) != 3 or len(self.counts) != 3 or len(self.successes) != 3:
            raise ValueError("the established ranking adapters require exactly three policies")
        if any(capacity <= 0 for capacity in self.capacities):
            raise ValueError("policy capacities must be positive")
        if any(count <= 0 or count > capacity for count, capacity in zip(self.counts, self.capacities)):
            raise ValueError("census counts must lie within policy capacities")
        if any(success < 0 or success > count for success, count in zip(self.successes, self.counts)):
            raise ValueError("success counts must lie within observations")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.task_count <= 0:
            raise ValueError("task_count must be positive")

    @property
    def policy_count(self) -> int:
        return len(self.counts)

    @property
    def union_events(self) -> int:
        return self.task_count * self.policy_count * max(self.capacities)

    @property
    def log_factor(self) -> float:
        return math.log(2.0 * self.union_events / self.delta)

    def radius(self, policy: int) -> float:
        count = self.counts[policy]
        if count <= 0:
            raise ValueError("confidence radius requires a positive count")
        return math.sqrt(self.log_factor / (2.0 * count))

    def mean(self, policy: int) -> float:
        return self.successes[policy] / self.counts[policy]

    def _certified_relation(self, first: int, second: int) -> int | None:
        first_mean = self.mean(first)
        second_mean = self.mean(second)
        first_radius = self.radius(first)
        second_radius = self.radius(second)
        if first_mean - first_radius > second_mean + second_radius:
            return 1
        if second_mean - second_radius > first_mean + first_radius:
            return -1
        return None

    def _advance_certified_comparisons(self) -> None:
        """Advance binary insertion without spending an observation."""

        while not self.completed and not self.stalled:
            if self.next_policy >= self.policy_count:
                self.completed = True
                self.active_pair = None
                return
            if self.low >= self.high:
                self.ordered.insert(self.low, self.next_policy)
                self.next_policy += 1
                self.low = 0
                self.high = len(self.ordered)
                continue
            if self.active_pair is None:
                midpoint = (self.low + self.high) // 2
                self.active_pair = (self.next_policy, self.ordered[midpoint])
            first, second = self.active_pair
            relation = self._certified_relation(first, second)
            if relation is None:
                return
            self.certified_relations.append((first, second, relation))
            if relation > 0:
                self.high = self.ordered.index(second)
            else:
                self.low = self.ordered.index(second) + 1
            self.active_pair = None

    def choose_policy(self, *, global_step: int) -> int:
        """Choose one available policy for the already-selected task."""

        self._advance_certified_comparisons()
        available = [
            policy
            for policy, capacity in enumerate(self.capacities)
            if self.counts[policy] < capacity
        ]
        if not available:
            raise RuntimeError("Saad task has no remaining cells")
        if self.active_pair is not None and not self.stalled:
            pair = list(self.active_pair)
            if all(policy in available for policy in pair):
                return _least_sampled(
                    pair,
                    self.counts,
                    master=self.master_seed,
                    labels=(
                        "saad-pair-tie",
                        self.repetition,
                        global_step,
                        self.task,
                        min(pair),
                        max(pair),
                    ),
                )
            # A fixed-budget comparison can end mid-pair.  Keep the unresolved
            # state recorded, but use the disclosed safe continuation.
            self.stalled = True
        return _least_sampled(
            available,
            self.counts,
            master=self.master_seed,
            labels=("saad-continuation-tie", self.repetition, global_step, self.task),
        )

    def update(self, policy: int, outcome: int) -> None:
        if policy not in range(self.policy_count):
            raise ValueError("unknown policy")
        if self.counts[policy] >= self.capacities[policy]:
            raise RuntimeError("Saad update exceeds without-replacement capacity")
        if outcome not in (0, 1):
            raise ValueError("binary outcomes are required")
        self.counts[policy] += 1
        self.successes[policy] += int(outcome)


def srank_round_targets(*, arm_count: int, horizon: int, rounds: int) -> tuple[int, ...]:
    """Return SRank's cumulative per-active-arm targets, including zero."""

    if arm_count != 3 or rounds <= 0 or horizon <= 0:
        raise ValueError("the frozen SRank specialization requires three arms and positive settings")
    denominator = arm_count ** (1.0 + 1.0 / rounds) * rounds
    targets = [0]
    for round_index in range(1, rounds + 1):
        target = math.floor(
            arm_count ** (round_index / rounds) * horizon / denominator + 1e-12
        )
        targets.append(max(targets[-1], max(1, target)))
    return tuple(targets)


@dataclass
class SRankSingletonState:
    """Fixed-budget SRank adapter with singleton clusters ``(1, 1, 1)``."""

    task: int
    repetition: int
    master_seed: int
    capacities: tuple[int, ...]
    counts: list[int]
    successes: list[int]
    horizon: int = 204
    rounds: int = 2
    active: set[int] = field(default_factory=lambda: {0, 1, 2})
    remaining_slots: list[int] = field(default_factory=lambda: [1, 1, 1])
    assignments: list[list[int]] = field(default_factory=lambda: [[], [], []])
    round_index: int = 0
    finished: bool = False
    round_history: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.capacities) != 3 or len(self.counts) != 3 or len(self.successes) != 3:
            raise ValueError("the fixed-budget SRank anchor requires exactly three policies")
        if any(capacity <= 0 for capacity in self.capacities):
            raise ValueError("policy capacities must be positive")
        if any(count <= 0 or count > capacity for count, capacity in zip(self.counts, self.capacities)):
            raise ValueError("census counts must lie within policy capacities")
        if any(success < 0 or success > count for success, count in zip(self.successes, self.counts)):
            raise ValueError("success counts must lie within observations")
        if self.horizon <= 0 or self.rounds <= 0:
            raise ValueError("SRank horizon and rounds must be positive")
        if sum(self.remaining_slots) != len(self.active):
            raise ValueError("remaining cluster slots must match active arms")

    @property
    def round_targets(self) -> tuple[int, ...]:
        return srank_round_targets(
            arm_count=len(self.counts), horizon=self.horizon, rounds=self.rounds
        )

    def mean(self, policy: int) -> float:
        return self.successes[policy] / self.counts[policy]

    def _ranked_active(self) -> list[int]:
        return sorted(
            self.active,
            key=lambda policy: (
                self.mean(policy),
                stable_seed(
                    self.master_seed,
                    "srank-mean-tie",
                    self.repetition,
                    self.task,
                    self.round_index,
                    policy,
                ),
            ),
            reverse=True,
        )

    def _cluster_for_rank(self, rank: int) -> int:
        offset = 0
        for cluster, slots in enumerate(self.remaining_slots):
            if offset <= rank < offset + slots:
                return cluster
            offset += slots
        raise RuntimeError("active rank has no remaining singleton cluster")

    def _classify_round(self, target: int, next_active_size: int) -> None:
        ranked = self._ranked_active()
        remove_count = len(ranked) - next_active_size
        if remove_count <= 0:
            self.round_index += 1
            return
        gaps: dict[int, float] = {}
        for rank, policy in enumerate(ranked):
            if rank == 0:
                gap = self.mean(ranked[0]) - self.mean(ranked[1]) if len(ranked) > 1 else math.inf
            elif rank == len(ranked) - 1:
                gap = self.mean(ranked[-2]) - self.mean(ranked[-1])
            else:
                gap = min(
                    self.mean(ranked[rank - 1]) - self.mean(policy),
                    self.mean(policy) - self.mean(ranked[rank + 1]),
                )
            gaps[policy] = max(0.0, gap)
        selected = sorted(
            ranked,
            key=lambda policy: (
                gaps[policy],
                stable_seed(
                    self.master_seed,
                    "srank-gap-tie",
                    self.repetition,
                    self.task,
                    self.round_index,
                    policy,
                ),
            ),
            reverse=True,
        )[:remove_count]
        # Cluster membership is defined by the active ranking at the start
        # of the round.  Snapshot it before removing any arm: decrementing a
        # lower cluster first must not shift the cluster assigned to a later
        # selected rank.
        cluster_by_rank = {
            rank: self._cluster_for_rank(rank) for rank in range(len(ranked))
        }
        for policy in selected:
            rank = ranked.index(policy)
            cluster = cluster_by_rank[rank]
            self.remaining_slots[cluster] -= 1
            self.assignments[cluster].append(policy)
            self.active.remove(policy)
        self.round_history.append(
            {
                "round": self.round_index,
                "target": target,
                "removed": list(selected),
                "gaps": {str(policy): gaps[policy] for policy in ranked},
            }
        )
        self.round_index += 1
        if not self.active or self.round_index >= self.rounds:
            self.finished = not self.active

    def _advance_rounds(self) -> None:
        targets = self.round_targets
        while self.active and self.round_index < self.rounds:
            target = targets[self.round_index + 1]
            if not all(self.counts[policy] >= target for policy in self.active):
                return
            next_active_size = (
                0
                if self.round_index + 1 >= self.rounds
                else math.floor(
                    len(self.counts) ** (1.0 - (self.round_index + 1) / self.rounds)
                )
            )
            self._classify_round(target, next_active_size)
        if not self.active:
            self.finished = True

    def choose_policy(self, *, global_step: int) -> int:
        self._advance_rounds()
        available = [
            policy
            for policy, capacity in enumerate(self.capacities)
            if self.counts[policy] < capacity
        ]
        if not available:
            raise RuntimeError("SRank task has no remaining cells")
        if self.active and self.round_index < self.rounds:
            target = self.round_targets[self.round_index + 1]
            eligible = [
                policy
                for policy in self.active
                if policy in available and self.counts[policy] < target
            ]
            if eligible:
                return _least_sampled(
                    eligible,
                    self.counts,
                    master=self.master_seed,
                    labels=("srank-round-tie", self.repetition, global_step, self.task, self.round_index),
                )
        return _least_sampled(
            available,
            self.counts,
            master=self.master_seed,
            labels=("srank-continuation-tie", self.repetition, global_step, self.task),
        )

    def update(self, policy: int, outcome: int) -> None:
        if policy not in range(len(self.counts)):
            raise ValueError("unknown policy")
        if self.counts[policy] >= self.capacities[policy]:
            raise RuntimeError("SRank update exceeds without-replacement capacity")
        if outcome not in (0, 1):
            raise ValueError("binary outcomes are required")
        self.counts[policy] += 1
        self.successes[policy] += int(outcome)


__all__ = [
    "SaadTaskwiseState",
    "SRankSingletonState",
    "srank_round_targets",
    "stable_seed",
]
