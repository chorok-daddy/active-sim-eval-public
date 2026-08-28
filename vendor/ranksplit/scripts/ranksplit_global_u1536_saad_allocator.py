"""Independent global-U=1536 Saad-style comparator adapter.

This module is intentionally separate from
``ranksplit_established_ranking_allocators``.  The amended study needs a
fresh implementation whose denominator is an explicit part of the contract;
importing the previous task-local adapter would make the contract audit
circular.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field


FROZEN_GLOBAL_UNION_EVENTS = 1536


def stable_seed(master: int, *labels: object) -> int:
    """Return the amended study's deterministic tie-break integer."""

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
class GlobalU1536SaadState:
    """Task-wise d=1 binary-insertion adapter with one global denominator.

    The state is task-local, while the finite-sample confidence rule uses the
    frozen global padded union bound ``U=1536`` for every task.  A relation is
    certified only when the two confidence intervals are strictly separated.
    """

    task: int
    repetition: int
    master_seed: int
    capacities: tuple[int, ...]
    counts: list[int]
    successes: list[int]
    delta: float = 0.05
    global_union_events: int = FROZEN_GLOBAL_UNION_EVENTS
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
            raise ValueError("the amended Saad adapter requires exactly three policies")
        if any(capacity <= 0 for capacity in self.capacities):
            raise ValueError("policy capacities must be positive")
        if any(
            count <= 0 or count > capacity
            for count, capacity in zip(self.counts, self.capacities)
        ):
            raise ValueError("census counts must lie within policy capacities")
        if any(
            success < 0 or success > count
            for success, count in zip(self.successes, self.counts)
        ):
            raise ValueError("success counts must lie within observations")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.global_union_events != FROZEN_GLOBAL_UNION_EVENTS:
            raise ValueError("the amended contract freezes global U=1536")

    @property
    def policy_count(self) -> int:
        return len(self.counts)

    @property
    def union_events(self) -> int:
        """Return the normative global denominator, never a local capacity."""

        return self.global_union_events

    @property
    def log_factor(self) -> float:
        return math.log(2.0 * self.global_union_events / self.delta)

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
        """Choose one available policy for this already-selected task."""

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
                        "saad-global-u1536-pair-tie",
                        self.repetition,
                        global_step,
                        self.task,
                        min(pair),
                        max(pair),
                    ),
                )
            # The finite-budget endpoint can exhaust one member of an
            # unresolved pair.  Preserve the unresolved pair and continue
            # safely among available policies.
            self.stalled = True
        return _least_sampled(
            available,
            self.counts,
            master=self.master_seed,
            labels=(
                "saad-global-u1536-continuation-tie",
                self.repetition,
                global_step,
                self.task,
            ),
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


__all__ = ["FROZEN_GLOBAL_UNION_EVENTS", "GlobalU1536SaadState", "stable_seed"]
