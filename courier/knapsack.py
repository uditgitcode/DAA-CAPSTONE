"""Phase 2b - Box filling with 0/1 Knapsack (Dynamic Programming).

Problem: the van holds at most W kg. Each parcel i has weight w_i and delivery
value v_i (priority + deadline urgency).  Choose a subset maximising total value
with total weight <= W.  Parcels cannot be split -> 0/1 knapsack.

Recurrence (i = first i parcels, c = capacity):
    dp[i][c] = dp[i-1][c]                                   if w_i > c
    dp[i][c] = max(dp[i-1][c], dp[i-1][c-w_i] + v_i)        otherwise
Base: dp[0][c] = 0.  Answer: dp[n][W].  Traceback recovers the chosen set.

Time O(n*W), space O(n*W) for the full table (needed for traceback) and an
O(W) 1-D variant is provided for value-only queries / benchmarking.
A greedy value/weight heuristic is included as a baseline to show why DP is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .models import Parcel


@dataclass
class KnapsackResult:
    selected: List[Parcel]
    left_behind: List[Parcel]
    total_value: int
    total_weight: int
    capacity: int
    table: List[List[int]] | None = None   # full DP table (for evidence)

    @property
    def utilisation(self) -> float:
        return 100.0 * self.total_weight / self.capacity if self.capacity else 0.0


def knapsack_dp(parcels: Sequence[Parcel], capacity: int, values: Sequence[int],
                keep_table: bool = True) -> KnapsackResult:
    n = len(parcels)
    W = capacity
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = parcels[i - 1].weight, values[i - 1]
        prev, cur = dp[i - 1], dp[i]
        for c in range(W + 1):
            best = prev[c]
            if w <= c and prev[c - w] + v > best:
                best = prev[c - w] + v
            cur[c] = best

    # traceback
    chosen_idx = []
    c = W
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen_idx.append(i - 1)
            c -= parcels[i - 1].weight
    chosen_idx.reverse()
    chosen_set = set(chosen_idx)
    selected = [parcels[i] for i in chosen_idx]
    left = [p for i, p in enumerate(parcels) if i not in chosen_set]
    return KnapsackResult(
        selected=selected,
        left_behind=left,
        total_value=dp[n][W],
        total_weight=sum(p.weight for p in selected),
        capacity=W,
        table=dp if keep_table else None,
    )


def knapsack_value_1d(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Space-optimised O(W) variant (value only)."""
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
    return dp[capacity]


def knapsack_greedy(parcels: Sequence[Parcel], capacity: int, values: Sequence[int]) -> KnapsackResult:
    """Baseline: take parcels in descending value/weight ratio while they fit."""
    order = sorted(range(len(parcels)), key=lambda i: values[i] / parcels[i].weight, reverse=True)
    load, val, chosen = 0, 0, []
    for i in order:
        if load + parcels[i].weight <= capacity:
            chosen.append(i); load += parcels[i].weight; val += values[i]
    cs = set(chosen)
    return KnapsackResult(
        selected=[parcels[i] for i in sorted(chosen)],
        left_behind=[p for i, p in enumerate(parcels) if i not in cs],
        total_value=val, total_weight=load, capacity=capacity,
    )


def knapsack_bruteforce_value(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Exhaustive 2^n check - only for verifying DP on small inputs (n <= 20)."""
    n, best = len(weights), 0
    for mask in range(1 << n):
        w = v = 0
        for i in range(n):
            if mask >> i & 1:
                w += weights[i]; v += values[i]
        if w <= capacity and v > best:
            best = v
    return best
