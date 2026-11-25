"""Phase 3b - Delivery route ordering (TSP approximation).

Visiting every selected stop once and returning to the depot is the Travelling
Salesman Problem (NP-hard).  We work on the k x k shortest-path matrix produced
by Dijkstra (a metric: it satisfies the triangle inequality).

  * nearest_neighbour   O(k^2)  - greedy: always drive to the closest unvisited stop.
  * two_opt             O(k^2) per pass - local search that removes crossing legs
                                  (used as the Phase-4 optimisation).
  * held_karp           O(k^2 2^k) - exact DP, only for k <= ~12, used to measure
                                  the approximation ratio of the heuristics.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

Matrix = Sequence[Sequence[float]]


def tour_length(tour: Sequence[int], d: Matrix) -> float:
    return sum(d[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def nearest_neighbour(d: Matrix, start: int = 0) -> List[int]:
    """Closed tour of matrix indices beginning and ending at `start`."""
    k = len(d)
    unvisited = set(range(k)) - {start}
    tour = [start]
    cur = start
    while unvisited:
        # tie-break on index for determinism
        nxt = min(unvisited, key=lambda j: (d[cur][j], j))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour.append(start)
    return tour


def two_opt(tour: List[int], d: Matrix, max_passes: int = 50) -> Tuple[List[int], int]:
    """Improve a closed tour by reversing segments while it shortens the tour.
    Returns (tour, number_of_improving_moves)."""
    t = list(tour)
    n = len(t)
    moves = 0
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            a, b = t[i - 1], t[i]
            for j in range(i + 1, n - 1):
                c, e = t[j], t[j + 1]
                delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
                if delta < -1e-9:
                    t[i:j + 1] = reversed(t[i:j + 1])
                    b = t[i]
                    moves += 1
                    improved = True
        if not improved:
            break
    return t, moves


def deadline_banded_route(d: Matrix, stop_deadline: Sequence[int], start: int = 0,
                          improve: bool = True) -> List[int]:
    """Phase-4 optimisation: deadline-aware route.

    Stops are grouped into bands by their earliest parcel deadline (1h, 2h, ...).
    Bands are served in deadline order; inside each band the van uses nearest
    neighbour from wherever it currently is, then 2-opt tidies each band segment
    with its two end points fixed (so the band order is never broken).
    Trades a few km for far fewer late parcels.
    """
    bands = sorted({stop_deadline[i] for i in range(len(d)) if i != start})
    tour, cur, seg_bounds = [start], start, []
    for b in bands:
        todo = {i for i in range(len(d)) if i != start and stop_deadline[i] == b}
        seg_start = len(tour) - 1
        while todo:
            nxt = min(todo, key=lambda j: (d[cur][j], j))
            tour.append(nxt); todo.remove(nxt); cur = nxt
        seg_bounds.append((seg_start, len(tour)))
    tour.append(start)
    if improve:
        for s, e in seg_bounds:
            # path tour[s] .. tour[e] with both ends fixed
            seg = tour[s:e + 1]
            if len(seg) > 3:
                seg, _ = two_opt(seg, d)
                tour[s:e + 1] = seg
    return tour


def schedule_cost(tour: Sequence[int], d: Matrix, stop_deadlines: Sequence[Sequence[int]],
                  speed: float, service_h: float, start_h: float = 0.0) -> Tuple[int, float, float]:
    """(late parcels, total hours late, km) for a closed tour - compared lexicographically."""
    clock, late, late_h, km = start_h, 0, 0.0, 0.0
    for a, b in zip(tour, tour[1:]):
        km += d[a][b]
        clock += d[a][b] / speed
        for dl in stop_deadlines[b]:
            if clock > dl:
                late += 1
                late_h += clock - dl
        if stop_deadlines[b]:
            clock += service_h
    return late, round(late_h, 6), km


def lateness_repair(tour: List[int], d: Matrix, stop_deadlines: Sequence[Sequence[int]],
                    speed: float, service_h: float, start_h: float = 0.0,
                    max_passes: int = 8) -> Tuple[List[int], int]:
    """Phase-4 optimisation: Or-opt relocation search on (late, late_hours, km).

    Starting from the distance-optimised tour, try moving every stop to every
    earlier/later position; keep a move only if it lowers the number of late
    parcels (then total lateness, then km).  Never makes the starting tour worse.
    O(k^3) per pass, so it is applied to the final tour only.
    """
    best = list(tour)
    best_cost = schedule_cost(best, d, stop_deadlines, speed, service_h, start_h)
    moves = 0
    for _ in range(max_passes):
        improved = False
        for i in range(1, len(best) - 1):
            node = best[i]
            base = best[:i] + best[i + 1:]
            for j in range(1, len(base)):
                if j == i:
                    continue
                cand = base[:j] + [node] + base[j:]
                c = schedule_cost(cand, d, stop_deadlines, speed, service_h, start_h)
                if c < best_cost:
                    best, best_cost, improved = cand, c, True
                    moves += 1
                    break
        if not improved or best_cost[0] == 0:
            break
    return best, moves


def held_karp(d: Matrix, start: int = 0) -> Tuple[float, List[int]]:
    """Exact TSP by bitmask DP (for small k only)."""
    k = len(d)
    if k == 1:
        return 0.0, [start, start]
    others = [i for i in range(k) if i != start]
    idx = {v: b for b, v in enumerate(others)}
    m = len(others)
    INF = float("inf")
    dp = [[INF] * m for _ in range(1 << m)]
    par = [[-1] * m for _ in range(1 << m)]
    for v in others:
        dp[1 << idx[v]][idx[v]] = d[start][v]
    for mask in range(1, 1 << m):
        for j in range(m):
            cur = dp[mask][j]
            if cur == INF or not (mask >> j & 1):
                continue
            for nb in range(m):
                if mask >> nb & 1:
                    continue
                nm = mask | (1 << nb)
                val = cur + d[others[j]][others[nb]]
                if val < dp[nm][nb]:
                    dp[nm][nb] = val
                    par[nm][nb] = j
    full = (1 << m) - 1
    best, last = INF, -1
    for j in range(m):
        val = dp[full][j] + d[others[j]][start]
        if val < best:
            best, last = val, j
    # rebuild
    order, mask = [], full
    while last != -1:
        order.append(others[last])
        prev = par[mask][last]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return best, [start] + order + [start]
