"""Phase 3a - Customer road graph + Dijkstra shortest paths.

Graph format: weighted, undirected adjacency list
    adj[u] = [(v, km), ...]
stored from the dataset edge list [(u, v, km), ...]. Adjacency lists suit road
networks because they are sparse (E ~ 3V), giving Dijkstra O((V+E) log V)
with a binary heap (heapq).
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Sequence, Tuple

INF = math.inf


class RoadGraph:
    def __init__(self, n_nodes: int, edges: Iterable[Tuple[int, int, float]]):
        self.n = n_nodes
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
        self.m = 0
        for u, v, w in edges:
            if w < 0:
                raise ValueError("Dijkstra requires non-negative edge weights")
            self.adj[u].append((v, float(w)))
            self.adj[v].append((u, float(w)))
            self.m += 1

    # ---------------------------------------------------------- Dijkstra
    def dijkstra(self, source: int) -> Tuple[List[float], List[int]]:
        """Single-source shortest paths. Returns (dist, parent)."""
        dist = [INF] * self.n
        parent = [-1] * self.n
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue                      # stale entry (lazy deletion)
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, parent

    @staticmethod
    def path_to(parent: Sequence[int], source: int, target: int) -> List[int]:
        if source == target:
            return [source]
        path, cur = [], target
        while cur != -1:
            path.append(cur)
            if cur == source:
                break
            cur = parent[cur]
        if path[-1] != source:
            return []                         # unreachable
        return path[::-1]

    def stop_distance_matrix(self, stops: Sequence[int]):
        """Run Dijkstra from every stop -> k x k distance matrix + path lookup.

        Cost: k * O((V+E) log V).  This is the 'metric closure' over the stops,
        which the TSP heuristic works on.
        """
        k = len(stops)
        matrix = [[0.0] * k for _ in range(k)]
        parents: Dict[int, List[int]] = {}
        for i, s in enumerate(stops):
            dist, parent = self.dijkstra(s)
            parents[s] = parent
            for j, t in enumerate(stops):
                matrix[i][j] = dist[t]
        return matrix, parents

    def is_connected(self) -> bool:
        dist, _ = self.dijkstra(0)
        return all(d < INF for d in dist)


def floyd_warshall(n: int, edges: Iterable[Tuple[int, int, float]]) -> List[List[float]]:
    """O(V^3) all-pairs reference used only to cross-check Dijkstra in tests/benchmarks."""
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0.0
    for u, v, w in edges:
        if w < d[u][v]:
            d[u][v] = d[v][u] = float(w)
    for k in range(n):
        dk = d[k]
        for i in range(n):
            dik = d[i][k]
            if dik == INF:
                continue
            di = d[i]
            for j in range(n):
                if dik + dk[j] < di[j]:
                    di[j] = dik + dk[j]
    return d
