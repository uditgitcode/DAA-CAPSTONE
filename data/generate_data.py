"""Synthetic dataset generator (reproducible - fixed seeds).

Road network model: random points in a square city; each node is linked to its
3 nearest neighbours, and a spanning tree over all nodes guarantees the graph is
connected. Edge length = straight-line distance x road factor (1.00-1.35).

Usage:  python data/generate_data.py           (writes medium + large)
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from courier.models import Dataset, Location, Parcel  # noqa: E402

AREA_NAMES = ["Sector", "Nagar", "Colony", "Market", "Park", "Enclave", "Vihar", "Plaza",
              "Gardens", "Heights", "Layout", "Chowk"]


def make_dataset(name: str, n_nodes: int, n_parcels: int, capacity: int, seed: int,
                 size_km: float = 20.0, k_near: int = 3, max_weight: int = 25,
                 max_deadline: int = 8) -> Dataset:
    rng = random.Random(seed)
    pts = [(size_km / 2, size_km / 2)] + [(rng.uniform(0, size_km), rng.uniform(0, size_km))
                                          for _ in range(n_nodes - 1)]
    locs = [Location(0, "Depot (Central Hub)", *pts[0])]
    for i in range(1, n_nodes):
        locs.append(Location(i, f"{AREA_NAMES[i % len(AREA_NAMES)]} {i}",
                             round(pts[i][0], 3), round(pts[i][1], 3)))

    def d(a, b):
        return math.dist(pts[a], pts[b])

    edges = {}

    def add(u, v):
        if u == v:
            return
        key = (min(u, v), max(u, v))
        if key not in edges:
            edges[key] = round(d(u, v) * rng.uniform(1.0, 1.35), 2)

    # k nearest neighbours
    for u in range(n_nodes):
        near = sorted((v for v in range(n_nodes) if v != u), key=lambda v: d(u, v))[:k_near]
        for v in near:
            add(u, v)
    # Prim-style spanning tree => connected
    in_tree, rest = {0}, set(range(1, n_nodes))
    best = {v: (d(0, v), 0) for v in rest}
    while rest:
        v = min(rest, key=lambda x: best[x][0])
        add(v, best[v][1])
        rest.remove(v); in_tree.add(v)
        for w in rest:
            if d(v, w) < best[w][0]:
                best[w] = (d(v, w), v)

    parcels = []
    for i in range(n_parcels):
        parcels.append(Parcel(
            parcel_id=f"P{i+1:03d}",
            destination=rng.randint(1, n_nodes - 1),
            weight=rng.randint(1, max_weight),
            priority=rng.choices([1, 2, 3, 4, 5], weights=[2, 3, 3, 2, 1])[0],
            deadline=rng.randint(1, max_deadline),
        ))
    return Dataset(name=name, vehicle_capacity=capacity, locations=locs,
                   edges=[(u, v, w) for (u, v), w in sorted(edges.items())],
                   parcels=parcels, depot=0,
                   meta={"generator_seed": seed, "area_km": size_km})


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, n, p, cap, seed in [("medium_city", 40, 60, 200, 7),
                                  ("large_city", 200, 400, 600, 11)]:
        ds = make_dataset(name, n, p, cap, seed)
        ds.save(os.path.join(here, f"{name}.json"))
        print(f"wrote {name}.json: {n} nodes, {len(ds.edges)} edges, {p} parcels, cap {cap} kg")
