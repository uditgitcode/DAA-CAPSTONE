"""Data model for the Smart Courier Planner.

A Parcel is something to deliver; a Location is a node in the road graph.
Node 0 is always the depot (where the van starts and ends its trip).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


@dataclass
class Parcel:
    parcel_id: str          # e.g. "P001"
    destination: int        # node id of the customer in the road graph
    weight: int             # kg (integer, so the DP table has integer columns)
    priority: int           # 1 (low) .. 5 (critical)
    deadline: int           # hours from the start of the shift (1 = most urgent)
    value: int = 0          # delivery score used by the knapsack (computed if 0)

    def score(self, max_deadline: int) -> int:
        """Delivery value used by knapsack.

        Higher priority and an earlier deadline both make a parcel more valuable
        to load on *this* trip.  Integer so the DP stays exact.
        value = 10*priority + urgency_bonus, urgency_bonus in [0, 20].
        """
        if self.value:
            return self.value
        urgency = (max_deadline - self.deadline) / max(1, max_deadline - 1)
        return 10 * self.priority + round(20 * urgency)


@dataclass
class Location:
    node_id: int
    name: str
    x: float
    y: float


@dataclass
class Dataset:
    name: str
    vehicle_capacity: int                      # kg
    locations: List[Location]
    edges: List[Tuple[int, int, float]]        # undirected (u, v, km)
    parcels: List[Parcel]
    depot: int = 0
    meta: Dict = field(default_factory=dict)

    # ---------- I/O ----------
    @staticmethod
    def load(path: str) -> "Dataset":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return Dataset(
            name=raw["name"],
            vehicle_capacity=raw["vehicle_capacity"],
            locations=[Location(**loc) for loc in raw["locations"]],
            edges=[tuple(e) for e in raw["edges"]],
            parcels=[Parcel(**p) for p in raw["parcels"]],
            depot=raw.get("depot", 0),
            meta=raw.get("meta", {}),
        )

    def save(self, path: str) -> None:
        data = {
            "name": self.name,
            "vehicle_capacity": self.vehicle_capacity,
            "depot": self.depot,
            "meta": self.meta,
            "locations": [asdict(l) for l in self.locations],
            "edges": [list(e) for e in self.edges],
            "parcels": [asdict(p) for p in self.parcels],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

    def location_name(self, node: int) -> str:
        return self.locations[node].name
