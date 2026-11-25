"""Integration layer: sorting -> knapsack -> Dijkstra -> TSP, with per-step timing.

Pipeline for ONE trip
  1. prioritise()          merge sort by (deadline, -priority, weight, id)
  2. knapsack_dp()         pick the best-value subset that fits the van
  3. stop_distance_matrix  Dijkstra from depot + every distinct customer stop
  4. nearest_neighbour()   order the stops (TSP approximation)
  5. two_opt()             optional local-search improvement (Phase 4)
  6. expand legs           turn stop order into the real road path + ETAs
plan_all_trips() repeats this on the left-over parcels until all are delivered.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from .graph import RoadGraph
from .knapsack import KnapsackResult, knapsack_dp
from .models import Dataset, Parcel
from .sorting import prioritise
from .tsp import (deadline_banded_route, lateness_repair, nearest_neighbour,
                  schedule_cost, tour_length, two_opt)

AVG_SPEED_KMPH = 30.0      # city van speed
SERVICE_MIN = 5.0          # minutes spent at each stop


@dataclass
class Leg:
    order: int
    from_node: int
    to_node: int
    distance_km: float
    road_path: List[int]
    arrival_h: float
    parcels: List[Parcel]


@dataclass
class TripPlan:
    trip_no: int
    sorted_parcels: List[Parcel]
    packing: KnapsackResult
    stops: List[int]                 # depot first, then distinct destinations
    stop_matrix: List[List[float]]
    nn_tour_nodes: List[int]
    nn_length_km: float
    final_tour_nodes: List[int]
    final_length_km: float
    two_opt_moves: int
    legs: List[Leg]
    route_mode: str = "distance"
    repair_moves: int = 0
    late_before_repair: int = -1
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def late_parcels(self) -> List[Parcel]:
        late = []
        for leg in self.legs:
            for p in leg.parcels:
                if leg.arrival_h > p.deadline:
                    late.append(p)
        return late


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def plan_trip(ds: Dataset, parcels: List[Parcel], graph: RoadGraph | None = None,
              use_two_opt: bool = True, sort_algo: str = "merge", trip_no: int = 1,
              start_time_h: float = 0.0, route_mode: str = "distance") -> TripPlan:
    timings: Dict[str, float] = {}

    # 1. sort -------------------------------------------------------------
    t0 = time.perf_counter()
    ordered = prioritise(parcels, sort_algo)
    timings["sorting"] = _ms(t0)

    # 2. knapsack --------------------------------------------------------
    t0 = time.perf_counter()
    max_deadline = max((p.deadline for p in ds.parcels), default=1)
    values = [p.score(max_deadline) for p in ordered]
    packing = knapsack_dp(ordered, ds.vehicle_capacity, values)
    # keep the dispatch (sorted) order for the selected list
    timings["knapsack_dp"] = _ms(t0)

    # 3. graph + dijkstra ------------------------------------------------
    if graph is None:
        t0 = time.perf_counter()
        graph = RoadGraph(len(ds.locations), ds.edges)
        timings["graph_build"] = _ms(t0)

    stops = [ds.depot]
    for p in packing.selected:
        if p.destination not in stops:
            stops.append(p.destination)
    t0 = time.perf_counter()
    matrix, parents = graph.stop_distance_matrix(stops)
    timings["dijkstra"] = _ms(t0)

    # 4. nearest neighbour -----------------------------------------------
    t0 = time.perf_counter()
    nn = nearest_neighbour(matrix, 0)
    timings["tsp_nearest_neighbour"] = _ms(t0)
    nn_len = tour_length(nn, matrix)

    # 5. 2-opt ------------------------------------------------------------
    moves = 0
    final = nn
    if use_two_opt and len(stops) > 3:
        t0 = time.perf_counter()
        final, moves = two_opt(nn, matrix)
        timings["two_opt"] = _ms(t0)
    else:
        timings["two_opt"] = 0.0
    stop_dls = [[p.deadline for p in packing.selected if p.destination == s] if i else []
                for i, s in enumerate(stops)]
    repair_moves, late_before = 0, -1
    if route_mode == "banded":
        # experiment kept for the report: serve deadline bands in order (worse - see REPORT)
        stop_dl = [0] + [min(x) for x in stop_dls[1:]]
        t0 = time.perf_counter()
        final = deadline_banded_route(matrix, stop_dl, 0, improve=use_two_opt)
        timings["deadline_route"] = _ms(t0)
    elif route_mode == "deadline":
        # Phase-4 optimisation: lateness-repair relocation search on the 2-opt tour
        late_before = schedule_cost(final, matrix, stop_dls, AVG_SPEED_KMPH,
                                    SERVICE_MIN / 60, start_time_h)[0]
        t0 = time.perf_counter()
        final, repair_moves = lateness_repair(final, matrix, stop_dls, AVG_SPEED_KMPH,
                                              SERVICE_MIN / 60, start_time_h)
        timings["lateness_repair"] = _ms(t0)
    final_len = tour_length(final, matrix)

    # 6. expand legs ------------------------------------------------------
    by_node: Dict[int, List[Parcel]] = {}
    for p in packing.selected:
        by_node.setdefault(p.destination, []).append(p)
    legs, clock = [], start_time_h
    for k in range(len(final) - 1):
        a, b = stops[final[k]], stops[final[k + 1]]
        km = matrix[final[k]][final[k + 1]]
        clock += km / AVG_SPEED_KMPH
        legs.append(Leg(k + 1, a, b, km, RoadGraph.path_to(parents[a], a, b),
                        round(clock, 3), by_node.get(b, []) if b != ds.depot else []))
        if b != ds.depot:
            clock += SERVICE_MIN / 60.0

    return TripPlan(
        trip_no=trip_no, sorted_parcels=ordered, packing=packing, stops=stops,
        stop_matrix=matrix,
        nn_tour_nodes=[stops[i] for i in nn], nn_length_km=nn_len,
        final_tour_nodes=[stops[i] for i in final], final_length_km=final_len,
        two_opt_moves=moves, legs=legs, timings_ms=timings, route_mode=route_mode,
        repair_moves=repair_moves, late_before_repair=late_before,
    )


def plan_all_trips(ds: Dataset, use_two_opt: bool = True, max_trips: int = 50,
                   route_mode: str = "distance") -> List[TripPlan]:
    graph = RoadGraph(len(ds.locations), ds.edges)
    remaining = [p for p in ds.parcels if p.weight <= ds.vehicle_capacity]
    trips, clock = [], 0.0
    while remaining and len(trips) < max_trips:
        plan = plan_trip(ds, remaining, graph, use_two_opt, trip_no=len(trips) + 1,
                         start_time_h=clock, route_mode=route_mode)
        if not plan.packing.selected:
            break
        trips.append(plan)
        clock = plan.legs[-1].arrival_h + 0.25 if plan.legs else clock   # 15 min reload
        remaining = plan.packing.left_behind
    return trips
