"""Console formatting of a TripPlan (the 'output trace' used as evidence)."""
from __future__ import annotations

from typing import List

from .graph import RoadGraph
from .models import Dataset
from .planner import TripPlan

LINE = "=" * 78
SUB = "-" * 78


def _table(headers: List[str], rows: List[List[object]]) -> str:
    cols = [str(h) for h in headers]
    data = [[str(c) for c in r] for r in rows]
    widths = [max(len(cols[i]), *(len(r[i]) for r in data)) if data else len(cols[i])
              for i in range(len(cols))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    out = [fmt.format(*cols), fmt.format(*["-" * w for w in widths])]
    out += [fmt.format(*r) for r in data]
    return "\n".join(out)


def dataset_summary(ds: Dataset, graph: RoadGraph) -> str:
    s = [LINE, f" SMART COURIER PLANNER  |  dataset: {ds.name}", LINE,
         f" Locations (graph nodes) : {len(ds.locations)}   (node {ds.depot} = depot)",
         f" Road segments (edges)   : {graph.m}",
         f" Parcels                 : {len(ds.parcels)}  total "
         f"{sum(p.weight for p in ds.parcels)} kg",
         f" Vehicle capacity        : {ds.vehicle_capacity} kg"]
    return "\n".join(s)


def adjacency_preview(ds: Dataset, graph: RoadGraph, limit: int = 12) -> str:
    out = ["", "[PHASE 3] Graph representation (adjacency list, km)"]
    for u in range(min(limit, graph.n)):
        nbrs = ", ".join(f"{v}:{w:g}" for v, w in sorted(graph.adj[u]))
        out.append(f"  {u:>3} {ds.location_name(u)[:18]:<18} -> {nbrs}")
    if graph.n > limit:
        out.append(f"  ... ({graph.n - limit} more nodes)")
    return "\n".join(out)


def sorted_section(ds: Dataset, plan: TripPlan, limit: int = 20) -> str:
    md = max(p.deadline for p in ds.parcels)
    rows = [[i + 1, p.parcel_id, p.deadline, p.priority, p.weight, p.score(md),
             f"{p.destination}:{ds.location_name(p.destination)[:16]}"]
            for i, p in enumerate(plan.sorted_parcels[:limit])]
    t = _table(["#", "Parcel", "Deadline(h)", "Priority", "Kg", "Value", "Destination"], rows)
    more = f"\n  ... ({len(plan.sorted_parcels) - limit} more)" if len(plan.sorted_parcels) > limit else ""
    return (f"\n[PHASE 2] Parcels sorted by Merge Sort  key=(deadline asc, priority desc, kg asc)\n"
            f"{t}{more}")


def dp_table_section(plan: TripPlan, step: int = 5, max_rows: int = 15) -> str:
    dp = plan.packing.table
    if dp is None:
        return ""
    W = plan.packing.capacity
    caps = list(range(0, W + 1, step))
    if caps[-1] != W:
        caps.append(W)
    rows = []
    items = plan.sorted_parcels
    for i in range(min(len(dp), max_rows + 1)):
        label = "0 (none)" if i == 0 else f"{i} {items[i-1].parcel_id} w={items[i-1].weight}"
        rows.append([label] + [dp[i][c] for c in caps])
    t = _table(["i / item"] + [f"c={c}" for c in caps], rows)
    note = f"\n  ... rows {max_rows+1}..{len(dp)-1} omitted" if len(dp) - 1 > max_rows else ""
    return (f"\n[PHASE 2] 0/1 Knapsack DP table dp[i][c] (every {step} kg shown; "
            f"full table {len(dp)}x{W+1})\n{t}{note}\n  Optimal value dp[n][W] = {dp[-1][W]}")


def packing_section(ds: Dataset, plan: TripPlan) -> str:
    k = plan.packing
    md = max(p.deadline for p in ds.parcels)
    rows = [[p.parcel_id, p.weight, p.priority, p.deadline, p.score(md),
             ds.location_name(p.destination)[:18]] for p in k.selected]
    s = [f"\n[PHASE 2] Packing plan - Trip {plan.trip_no}",
         _table(["Parcel", "Kg", "Priority", "Deadline", "Value", "Destination"], rows),
         f"  Loaded {len(k.selected)} parcels | {k.total_weight}/{k.capacity} kg "
         f"({k.utilisation:.1f}% full) | total value {k.total_value}"]
    if k.left_behind:
        s.append("  Left for next trip: " + ", ".join(p.parcel_id for p in k.left_behind[:25])
                 + (" ..." if len(k.left_behind) > 25 else ""))
    return "\n".join(s)


def dijkstra_section(ds: Dataset, plan: TripPlan, graph: RoadGraph) -> str:
    dist, parent = graph.dijkstra(ds.depot)
    rows = []
    for node in plan.stops[1:]:
        path = RoadGraph.path_to(parent, ds.depot, node)
        rows.append([node, ds.location_name(node)[:18], f"{dist[node]:.1f}",
                     " > ".join(map(str, path))])
    s = ["\n[PHASE 3] Dijkstra shortest paths from depot to each stop",
         _table(["Node", "Name", "km", "Shortest road path"], rows)]
    k = len(plan.stops)
    if k <= 12:
        mrows = [[plan.stops[i]] + [f"{x:.1f}" for x in plan.stop_matrix[i]] for i in range(k)]
        s.append("\n  Stop-to-stop distance matrix (km, from Dijkstra)")
        s.append(_table(["from\\to"] + [str(x) for x in plan.stops], mrows))
    return "\n".join(s)


def _seq(nodes, head: int = 12, tail: int = 4) -> str:
    if len(nodes) <= head + tail + 1:
        return " -> ".join(map(str, nodes))
    return (" -> ".join(map(str, nodes[:head])) + f" -> ... ({len(nodes)-head-tail} more) -> "
            + " -> ".join(map(str, nodes[-tail:])))


def route_section(ds: Dataset, plan: TripPlan, max_legs: int = 25) -> str:
    s = ["\n[PHASE 3] Route (Nearest-Neighbour TSP approximation)",
         "  NN order   : " + _seq(plan.nn_tour_nodes),
         f"  NN length  : {plan.nn_length_km:.2f} km"]
    if plan.route_mode == "banded":
        s += ["  [experiment] deadline-banded route (NN + 2-opt inside each band)",
              "  final order: " + _seq(plan.final_tour_nodes),
              f"  final length: {plan.final_length_km:.2f} km"]
    elif plan.route_mode == "deadline":
        s += ["  [PHASE 4 optimisation] 2-opt tour + lateness-repair (Or-opt relocation)",
              "  final order: " + _seq(plan.final_tour_nodes),
              f"  final length: {plan.final_length_km:.2f} km   ({plan.repair_moves} relocations, "
              f"late parcels {plan.late_before_repair} -> {len(plan.late_parcels)})"]
    elif plan.two_opt_moves:
        s += ["  2-opt order: " + _seq(plan.final_tour_nodes),
              f"  2-opt length: {plan.final_length_km:.2f} km "
              f"({plan.two_opt_moves} improving swaps, "
              f"{100*(plan.nn_length_km-plan.final_length_km)/plan.nn_length_km:.1f}% shorter)"]
    else:
        s.append("  2-opt      : no improvement found (NN tour already 2-optimal)")
    rows = []
    legs = plan.legs
    if len(legs) > max_legs:
        legs = legs[:max_legs - 3] + [None] + legs[-2:]
    for leg in legs:
        if leg is None:
            rows.append(["...", "", f"({len(plan.legs) - max_legs + 1} legs hidden)", "", "", "", "", ""])
            continue
        ids = ",".join(p.parcel_id for p in leg.parcels) or "(return)"
        status = ""
        if leg.parcels:
            status = "LATE" if any(leg.arrival_h > p.deadline for p in leg.parcels) else "on time"
        rows.append([leg.order, f"{leg.from_node}->{leg.to_node}",
                     ds.location_name(leg.to_node)[:16], f"{leg.distance_km:.1f}",
                     f"{leg.arrival_h:.2f}", ids[:22], status,
                     ">".join(map(str, leg.road_path))[:24]])
    s.append("\n  Delivery schedule (30 km/h, 5 min per stop)")
    s.append(_table(["Leg", "Stops", "Arrive at", "km", "ETA(h)", "Parcels", "Deadline", "Road path"], rows))
    s.append(f"  TOTAL ROUTE LENGTH: {plan.final_length_km:.2f} km   "
             f"late parcels: {len(plan.late_parcels)}")
    return "\n".join(s)


def timing_section(plan: TripPlan) -> str:
    rows = [[k, f"{v:.3f}"] for k, v in plan.timings_ms.items()]
    total = sum(plan.timings_ms.values())
    return "\n[PHASE 4] Runtime of each step (ms)\n" + _table(["Step", "ms"], rows) + \
        f"\n  total: {total:.3f} ms"
