"""Smart Courier Planner - command-line entry point.

Examples
  python main.py data/small_city.json
  python main.py data/small_city.json --dp-table --adjacency
  python main.py data/medium_city.json --all-trips
  python main.py data/large_city.json --no-2opt
  python main.py data/medium_city.json --all-trips --deadline-aware
"""
from __future__ import annotations

import argparse
import sys

from courier import report
from courier.graph import RoadGraph
from courier.models import Dataset
from courier.planner import plan_all_trips, plan_trip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Courier route + packing planner")
    ap.add_argument("dataset", help="path to dataset JSON")
    ap.add_argument("--all-trips", action="store_true",
                    help="keep planning trips until every parcel is delivered")
    ap.add_argument("--no-2opt", action="store_true", help="use plain nearest-neighbour route")
    ap.add_argument("--deadline-aware", action="store_true",
                    help="Phase-4 optimisation: lateness-repair local search after 2-opt")
    ap.add_argument("--banded", action="store_true",
                    help="experiment: serve stops strictly in deadline bands")
    ap.add_argument("--dp-table", action="store_true", help="print the knapsack DP table")
    ap.add_argument("--adjacency", action="store_true", help="print graph adjacency list")
    ap.add_argument("--sort", default="merge", choices=["merge", "quick", "heap", "insertion"])
    args = ap.parse_args(argv)

    ds = Dataset.load(args.dataset)
    graph = RoadGraph(len(ds.locations), ds.edges)
    if not graph.is_connected():
        print("ERROR: road graph is not connected", file=sys.stderr)
        return 1

    print(report.dataset_summary(ds, graph))
    if args.adjacency:
        print(report.adjacency_preview(ds, graph))

    mode = "deadline" if args.deadline_aware else ("banded" if args.banded else "distance")
    if args.all_trips:
        trips = plan_all_trips(ds, use_two_opt=not args.no_2opt, route_mode=mode)
    else:
        trips = [plan_trip(ds, ds.parcels, graph, not args.no_2opt, args.sort, route_mode=mode)]

    for plan in trips:
        print("\n" + report.LINE + f"\n TRIP {plan.trip_no}\n" + report.LINE)
        if plan.trip_no == 1:
            print(report.sorted_section(ds, plan))
            if args.dp_table:
                print(report.dp_table_section(plan))
        print(report.packing_section(ds, plan))
        print(report.dijkstra_section(ds, plan, graph))
        print(report.route_section(ds, plan))
        print(report.timing_section(plan))

    if len(trips) > 1:
        print("\n" + report.LINE + "\n SUMMARY OF ALL TRIPS\n" + report.LINE)
        rows = [[t.trip_no, len(t.packing.selected), t.packing.total_weight,
                 f"{t.packing.utilisation:.0f}%", len(t.stops) - 1,
                 f"{t.final_length_km:.1f}", len(t.late_parcels)] for t in trips]
        print(report._table(["Trip", "Parcels", "Kg", "Fill", "Stops", "km", "Late"], rows))
        print(f"  Total distance: {sum(t.final_length_km for t in trips):.1f} km, "
              f"parcels delivered: {sum(len(t.packing.selected) for t in trips)}/{len(ds.parcels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
