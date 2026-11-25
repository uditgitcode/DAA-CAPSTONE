"""Correctness tests.  Run:  python -m unittest discover -s tests -v"""
import os
import random
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from courier.graph import INF, RoadGraph, floyd_warshall           # noqa: E402
from courier.knapsack import (knapsack_bruteforce_value, knapsack_dp,  # noqa: E402
                              knapsack_greedy, knapsack_value_1d)
from courier.models import Dataset, Parcel                           # noqa: E402
from courier.planner import plan_all_trips, plan_trip               # noqa: E402
from courier.sorting import SORTERS, parcel_key, prioritise         # noqa: E402
from courier.tsp import (held_karp, lateness_repair, nearest_neighbour,  # noqa: E402
                         schedule_cost, tour_length, two_opt)
from data.generate_data import make_dataset                          # noqa: E402

SMALL = os.path.join(ROOT, "data", "small_city.json")


def rand_parcels(n, seed):
    r = random.Random(seed)
    return [Parcel(f"P{i:04d}", r.randint(1, 9), r.randint(1, 20), r.randint(1, 5),
                   r.randint(1, 8)) for i in range(n)]


class TestSorting(unittest.TestCase):
    def test_all_sorters_match_builtin(self):
        for n in (0, 1, 2, 7, 100, 513):
            ps = rand_parcels(n, n)
            expected = sorted(ps, key=parcel_key)
            for name, fn in SORTERS.items():
                with self.subTest(algo=name, n=n):
                    self.assertEqual(fn(ps, key=parcel_key), expected)

    def test_merge_sort_is_stable(self):
        data = [(3, "a"), (1, "b"), (3, "c"), (1, "d"), (2, "e")]
        out = SORTERS["merge"](data, key=lambda t: t[0])
        self.assertEqual(out, [(1, "b"), (1, "d"), (2, "e"), (3, "a"), (3, "c")])

    def test_priority_tiebreak(self):
        a = Parcel("A", 1, 5, priority=2, deadline=3)
        b = Parcel("B", 1, 5, priority=5, deadline=3)
        c = Parcel("C", 1, 5, priority=1, deadline=1)
        self.assertEqual([p.parcel_id for p in prioritise([a, b, c])], ["C", "B", "A"])


class TestKnapsack(unittest.TestCase):
    def test_dp_equals_bruteforce(self):
        r = random.Random(1)
        for trial in range(40):
            n = r.randint(1, 14)
            ps = rand_parcels(n, trial)
            vals = [r.randint(1, 60) for _ in ps]
            cap = r.randint(5, 80)
            res = knapsack_dp(ps, cap, vals)
            self.assertEqual(res.total_value,
                             knapsack_bruteforce_value([p.weight for p in ps], vals, cap))
            self.assertLessEqual(res.total_weight, cap)
            self.assertEqual(sum(vals[ps.index(p)] for p in res.selected), res.total_value)
            self.assertEqual(res.total_value,
                             knapsack_value_1d([p.weight for p in ps], vals, cap))

    def test_dp_never_worse_than_greedy(self):
        for s in range(30):
            ps = rand_parcels(25, s)
            vals = [p.score(8) for p in ps]
            self.assertGreaterEqual(knapsack_dp(ps, 60, vals).total_value,
                                    knapsack_greedy(ps, 60, vals).total_value)

    def test_greedy_counterexample(self):
        # classic case where value/weight greedy is sub-optimal
        ps = [Parcel("A", 1, 10, 1, 1), Parcel("B", 1, 20, 1, 1), Parcel("C", 1, 30, 1, 1)]
        vals = [60, 100, 120]
        self.assertEqual(knapsack_dp(ps, 50, vals).total_value, 220)
        self.assertEqual(knapsack_greedy(ps, 50, vals).total_value, 160)

    def test_zero_capacity_and_oversized(self):
        ps = [Parcel("A", 1, 10, 3, 1)]
        self.assertEqual(knapsack_dp(ps, 0, [5]).selected, [])
        self.assertEqual(knapsack_dp(ps, 9, [5]).total_value, 0)


class TestGraph(unittest.TestCase):
    def test_dijkstra_matches_floyd_warshall(self):
        for seed in (1, 2, 3):
            ds = make_dataset("t", 35, 5, 50, seed)
            g = RoadGraph(len(ds.locations), ds.edges)
            fw = floyd_warshall(g.n, ds.edges)
            for s in range(0, g.n, 5):
                dist, _ = g.dijkstra(s)
                for t in range(g.n):
                    self.assertAlmostEqual(dist[t], fw[s][t], places=6)

    def test_path_reconstruction_length(self):
        ds = Dataset.load(SMALL)
        g = RoadGraph(len(ds.locations), ds.edges)
        w = {(u, v): c for u, v, c in ds.edges}
        w.update({(v, u): c for u, v, c in ds.edges})
        dist, par = g.dijkstra(0)
        for t in range(g.n):
            path = RoadGraph.path_to(par, 0, t)
            self.assertEqual(path[0], 0); self.assertEqual(path[-1], t)
            self.assertAlmostEqual(sum(w[(path[i], path[i+1])] for i in range(len(path)-1)),
                                   dist[t])

    def test_known_small_answers(self):
        ds = Dataset.load(SMALL)
        dist, _ = RoadGraph(len(ds.locations), ds.edges).dijkstra(0)
        self.assertAlmostEqual(dist[5], 7.5)    # 0>4>5
        self.assertAlmostEqual(dist[9], 6.1)    # 0>1>9

    def test_disconnected(self):
        g = RoadGraph(3, [(0, 1, 1.0)])
        self.assertEqual(g.dijkstra(0)[0][2], INF)
        self.assertFalse(g.is_connected())

    def test_negative_edge_rejected(self):
        with self.assertRaises(ValueError):
            RoadGraph(2, [(0, 1, -1)])


class TestTSP(unittest.TestCase):
    def _metric(self, k, seed):
        ds = make_dataset("t", 40, 1, 10, seed)
        g = RoadGraph(len(ds.locations), ds.edges)
        stops = [0] + random.Random(seed).sample(range(1, 40), k - 1)
        return g.stop_distance_matrix(stops)[0]

    def test_tour_is_valid_permutation(self):
        d = self._metric(9, 4)
        for t in (nearest_neighbour(d), two_opt(nearest_neighbour(d), d)[0]):
            self.assertEqual(t[0], 0); self.assertEqual(t[-1], 0)
            self.assertEqual(sorted(t[:-1]), list(range(9)))

    def test_heuristics_vs_optimal(self):
        for seed in range(10):
            d = self._metric(8, seed)
            opt, opt_tour = held_karp(d)
            self.assertAlmostEqual(tour_length(opt_tour, d), opt)
            nn = nearest_neighbour(d)
            imp, _ = two_opt(nn, d)
            self.assertGreaterEqual(tour_length(nn, d) + 1e-9, opt)
            self.assertLessEqual(tour_length(imp, d), tour_length(nn, d) + 1e-9)


    def test_lateness_repair_never_worse(self):
        for seed in range(8):
            d = self._metric(10, seed)
            r = random.Random(seed)
            dls = [[]] + [[r.randint(1, 3)] for _ in range(9)]
            start, _ = two_opt(nearest_neighbour(d), d)
            fixed, _ = lateness_repair(start, d, dls, 30.0, 5 / 60)
            self.assertLessEqual(schedule_cost(fixed, d, dls, 30.0, 5 / 60),
                                 schedule_cost(start, d, dls, 30.0, 5 / 60))
            self.assertEqual(sorted(fixed[:-1]), list(range(10)))


class TestPipeline(unittest.TestCase):
    def test_small_end_to_end(self):
        ds = Dataset.load(SMALL)
        plan = plan_trip(ds, ds.parcels)
        self.assertEqual(plan.packing.total_value, 343)
        self.assertLessEqual(plan.packing.total_weight, ds.vehicle_capacity)
        self.assertAlmostEqual(plan.final_length_km, 28.2, places=6)
        delivered = {p.destination for p in plan.packing.selected}
        self.assertEqual(delivered, set(plan.final_tour_nodes) - {0})
        self.assertEqual(plan.late_parcels, [])

    def test_all_trips_deliver_everything(self):
        ds = Dataset.load(os.path.join(ROOT, "data", "medium_city.json"))
        trips = plan_all_trips(ds)
        ids = [p.parcel_id for t in trips for p in t.packing.selected]
        self.assertEqual(sorted(ids), sorted(p.parcel_id for p in ds.parcels))
        self.assertEqual(len(ids), len(set(ids)))
        for t in trips:
            self.assertLessEqual(t.packing.total_weight, ds.vehicle_capacity)


if __name__ == "__main__":
    unittest.main()
