"""Phase 4 - Performance testing.

Runs five experiments, writes CSVs to results/ and charts to docs/charts/.
    python benchmark.py            (full run, ~1-2 min)
    python benchmark.py --quick    (smaller sizes, for a fast check)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from courier.graph import RoadGraph, floyd_warshall                          # noqa: E402
from courier.knapsack import knapsack_dp, knapsack_greedy, knapsack_value_1d  # noqa: E402
from courier.planner import plan_trip                                        # noqa: E402
from courier.sorting import SORTERS, parcel_key                              # noqa: E402
from courier.tsp import held_karp, nearest_neighbour, tour_length, two_opt   # noqa: E402
from data.generate_data import make_dataset                                  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
CH = os.path.join(ROOT, "docs", "charts")

# validated categorical palette, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False,
        "figure.dpi": 130, "lines.linewidth": 2, "lines.markersize": 6,
    })


def timeit(fn, repeat=5):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def write_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


# ------------------------------------------------------------------ Exp 1
def exp_pipeline(quick):
    sizes = [(10, 12, 50), (25, 50, 150), (50, 100, 250), (100, 200, 400),
             (200, 400, 600), (400, 800, 1000)]
    if not quick:
        sizes.append((800, 1600, 1500))
    steps = ["sorting", "knapsack_dp", "dijkstra", "tsp_nearest_neighbour", "two_opt"]
    rows = []
    for n, p, cap in sizes:
        ds = make_dataset(f"bench_{n}", n, p, cap, seed=n)
        g = RoadGraph(n, ds.edges)
        acc = {s: [] for s in steps}
        reps = 5 if n <= 200 else 3
        for _ in range(reps):
            plan = plan_trip(ds, ds.parcels, g)
            for s in steps:
                acc[s].append(plan.timings_ms[s])
        med = {s: statistics.median(acc[s]) for s in steps}
        rows.append([n, p, cap, len(ds.edges), len(plan.packing.selected), len(plan.stops) - 1,
                     round(plan.nn_length_km, 2), round(plan.final_length_km, 2)]
                    + [round(med[s], 4) for s in steps] + [round(sum(med.values()), 4)])
        print(f"  pipeline n={n:4d} parcels={p:5d} total={sum(med.values()):9.2f} ms")
    header = ["nodes", "parcels", "capacity", "edges", "selected", "stops", "nn_km",
              "final_km"] + [f"{s}_ms" for s in steps] + ["total_ms"]
    write_csv("pipeline_runtime.csv", header, rows)

    # chart 1: runtime per step vs size (log-log)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Merge sort", "Knapsack DP", "Dijkstra (all stops)", "NN-TSP", "2-opt"]
    xs = [r[1] for r in rows]
    for i, s in enumerate(steps):
        ys = [r[8 + i] for r in rows]
        ax.plot(xs, ys, color=C[i], marker=MARKERS[i], label=labels[i])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of parcels (graph nodes = parcels / 2)")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("Runtime of each pipeline step as the dataset grows")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(CH, "01_pipeline_runtime.png")); plt.close(fig)

    # chart 2: share of total runtime (grouped bars, small vs large)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    pick = [rows[0], rows[len(rows) // 2], rows[-1]]
    width = 0.26
    for j, r in enumerate(pick):
        tot = r[-1]
        vals = [100 * r[8 + i] / tot for i in range(len(steps))]
        xpos = [i + (j - 1) * (width + 0.02) for i in range(len(steps))]
        bars = ax.bar(xpos, vals, width, color=C[j], label=f"{r[1]} parcels / {r[0]} nodes")
        for b, v in zip(bars, vals):
            if v >= 8:
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center",
                        fontsize=7.5, color=TEXT)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(labels)
    ax.set_ylabel("Share of pipeline runtime (%)"); ax.grid(axis="x", visible=False)
    ax.set_title("Which step dominates? Share of total runtime")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(CH, "02_runtime_share.png")); plt.close(fig)

    # chart 7: route length NN vs NN+2-opt on the pipeline datasets
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xi = list(range(len(rows)))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in xi], [r[6] for r in rows], w, color=C[0], label="Nearest neighbour")
    b2 = ax.bar([i + w / 2 + 0.01 for i in xi], [r[7] for r in rows], w, color=C[1], label="NN + 2-opt")
    for b, r in zip(b2, rows):
        if r[6] > r[7]:
            ax.text(b.get_x() + b.get_width() / 2, r[7] + 6, f"-{100*(r[6]-r[7])/r[6]:.0f}%",
                    ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(xi); ax.set_xticklabels([f"{r[5]} stops" for r in rows])
    ax.set_ylabel("Trip route length (km)"); ax.grid(axis="x", visible=False)
    ax.set_title("Trip-1 route length: NN vs NN + 2-opt improvement")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(CH, "07_route_length.png")); plt.close(fig)
    return rows


# ------------------------------------------------------------------ Exp 2
def exp_sorting(quick):
    from courier.models import Parcel
    ns = [100, 250, 500, 1000, 2000, 5000] + ([] if quick else [10000, 20000])
    r = random.Random(3)
    rows = []
    for n in ns:
        ps = [Parcel(f"P{i:05d}", 1, r.randint(1, 30), r.randint(1, 5), r.randint(1, 8))
              for i in range(n)]
        row = [n]
        for name in ["merge", "quick", "heap", "insertion"]:
            if name == "insertion" and n > 2000:
                row.append("")
                continue
            row.append(round(timeit(lambda: SORTERS[name](ps, key=parcel_key), 3), 3))
        row.append(round(timeit(lambda: sorted(ps, key=parcel_key), 3), 3))
        rows.append(row)
        print(f"  sorting n={n}: {row[1:]}")
    write_csv("sorting_comparison.csv", ["n", "merge_ms", "quick_ms", "heap_ms",
                                         "insertion_ms", "python_timsort_ms"], rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = ["Merge sort (chosen)", "Quick sort", "Heap sort", "Insertion sort",
             "Python Timsort (reference)"]
    for i, nm in enumerate(names):
        pts = [(rw[0], rw[1 + i]) for rw in rows if rw[1 + i] != ""]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[i], marker=MARKERS[i],
                label=nm, linestyle="--" if i == 4 else "-")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of parcels"); ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("Parcel prioritisation: sorting algorithm comparison")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(os.path.join(CH, "03_sorting_comparison.png")); plt.close(fig)
    return rows


# ------------------------------------------------------------------ Exp 3
def exp_tsp(quick):
    rows = []
    trials = 8 if quick else 15
    for k in range(5, 12):
        ratios_nn, ratios_2o, t_nn, t_2o, t_hk = [], [], [], [], []
        for s in range(trials):
            ds = make_dataset("t", 60, 1, 10, seed=1000 + 17 * k + s)
            g = RoadGraph(60, ds.edges)
            stops = [0] + random.Random(s * 31 + k).sample(range(1, 60), k - 1)
            d, _ = g.stop_distance_matrix(stops)
            t0 = time.perf_counter(); nn = nearest_neighbour(d); t_nn.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); im, _ = two_opt(nn, d); t_2o.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); opt, _ = held_karp(d); t_hk.append(time.perf_counter() - t0)
            ratios_nn.append(tour_length(nn, d) / opt)
            ratios_2o.append(tour_length(im, d) / opt)
        rows.append([k, round(statistics.mean(ratios_nn), 4), round(max(ratios_nn), 4),
                     round(statistics.mean(ratios_2o), 4), round(max(ratios_2o), 4),
                     round(1000 * statistics.mean(t_nn), 4), round(1000 * statistics.mean(t_2o), 4),
                     round(1000 * statistics.mean(t_hk), 3)])
        print(f"  tsp k={k}: NN avg ratio {rows[-1][1]}, 2-opt {rows[-1][3]}, HK {rows[-1][7]} ms")
    write_csv("tsp_quality.csv", ["stops", "nn_ratio_avg", "nn_ratio_worst", "twoopt_ratio_avg",
                                  "twoopt_ratio_worst", "nn_ms", "twoopt_ms", "heldkarp_ms"], rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    ks = [r[0] for r in rows]
    a1.plot(ks, [100 * (r[1] - 1) for r in rows], color=C[0], marker="o", label="Nearest neighbour")
    a1.plot(ks, [100 * (r[3] - 1) for r in rows], color=C[1], marker="s", label="NN + 2-opt")
    a1.set_xlabel("Stops in the trip (incl. depot)"); a1.set_ylabel("% longer than optimal route")
    a1.set_title("Route quality vs exact optimum"); a1.legend(fontsize=8.5)
    a1.set_ylim(bottom=0)
    a2.plot(ks, [r[5] for r in rows], color=C[0], marker="o", label="Nearest neighbour  O(k²)")
    a2.plot(ks, [r[6] for r in rows], color=C[1], marker="s", label="2-opt  O(k²)/pass")
    a2.plot(ks, [r[7] for r in rows], color=C[2], marker="^", label="Held-Karp exact  O(k²·2ᵏ)")
    a2.set_yscale("log"); a2.set_xlabel("Stops in the trip (incl. depot)")
    a2.set_ylabel("Runtime (ms, log scale)"); a2.set_title("Why approximate? Runtime")
    a2.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(CH, "04_tsp_quality.png")); plt.close(fig)
    return rows


# ------------------------------------------------------------------ Exp 4
def exp_knapsack(quick):
    from courier.models import Parcel
    rows = []
    r = random.Random(9)
    configs = [(20, 100), (50, 200), (100, 400), (200, 600), (400, 1000)] + \
        ([] if quick else [(800, 1500), (1600, 2000)])
    for n, W in configs:
        ps = [Parcel(f"P{i}", 1, r.randint(1, 25), r.randint(1, 5), r.randint(1, 8)) for i in range(n)]
        vals = [p.score(8) for p in ps]
        t2d = timeit(lambda: knapsack_dp(ps, W, vals), 3)
        t1d = timeit(lambda: knapsack_value_1d([p.weight for p in ps], vals, W), 3)
        tg = timeit(lambda: knapsack_greedy(ps, W, vals), 3)
        dp_v = knapsack_dp(ps, W, vals).total_value
        g_v = knapsack_greedy(ps, W, vals).total_value
        rows.append([n, W, n * W, round(t2d, 3), round(t1d, 3), round(tg, 4), dp_v, g_v,
                     round(100 * (dp_v - g_v) / dp_v, 3)])
        print(f"  knapsack n={n} W={W}: 2D {t2d:.1f} ms 1D {t1d:.1f} ms greedy gap {rows[-1][-1]}%")
    write_csv("knapsack_comparison.csv", ["n", "capacity", "n_times_W", "dp2d_ms", "dp1d_ms",
                                          "greedy_ms", "dp_value", "greedy_value", "greedy_gap_pct"], rows)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = [rw[2] for rw in rows]
    ax.plot(x, [rw[3] for rw in rows], color=C[0], marker="o", label="DP full table (with traceback)")
    ax.plot(x, [rw[4] for rw in rows], color=C[1], marker="s", label="DP 1-D array (value only)")
    ax.plot(x, [rw[5] for rw in rows], color=C[2], marker="^", label="Greedy value/weight")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Table size n × W (cells)"); ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("0/1 Knapsack: DP grows linearly with n×W")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(os.path.join(CH, "05_knapsack_runtime.png")); plt.close(fig)
    return rows


# ------------------------------------------------------------------ Exp 5
def exp_shortest_paths(quick):
    rows = []
    for n in [25, 50, 100, 200] + ([] if quick else [300]):
        ds = make_dataset("sp", n, 1, 10, seed=n + 5)
        g = RoadGraph(n, ds.edges)
        k = max(5, n // 5)
        stops = list(range(0, n, n // k))[:k]
        t_dk = timeit(lambda: g.stop_distance_matrix(stops), 3)
        t_all = timeit(lambda: [g.dijkstra(s) for s in range(n)], 3)
        t_fw = timeit(lambda: floyd_warshall(n, ds.edges), 1 if n >= 200 else 3)
        rows.append([n, len(ds.edges), k, round(t_dk, 3), round(t_all, 3), round(t_fw, 3)])
        print(f"  shortest paths n={n}: dijkstra(k stops) {t_dk:.2f} all {t_all:.2f} FW {t_fw:.2f} ms")
    write_csv("shortest_path_comparison.csv", ["nodes", "edges", "stops", "dijkstra_stops_ms",
                                               "dijkstra_all_nodes_ms", "floyd_warshall_ms"], rows)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = [rw[0] for rw in rows]
    ax.plot(xs, [rw[3] for rw in rows], color=C[0], marker="o", label="Dijkstra from trip stops only (used)")
    ax.plot(xs, [rw[4] for rw in rows], color=C[1], marker="s", label="Dijkstra from every node")
    ax.plot(xs, [rw[5] for rw in rows], color=C[2], marker="^", label="Floyd-Warshall O(V³)")
    ax.set_yscale("log"); ax.set_xlabel("Graph nodes"); ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("Shortest-path strategy comparison")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(os.path.join(CH, "06_shortest_paths.png")); plt.close(fig)
    return rows


# ------------------------------------------------------------------ Exp 6
def exp_deadline(quick):
    rows = []
    sizes = [(25, 50, 150), (40, 60, 200), (50, 100, 250), (100, 200, 400), (200, 400, 600)]
    seeds = [1, 2, 3] if quick else [1, 2, 3, 4, 5]
    for n, p, cap in sizes:
        acc = {m: [0, 0.0, 0.0] for m in ("distance", "deadline", "banded")}
        for sd in seeds:
            ds = make_dataset("dl", n, p, cap, seed=100 * n + sd)
            g = RoadGraph(n, ds.edges)
            for m in acc:
                t0 = time.perf_counter()
                plan = plan_trip(ds, ds.parcels, g, route_mode=m)
                acc[m][2] += (time.perf_counter() - t0) * 1000
                acc[m][0] += len(plan.late_parcels)
                acc[m][1] += plan.final_length_km
        k = len(seeds)
        row = [n, p]
        for m in ("distance", "deadline", "banded"):
            row += [round(acc[m][0] / k, 2), round(acc[m][1] / k, 1), round(acc[m][2] / k, 2)]
        rows.append(row)
        print(f"  deadline n={n}: late dist/repair/banded = {row[2]}/{row[5]}/{row[8]}  "
              f"km = {row[3]}/{row[6]}/{row[9]}")
    write_csv("deadline_optimisation.csv",
              ["nodes", "parcels", "late_2opt", "km_2opt", "ms_2opt", "late_repair", "km_repair",
               "ms_repair", "late_banded", "km_banded", "ms_banded"], rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    xi = list(range(len(rows)))
    w = 0.26
    names = ["NN + 2-opt (distance only)", "+ lateness repair (Phase-4 fix)", "Deadline bands (experiment)"]
    for j in range(3):
        xs = [i + (j - 1) * (w + 0.02) for i in xi]
        a1.bar(xs, [r[2 + 3 * j] for r in rows], w, color=C[j], label=names[j])
        a2.bar(xs, [r[3 + 3 * j] for r in rows], w, color=C[j], label=names[j])
    for a, t, yl in ((a1, "Late parcels on trip 1 (avg of seeds)", "Late parcels"),
                     (a2, "Route length on trip 1", "km")):
        a.set_xticks(xi); a.set_xticklabels([f"{r[1]} parcels" for r in rows], fontsize=8.5)
        a.set_title(t); a.set_ylabel(yl); a.grid(axis="x", visible=False)
    a1.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(CH, "08_deadline_optimisation.png")); plt.close(fig)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True); os.makedirs(CH, exist_ok=True)
    style()
    print("[1/6] Pipeline runtime by dataset size"); exp_pipeline(a.quick)
    print("[2/6] Sorting algorithms"); exp_sorting(a.quick)
    print("[3/6] TSP approximation quality"); exp_tsp(a.quick)
    print("[4/6] Knapsack DP vs greedy"); exp_knapsack(a.quick)
    print("[5/6] Shortest-path strategies"); exp_shortest_paths(a.quick)
    print("[6/6] Deadline-aware optimisation"); exp_deadline(a.quick)
    print(f"CSV -> {RES}\nCharts -> {CH}")
