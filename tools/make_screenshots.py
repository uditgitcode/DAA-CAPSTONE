"""Produce evidence for the report:
  * docs/output_traces/*.txt   - full console output of each demo run
  * docs/screenshots/*.png     - terminal-style screenshots of those runs
  * docs/charts/route_map_*.png - map of the road graph with the planned route
Run from the project root:  python tools/make_screenshots.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from courier.graph import RoadGraph     # noqa: E402
from courier.models import Dataset      # noqa: E402
from courier.planner import plan_all_trips, plan_trip  # noqa: E402

TR = os.path.join(ROOT, "docs", "output_traces")
SS = os.path.join(ROOT, "docs", "screenshots")
CH = os.path.join(ROOT, "docs", "charts")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]


def run(cmd: list[str]) -> str:
    out = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    return out.stdout + out.stderr


def terminal_png(lines: list[str], title: str, path: str, prompt: str | None = None):
    font = ImageFont.truetype(FONT, 15)
    bold = ImageFont.truetype(FONT_B, 15)
    cw, lh = font.getbbox("M")[2], 20
    if prompt:
        lines = [prompt] + lines
    width = max(80, max((len(l) for l in lines), default=80)) * cw + 40
    height = len(lines) * lh + 70
    img = Image.new("RGB", (width, height), "#1e1f24")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 34], fill="#2d2f36")
    for i, col in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.ellipse([14 + i * 22, 11, 26 + i * 22, 23], fill=col)
    d.text((width // 2, 17), title, font=font, fill="#c8c8c8", anchor="mm")
    y = 48
    for i, line in enumerate(lines):
        colour = "#e6e6e6"
        f = font
        if prompt and i == 0:
            colour, f = "#7ee787", bold
        elif line.startswith("[PHASE"):
            colour, f = "#79c0ff", bold
        elif line.startswith("=") or line.startswith(" TRIP") or line.startswith(" SMART") \
                or line.startswith(" SUMMARY"):
            colour = "#f2a600"
        elif "TOTAL ROUTE" in line or "Optimal value" in line or line.startswith("  Loaded") \
                or line.strip().startswith("OK") or "ok" == line.strip()[-2:]:
            colour, f = "#7ee787", bold
        elif "LATE" in line:
            colour = "#ff7b72"
        d.text((20, y), line, font=f, fill=colour)
        y += lh
    img.save(path)


def section(text: str, start: str, end: str | None) -> list[str]:
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = len(lines) if end is None else next(k for k, l in enumerate(lines[i + 1:], i + 1)
                                            if l.startswith(end))
    while j > i and not lines[j - 1].strip():
        j -= 1
    return lines[i:j]


def route_map(ds: Dataset, trips, path: str, title: str, label_nodes: bool, ax=None, colour_offset=0):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 7.2), dpi=130)
    pos = {l.node_id: (l.x, l.y) for l in ds.locations}
    for u, v, w in ds.edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#d0d0d0", lw=1, zorder=1)
        if label_nodes:
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.text(mx, my, f"{w:g}", fontsize=7, color="#888888", ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.5), zorder=2)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=22 if not label_nodes else 60, color="#9a9a9a", zorder=3)
    for t_i, plan in enumerate(trips):
        col = C[(t_i + colour_offset) % len(C)]
        first = True
        for leg in plan.legs:
            p = leg.road_path
            for a, b in zip(p, p[1:]):
                ax.annotate("", xy=pos[b], xytext=pos[a], zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                            shrinkA=5, shrinkB=5, mutation_scale=12))
            if first:
                ax.plot([], [], color=col, lw=2,
                        label=f"Trip {plan.trip_no}: {len(plan.stops)-1} stops, "
                              f"{plan.final_length_km:.1f} km")
                first = False
        stops = plan.stops[1:]
        ax.scatter([pos[s][0] for s in stops], [pos[s][1] for s in stops], s=70 if label_nodes else 30,
                   color=col, edgecolor="white", linewidth=1.5, zorder=5)
    if label_nodes:
        order = {n: i for i, n in enumerate(trips[0].final_tour_nodes[1:-1], 1)}
        for loc in ds.locations:
            tag = f"{loc.node_id} {loc.name}" + (f"  (stop #{order[loc.node_id]})"
                                                  if loc.node_id in order else "")
            ax.text(loc.x, loc.y + 0.32, tag, fontsize=8, ha="center", color="#191919", zorder=6)
    d = ds.locations[ds.depot]
    ax.scatter([d.x], [d.y], marker="s", s=150, color="#191919", zorder=7, label="Depot")
    ax.set_title(title, fontweight="bold", fontsize=11 if own else 10)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_aspect("equal"); ax.grid(color="#eeeeee")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 if own else -0.12), ncol=2,
              frameon=False, fontsize=8.5)
    if own:
        fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    for p in (TR, SS, CH):
        os.makedirs(p, exist_ok=True)

    # ---------------- runs
    small = run(["main.py", "data/small_city.json", "--adjacency", "--dp-table"])
    medium = run(["main.py", "data/medium_city.json", "--all-trips"])
    large = run(["main.py", "data/large_city.json"])
    dl = run(["main.py", "data/medium_city.json", "--deadline-aware"])
    tests = run(["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"])
    for name, txt in [("small_city_run", small), ("medium_city_all_trips", medium),
                      ("large_city_run", large), ("medium_city_deadline_aware", dl), ("unit_tests", tests)]:
        with open(os.path.join(TR, f"{name}.txt"), "w") as fh:
            fh.write(txt)

    p_small = "$ python main.py data/small_city.json --adjacency --dp-table"
    terminal_png(section(small, "=" * 78, " TRIP 1")[:-1] + [""] +
                 section(small, "[PHASE 2] Parcels sorted", "[PHASE 2] 0/1"),
                 "Phase 2/3 - dataset, graph & sorted parcels",
                 os.path.join(SS, "01_small_input_graph_sorting.png"), p_small)
    terminal_png(section(small, "[PHASE 2] 0/1", "[PHASE 3] Dijkstra"),
                 "Phase 2 - knapsack DP table & packing plan",
                 os.path.join(SS, "02_small_dp_table_packing.png"))
    terminal_png(section(small, "[PHASE 3] Dijkstra", "[PHASE 3] Route"),
                 "Phase 3 - Dijkstra shortest paths",
                 os.path.join(SS, "03_small_dijkstra.png"))
    terminal_png(section(small, "[PHASE 3] Route", None),
                 "Phase 3/4 - route, schedule & runtimes",
                 os.path.join(SS, "04_small_route_timing.png"))
    ml = medium.splitlines()
    k = next(i for i, l in enumerate(ml) if l.startswith(" SUMMARY"))
    terminal_png(section(medium, "=" * 78, " TRIP 1")[:-1] + ["   ... (per-trip details in output trace) ..."]
                 + ml[k - 1:],
                 "Medium dataset - multi-trip planning",
                 os.path.join(SS, "05_medium_all_trips.png"),
                 "$ python main.py data/medium_city.json --all-trips")
    lg = large.splitlines()
    rt = section(large, "[PHASE 3] Route", None)
    terminal_png(section(large, "=" * 78, " TRIP 1")[:-1] + [""] +
                 section(large, "[PHASE 2] Packing", "[PHASE 3] Dijkstra")[:6] + ["  ..."] +
                 rt[:4] + ["  ..."] + rt[-12:],
                 "Large dataset - 200 nodes / 400 parcels",
                 os.path.join(SS, "06_large_run.png"), "$ python main.py data/large_city.json")
    rd = section(dl, "[PHASE 3] Route", None)
    terminal_png(rd[:8] + ["  ..."] + rd[-14:], "Phase 4 optimisation - lateness repair",
                 os.path.join(SS, "09_deadline_aware_optimisation.png"),
                 "$ python main.py data/medium_city.json --deadline-aware")
    tl = tests.splitlines()
    terminal_png(tl, "Correctness - unit tests", os.path.join(SS, "07_unit_tests.png"),
                 "$ python -m unittest discover -s tests -t . -v")
    blog = os.path.join(ROOT, "results", "benchmark_log.txt")
    if os.path.exists(blog):
        terminal_png(open(blog).read().splitlines(), "Phase 4 - benchmark run",
                     os.path.join(SS, "08_benchmark_run.png"), "$ python benchmark.py")

    # ---------------- route maps
    ds = Dataset.load(os.path.join(ROOT, "data", "small_city.json"))
    plan = plan_trip(ds, ds.parcels)
    route_map(ds, [plan], os.path.join(CH, "route_map_small.png"),
              "small_city - Trip 1 route (NN + 2-opt) on the road graph", True)
    nn_plan = plan_trip(ds, ds.parcels, use_two_opt=False)
    route_map(ds, [nn_plan], os.path.join(CH, "route_map_small_nn_only.png"),
              "small_city - Trip 1 route, nearest neighbour only", True)
    dm = Dataset.load(os.path.join(ROOT, "data", "medium_city.json"))
    trips = plan_all_trips(dm)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=120)
    for i, (t, ax) in enumerate(zip(trips, axes.flat)):
        route_map(dm, [t], "", f"Trip {t.trip_no}", False, ax=ax, colour_offset=i)
    fig.suptitle("medium_city - one panel per van trip (DP packing + NN/2-opt route)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(CH, "route_map_medium_all_trips.png"))
    plt.close(fig)
    print("traces, screenshots and route maps written")


if __name__ == "__main__":
    main()
