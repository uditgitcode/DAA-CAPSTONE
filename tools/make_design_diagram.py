"""Draws docs/charts/00_system_design.png - the Phase-1 pipeline design diagram."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "charts", "00_system_design.png")
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#6a4ce2"]

fig, ax = plt.subplots(figsize=(12, 6.2), dpi=130)
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=col, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42,
                                boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, ec=col))
    ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", color="white",
            fontsize=9.5, fontweight="bold")
    ax.text(x + 0.12, y + h - 0.55, body, ha="left", va="top", fontsize=8, color="#191919",
            linespacing=1.35)


def arrow(x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color="#555555", lw=1.5))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.13, label, ha="center", fontsize=7.5,
                color="#555555", style="italic")


box(0.1, 3.3, 2.2, 2.6, "INPUT  (JSON)", "Parcels:\n id, destination,\n weight kg, priority 1-5,\n deadline h\n"
    "Locations: id, name, x, y\nEdges: (u, v, km)\nVehicle capacity W", "#191919")
box(2.75, 3.3, 2.2, 2.6, "1  SORT", "Merge sort  O(n log n)\nstable\nkey = (deadline asc,\n"
    "  priority desc,\n  weight asc, id)\n-> dispatch order", C[0])
box(5.4, 3.3, 2.2, 2.6, "2  PACK", "0/1 Knapsack DP\nO(n*W)\nvalue = 10*priority\n  + urgency(0..20)\n"
    "dp table + traceback\n-> parcels for trip\n-> left-overs", C[1])
box(8.05, 3.3, 2.2, 2.6, "3  SHORTEST PATHS", "Road graph =\n adjacency list\nDijkstra (binary heap)\n"
    "from depot + each stop\nO(k (V+E) log V)\n-> k x k km matrix\n-> road paths", C[2])
box(8.05, 0.2, 2.2, 2.6, "4  ROUTE (TSP)", "Nearest neighbour\n O(k^2) approximation\n+ 2-opt improvement\n"
    "(+ lateness repair,\n optional)\n-> visiting order", C[3])
box(5.4, 0.2, 2.2, 2.6, "5  OUTPUT", "Packing plan\nStop sequence +\n full road path\nETA & deadline check\n"
    "Route length km\nPer-step runtime", C[4])
box(2.75, 0.2, 2.2, 2.6, "LOOP: next trip", "Left-over parcels\nbecome input of the\nnext trip until all\n"
    "parcels delivered\n(--all-trips)", "#666666")
box(0.1, 0.2, 2.2, 2.6, "PHASE 4  BENCH", "benchmark.py times\nevery step on 12 ->\n1600 parcels; compares\n"
    "sorts, DP vs greedy,\nNN vs 2-opt vs exact,\nDijkstra vs Floyd", "#666666")

arrow(2.3, 4.6, 2.75, 4.6, "")
arrow(4.95, 4.6, 5.4, 4.6, "sorted")
arrow(7.6, 4.6, 8.05, 4.6, "stops")
arrow(9.15, 3.3, 9.15, 2.8, "matrix")
arrow(8.05, 1.5, 7.6, 1.5, "tour")
arrow(5.4, 1.5, 4.95, 1.5, "left-overs")
arrow(3.85, 2.8, 3.85, 3.3, "")
ax.text(6, 6.08, "Smart Courier Planner - how the algorithms work together", ha="center",
        fontsize=13, fontweight="bold")
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
