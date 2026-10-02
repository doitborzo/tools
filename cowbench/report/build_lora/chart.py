"""chart_exact.png: exact-match error of every configuration, with and without
the cows annotated ruminating. Numbers from data.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
ALL, NR = "#2a78d6", "#eb6834"          # categorical slots 1 and 2, validated
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
rows = D["configs"]
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
h = 0.36
for i, c in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y + h / 2 + 0.01, c["exact"] * 100, height=h, color=ALL, zorder=2)
    ax.text(c["exact"] * 100 + 0.6, y + h / 2 + 0.01, f"{c['exact'] * 100:.1f}".replace(".", ","),
            va="center", fontsize=7.5, color=INK2)
    if c["nr_exact"] is not None:
        ax.barh(y - h / 2 - 0.01, c["nr_exact"] * 100, height=h, color=NR, zorder=2)
        ax.text(c["nr_exact"] * 100 + 0.6, y - h / 2 - 0.01,
                ("≈" if c.get("nr_note") else "") + f"{c['nr_exact'] * 100:.1f}".replace(".", ","),
                va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([c["short"] for c in reversed(rows)])
ax.set_xlim(0, 52)
ax.set_xlabel("Помилка повного збігу (поза і активність), %")
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=ALL, label="усі 2532 корови"),
                   Patch(color=NR, label="без корів, розмічених як «жуйка» (2149)")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "chart_exact.png"), facecolor="white")
print("chart_exact.png")
