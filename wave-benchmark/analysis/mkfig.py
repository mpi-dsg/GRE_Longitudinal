#!/usr/bin/env python3
"""Per-batch slow-operation trace at 400M keys, with predicted wave positions marked."""
import glob, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
R = _os.environ.get("WAVE_RESULTS",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "results", "raw"))

C = "tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops batch_ns mem".split()
BULK, TOTAL = 100_000_000, 400_000_000
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/waves"
os.makedirs(OUT, exist_ok=True)

def L(f):
    return [dict(zip(C, l.strip().split(','))) for l in open(f) if l.count(',') == 16]

def predicted():
    w, x = [], BULK * 8 / 7
    while x < TOTAL:
        w.append(x); x *= 4 / 3
    return w

series = [("alexol", "ALEX-OL", "#c0392b", "-", "o"),
          ("alexolstag", "ALEX-OL, randomized density", "#2980b9", "-", "s"),
          ("sali", "SALI", "#27ae60", "--", "^"),
          ("btreebulk", "B$^+$-tree-OLC", "#7f8c8d", ":", "d")]

fig, ax = plt.subplots(figsize=(7.0, 2.7))
for v in predicted():
    ax.axvline(v / 1e6, color="0.85", lw=6, zorder=0)
ax.text(predicted()[0] / 1e6, 1.6e4, " predicted waves", fontsize=7, color="0.45", va="top")

plotted = 0
for key, lab, col, ls, mk in series:
    fs = sorted(glob.glob(f"{R}/scale400_books/{key}_books_t1_s*.csv"))
    if not fs: continue
    x = L(fs[0])
    if len(x) < 20: continue
    ax.plot([int(r['keys']) / 1e6 for r in x], [max(int(r['n100']), 0.5) for r in x],
            label=lab, color=col, ls=ls, marker=mk, ms=3, lw=1.2)
    plotted += 1

ax.set_yscale("log")
ax.set_xlabel("keys in index (millions)")
ax.set_ylabel("inserts $>100\\,\\mu$s\nper batch")
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_scale400.pdf", bbox_inches="tight")
print(f"wrote {OUT}/fig_scale400.pdf with {plotted} series")
