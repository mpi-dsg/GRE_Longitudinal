#!/usr/bin/env python3
"""Per-batch slow-operation trace at 400M keys, with predicted wave positions marked."""
import glob, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Drawn at its printed size (one column) so every label is 10 pt, as the CFP requires.
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})

import os as _os
R = _os.environ.get("WAVE_RESULTS",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results", "raw"))

C = "tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops batch_ns mem".split()
BULK, TOTAL = 100_000_000, 400_000_000
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/waves"
os.makedirs(OUT, exist_ok=True)

def L(f):
    # Earlier driver: 17 columns. Final driver: 23 columns, with batch_ns and mem last.
    out = []
    for l in open(f):
        p = l.strip().split(',')
        if p[0] != "x":
            continue
        if len(p) == 23:
            p = p[:15] + p[21:]
        if len(p) == 17:
            out.append(dict(zip(C, p)))
    return out

def predicted():
    w, x = [], BULK * 8 / 7
    while x < TOTAL:
        w.append(x); x *= 4 / 3
    return w

# ALEX-OL series from the final driver and fixed code; SALI and the B+-tree from the earlier driver.
series = [("r6/b400/base_books_t1_s1866.csv", "ALEX-OL", "#c0392b", "-", "o"),
          ("r13/stag400/stag_books_t1_s1866.csv", "randomized density", "#2980b9", "-", "s"),
          ("scale400_books/sali_books_t1_s1866.csv", "SALI", "#27ae60", "--", "^"),
          ("scale400_books/btreebulk_books_t1_s1866.csv", "B$^+$-tree-OLC", "#7f8c8d", ":", "d")]

fig, ax = plt.subplots(figsize=(3.33, 2.6), layout="constrained")
# Each point is a batch, drawn at its midpoint; a band covers each batch that contains a predicted
# burst position.
BS = 12_500_000
for v in predicted():
    lo = BULK + (v - BULK) // BS * BS
    ax.axvspan(lo / 1e6, (lo + BS) / 1e6, color="0.85", lw=0, zorder=0)

plotted = 0
for key, lab, col, ls, mk in series:
    fs = sorted(glob.glob(f"{R}/{key}"))
    if not fs: continue
    x = L(fs[0])
    if len(x) < 20: continue
    ax.plot([(int(r['keys']) - BS / 2) / 1e6 for r in x], [max(int(r['n100']), 0.5) for r in x],
            label=lab, color=col, ls=ls, marker=mk, ms=3, lw=1.2)
    plotted += 1

ax.set_yscale("log")
ax.set_xlabel("keys in index (millions)")
ax.set_ylabel("inserts $>100\\,\\mu$s\nper batch")
fig.legend(frameon=False, ncol=2, loc="outside upper center", handlelength=1.4,
           columnspacing=0.8, handletextpad=0.4)
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(f"{OUT}/fig_scale400.pdf")
print(f"wrote {OUT}/fig_scale400.pdf with {plotted} series")
