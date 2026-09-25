#!/usr/bin/env python3
"""Slow inserts against thread count at 4M keys, one line per arm (paper Figure: fig_mech4m).

Same data and rule as the 4M design table in mk_mech.py: results/raw/r6/m4, median of the
nine runs (three seeds x three repetitions) per cell, complete runs of 30 batches only.
Drawn at its printed size (one column) so every label is 10 pt.
"""
import glob, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/waves"
os.makedirs(OUT, exist_ok=True)
THREADS = [1, 2, 4, 8, 16]


def slow(f):
    xs = [l.split(",") for l in open(f) if l.startswith("x,") and l.count(",") == 22]
    return sum(int(r[12]) for r in xs) if len(xs) == 30 else None


def cell(a, T):
    v = [s for s in (slow(f) for f in sorted(glob.glob(f"{R}/r6/m4/{a}_t{T}_s*_r*.csv"))) if s is not None]
    return st.median(v), len(v)


# label, color, line style, marker. Distinct styles and markers keep it readable in grayscale.
ARMS = [("base", "ALEX-OL", "#c0392b", "-", "o"),
        ("bg", "background only", "#8e44ad", ":", "v"),
        ("ns2048", "2048-entry nodes", "#7f8c8d", "--", "d"),
        ("side", "side buffer only", "#e67e22", ":", "^"),
        ("bgside", "side + 1 bg", "#2980b9", "-", "s"),
        ("bgside2", "side + 2 bg", "#16a085", "--", "P"),
        ("bgside4", "side + 4 bg", "#27ae60", "-.", "D")]

fig, ax = plt.subplots(figsize=(3.33, 3.1), layout="constrained")
for a, lab, col, ls, mk in ARMS:
    ys, ns = zip(*(cell(a, T) for T in THREADS))
    assert all(n == 9 for n in ns), (a, ns)
    ax.plot(THREADS, [max(y, 0.5) for y in ys], label=lab, color=col, ls=ls, marker=mk, ms=4, lw=1.2)
    print(f"  {a:8s} " + " ".join(f"{y:.0f}" for y in ys))
ax.set_xscale("log", base=2)
ax.set_xticks(THREADS, [str(t) for t in THREADS])
ax.minorticks_off()
ax.set_yscale("log")
ax.set_xlabel("threads")
ax.set_ylabel("inserts $>100\\,\\mu$s per run")
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.legend(frameon=False, ncol=2, loc="outside upper center", handlelength=1.8,
           columnspacing=0.8, handletextpad=0.4)
fig.savefig(f"{OUT}/fig_mech4m.pdf")
print(f"wrote {OUT}/fig_mech4m.pdf")
