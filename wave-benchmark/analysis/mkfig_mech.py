#!/usr/bin/env python3
"""Per-batch slow inserts at 400M keys: ALEX-OL against the side buffer with background
expansion, at 1 and 16 threads. One seed (1866), same binary and machine for both arms."""
import glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Drawn at its printed size (full text width) so every label is 10 pt, as the CFP requires.
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
BULK, TOTAL = 100_000_000, 400_000_000
NF = 23  # fields in an x-row of the current driver
BS = 12_500_000  # keys per batch; each point is drawn at its batch's midpoint


def rows(f):
    out = []
    for line in open(f):
        p = line.strip().split(",")
        if p[0] == "x" and len(p) == NF:
            out.append(((int(p[5]) - BS / 2) / 1e6, max(int(p[12]), 0.5)))
    return out


def predicted():
    w, x = [], BULK * 8 / 7
    while x < TOTAL:
        w.append(BULK + (x - BULK) // BS * BS)
        x *= 4 / 3
    return w


arms = [("base", "ALEX-OL", "#c0392b", "-", "o"),
        ("side", "side buffer only", "#e67e22", ":", "^"),
        ("bgside", "side buffer + background", "#2980b9", "-", "s"),
        ("bgside4", "side buffer + 4 background (16 threads)", "#27ae60", "--", "d")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, layout="constrained")
for ax, T in zip(axes, (1, 16)):
    for v in predicted():
        # A band covers each batch that contains a predicted burst position.
        ax.axvspan(v / 1e6, (v + BS) / 1e6, color="0.88", lw=0, zorder=0)
    for key, lab, col, ls, mk in arms:
        fs = sorted(glob.glob(f"{R}/{'r14' if T == 16 else 'r6'}/b400/{key}_books_t{T}_s1866.csv"))
        if not fs:
            continue
        x = rows(fs[0])
        ax.plot([a for a, _ in x], [b for _, b in x], label=lab, color=col, ls=ls, marker=mk,
                ms=2.5, lw=1.1)
    ax.set_yscale("log")
    ax.set_title(f"{T} thread{'s' if T > 1 else ''}")
    ax.set_xlabel("keys in index (millions)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("inserts $>100\\,\\mu$s\nper batch")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="outside upper center", ncol=2)
fig.savefig(f"{OUT}/fig_mech400.pdf")
print(f"wrote {OUT}/fig_mech400.pdf")
