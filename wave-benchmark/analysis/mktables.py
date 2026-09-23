#!/usr/bin/env python3
"""Emit LaTeX tables for waves.tex straight from the result CSVs.

Every number in the paper comes from here, so no value is transcribed by hand.
"""
import glob, statistics as st, os, sys

import os as _os
R = _os.environ.get("WAVE_RESULTS",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "results", "raw"))
COL = "tag index seed threads batch keys p50 p99 p999 max batch_ns mem".split()
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)

def L(f):
    return [dict(zip(COL, l.strip().split(',')))
            for l in open(f) if l.count(',') == 11 and not l.startswith('tag')]

def runs(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        x = L(f)
        if len(x) < 25: continue
        p = sorted(float(r['p999']) / 1000 for r in x)
        m = st.median(p)
        out.append(dict(floor=m, worst=p[-1], burst=p[-1] / m,
                        wall=sum(float(r['batch_ns']) for r in x) / 1e9,
                        mem=float(x[-1]['mem']) / 1e6))
    return out

def med(rs, k): return st.median([r[k] for r in rs]) if rs else float('nan')

THREADS = [1, 2, 4, 8, 16]
ROWS = [("alexol", "ALEX-OL", "learned"), ("lippol", "LIPP-OL", "learned"),
        ("sali", "SALI", "learned"), ("btreeolc", "B$^+$-tree-OLC", "classical"),
        ("artolc", "ART-OLC", "classical")]

# Table 1: burstiness across thread counts, all indexes.
with open(f"{OUT}/tab_burst.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
    o.write("Index & Class & " + " & ".join(f"{t}\\,thr" for t in THREADS) + " \\\\\n\\midrule\n")
    for key, lab, cls in ROWS:
        cells = []
        for t in THREADS:
            r = runs(f"{R}/sweep4m_threads/{key}_t{t}_s*_r*.csv")
            cells.append(f"{med(r,'burst'):.1f}$\\times$" if r else "--")
        o.write(f"{lab} & {cls} & " + " & ".join(cells) + " \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")

# Table 2: absolute worst-batch p99.9.
with open(f"{OUT}/tab_worst.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
    o.write("Index & Class & " + " & ".join(f"{t}\\,thr" for t in THREADS) + " \\\\\n\\midrule\n")
    for key, lab, cls in ROWS:
        cells = []
        for t in THREADS:
            r = runs(f"{R}/sweep4m_threads/{key}_t{t}_s*_r*.csv")
            cells.append(f"{med(r,'worst'):.1f}" if r else "--")
        o.write(f"{lab} & {cls} & " + " & ".join(cells) + " \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")

# Table 3 (the remedy) is NOT generated here. It was originally written from p99.9 over
# 100k batches, which reports a 240-fold improvement that does not exist: the baseline puts
# ~92 slow inserts in a wave batch, above the 100th-rank cutoff, and the randomized arm
# spreads the same operations at ~20 per batch, below it. That claim was withdrawn. The
# table now comes from mk_remedy.py, which counts slow operations instead.

# Table 4: the B+-tree given the same remedy.
BT = [("unif07", "uniform 0.70"), ("rand07", "randomized $0.70\\pm0.20$"),
      ("unif09", "uniform 0.90"), ("rand09", "randomized $0.90\\pm0.09$")]
with open(f"{OUT}/tab_btree.tex", "w") as o:
    o.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    o.write("Leaf fill & " + " & ".join(f"{t}\\,thr" for t in (1, 4, 8, 16)) + " \\\\\n\\midrule\n")
    for key, lab in BT:
        cells = []
        for t in (1, 4, 8, 16):
            r = runs(f"{R}/btree_remedy/{key}_t{t}_s*_r*.csv")
            cells.append(f"{med(r,'burst'):.1f}$\\times$" if r else "--")
        o.write(f"{lab} & " + " & ".join(cells) + " \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")

print("wrote:", ", ".join(sorted(os.listdir(OUT))))
