#!/usr/bin/env python3
"""Remedy table, from slow-operation counts.

Deliberately not percentiles: over a 100k batch, p99.9 is the 100th largest sample, so a
change that merely spreads the same slow operations more thinly reports a large improvement
that has not occurred. See results/README.md.
"""
import glob, os, statistics as st, sys
R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/waves"
# Final-driver rows: 23 fields (see results/README.md). Source: r13/sweep, the rerun on the
# final binary of the original remedy runs (alexol = released density, alexolstag = randomized).
C = ("tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops "
     "r_p50 r_p99 r_p999 r_max r_n100 r_ops batch_ns mem").split()
SRC = {"base": "alexol", "stag": "alexolstag"}
BULK, TOT = 1_000_000, 4_000_000

def L(f):
    return [dict(zip(C, l.strip().split(','))) for l in open(f) if l.startswith('x,') and l.count(',') == 22]

def cell(key, T):
    w, x = [], BULK * 8 / 7
    while x < TOT: w.append(x); x *= 4 / 3
    tot, wv, qt, p9 = [], [], [], []
    for f in sorted(glob.glob(f"{R}/r13/sweep/{SRC[key]}_t{T}_s*_r*.csv")):
        v = L(f)
        if len(v) < 25: continue
        keys = [int(r['keys']) for r in v]; bs = keys[1] - keys[0]
        wi = {min(range(len(keys)), key=lambda i: abs(keys[i] - bs / 2 - u)) for u in w}
        g = [int(r['n100']) for r in v]
        tot.append(sum(g)); wv.append(st.median([g[i] for i in wi]))
        qt.append(st.median([g[i] for i in range(len(v)) if i not in wi]))
        p9.append(max(float(r['p999']) for r in v) / 1000)
    return None if not tot else [st.median(z) for z in (tot, wv, qt, p9)]

with open(f"{OUT}/tab_remedy.tex", "w") as o:
    o.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
    o.write("& & \\multicolumn{3}{c}{inserts $>100\\,\\mu$s} & p99.9 \\\\\n")
    o.write("\\cmidrule(lr){3-5}\n")
    o.write("Thr & density & total & wave & quiet & worst \\\\\n\\midrule\n")
    for T in (1, 16):
        for key, lab in (("base", "released"), ("stag", "randomized")):
            c = cell(key, T)
            if c: o.write(f"{T} & {lab} & {c[0]:.0f} & {c[1]:.0f} & {c[2]:.0f} & {c[3]:.1f} \\\\\n")
        if T == 1: o.write("\\midrule\n")
    o.write("\\bottomrule\n\\end{tabular}\n")
print(f"wrote {OUT}/tab_remedy.tex")
