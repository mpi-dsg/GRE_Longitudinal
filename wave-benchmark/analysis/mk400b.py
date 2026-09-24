#!/usr/bin/env python3
"""400M-key comparison table, generated from the CSVs."""
import glob, os, statistics as st, sys

import os as _os
R = _os.environ.get("WAVE_RESULTS",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results", "raw"))
C="tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops batch_ns mem".split()
OUT=sys.argv[1] if len(sys.argv)>1 else "figures/waves"
os.makedirs(OUT,exist_ok=True)
BULK,TOT=100_000_000,400_000_000
def L(f): return [dict(zip(C,l.strip().split(','))) for l in open(f) if l.count(',')==16]
def waves(keys):
    w,x=[],BULK*8/7
    while x<TOT: w.append(x); x*=4/3
    bs=keys[1]-keys[0]
    return {min(range(len(keys)),key=lambda i:abs(keys[i]-bs/2-v)) for v in w}
ROWS=[("alexol","ALEX-OL"),("alexolstag","\\quad + randomized density"),
      ("lippol","LIPP-OL"),("sali","SALI"),
      ("btreebulk","B$^+$-tree-OLC"),("artolc","ART-OLC")]
def cell(key,T):
    fs=sorted(glob.glob(f"{R}/scale400_books/{key}_books_t{T}_s*.csv"))
    out=[]
    for f in fs:
        x=L(f)
        if len(x)<20: continue
        keys=[int(r['keys']) for r in x]; wi=waves(keys)
        g=[int(r['n100']) for r in x]; tot=sum(g)
        out.append((st.median([g[i] for i in wi]), st.median([g[i] for i in range(len(x)) if i not in wi]),
                    tot, 100*sum(g[i] for i in wi)/max(1,tot), float(x[-1]['mem'])/1e9))
    if not out: return None
    return [st.median([o[k] for o in out]) for k in range(5)]
with open(f"{OUT}/tab_scale400.tex","w") as o:
    o.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    o.write("& \\multicolumn{3}{c}{1 thread} & \\multicolumn{3}{c}{16 threads} & \\\\\n")
    o.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    o.write("Index & wave & quiet & total & wave & quiet & total & mem \\\\\n\\midrule\n")
    for key,lab in ROWS:
        a=cell(key,1); b=cell(key,16)
        if not a: continue
        mem = f"{a[4]:.1f}" if a[4] > 0 else "--"
        bs = (f"{b[0]:.0f} & {b[1]:.0f} & {b[2]:.0f}" if b else "-- & -- & --")
        o.write(f"{lab} & {a[0]:.0f} & {a[1]:.0f} & {a[2]:.0f} & {bs} & {mem} \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")
print("wrote", f"{OUT}/tab_scale400.tex")
