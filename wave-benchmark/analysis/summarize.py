#!/usr/bin/env python3
"""Print every headline number in the paper, computed from results/raw.

Needs nothing but this repository. Run it to check the claims without taking anyone's
word for them:

    python3 analysis/summarize.py
"""
import glob, os, statistics as st, sys

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
C17 = ("tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops "
       "batch_ns mem").split()

def L(f, n=16):
    return [dict(zip(C17, l.strip().split(','))) for l in open(f) if l.count(',') == n]

def waves(bulk, total):
    w, x = [], bulk * 8 / 7
    while x < total: w.append(x); x *= 4 / 3
    return w

def split(f, bulk, total, n=16):
    x = L(f, n)
    if len(x) < 5: return None
    keys = [int(r['keys']) for r in x]; bs = keys[1] - keys[0]
    wi = {min(range(len(keys)), key=lambda i: abs(keys[i] - bs / 2 - v))
          for v in waves(bulk, total)}
    g = [int(r['n100']) for r in x]
    return (st.median([g[i] for i in wi]),
            st.median([g[i] for i in range(len(x)) if i not in wi]),
            sum(g), 100 * sum(g[i] for i in wi) / max(sum(g), 1),
            float(x[-1]['mem']) / 1e9)

def hdr(t): print(f"\n{t}\n" + "-" * len(t))

hdr("400M SOSD books, bulk 100M, slow inserts (>100us) per batch")
print(f"{'index':<24}{'thr':>4}{'wave':>8}{'quiet':>8}{'ratio':>8}{'total':>9}{'in-wave':>9}{'mem GB':>9}")
for T in (1, 16):
    for k, lab in (("alexol","ALEX-OL"),("alexolstag","  +randomized"),("lippol","LIPP-OL"),
                   ("sali","SALI"),("btreebulk","B+tree-OLC"),("artolc","ART-OLC")):
        f = f"{R}/scale400_books/{k}_books_t{T}_s1866.csv"
        if not os.path.exists(f): continue
        r = split(f, 1e8, 4e8)
        if r: print(f"{lab:<24}{T:>4}{r[0]:>8.0f}{r[1]:>8.0f}{r[0]/max(r[1],1):>7.0f}x"
                    f"{r[2]:>9.0f}{r[3]:>8.0f}%{(f'{r[4]:.1f}' if r[4]>0 else '--'):>9}")

hdr("400M OSM (hard key distribution)")
for T in (1, 16):
    for k, lab in (("alexol","ALEX-OL"),("alexolstag","  +randomized"),("sali","SALI"),("artolc","ART-OLC")):
        f = f"{R}/scale400_osm/{k}_t{T}_s1866.csv"
        if not os.path.exists(f): continue
        r = split(f, 1e8, 4e8)
        if r: print(f"{lab:<24}{T:>4}{r[0]:>8.0f}{r[1]:>8.0f}{r[0]/max(r[1],1):>7.0f}x{r[2]:>9.0f}{r[3]:>8.0f}%")

hdr("The remedy: randomized bulk-load density (1M->4M, 9 runs/cell)")
print("counts, not percentiles: see results/README.md for why")
for T in (1, 16):
    for k, lab in (("base","baseline"),("stag","randomized")):
        tot = []
        for f in sorted(glob.glob(f"{R}/remedy_recheck/{k}_t{T}_s*_r*.csv")):
            x = L(f)
            if len(x) >= 25: tot.append(sum(int(r['n100']) for r in x))
        if tot: print(f"  {T:>2} thr  {lab:<12} total slow inserts {st.median(tot):>8.0f}")

hdr("Node size (the remedy that reduces cost rather than moving it)")
print(f"{'thr':>4}{'entries':>9}{'total slow':>12}{'wall s':>9}")
for T in (1, 16):
    for nb in (4096, 32768, 262144, 524288, 2097152):
        tot, wall = [], []
        for f in sorted(glob.glob(f"{R}/nodesize/nb{nb}_t{T}_s*_r*.csv")):
            x = L(f)
            if len(x) >= 25:
                tot.append(sum(int(r['n100']) for r in x))
                wall.append(sum(float(r['batch_ns']) for r in x) / 1e9)
        if tot: print(f"{T:>4}{nb//16:>9}{st.median(tot):>12.0f}{st.median(wall):>9.2f}")

hdr("Concurrency mechanism: insert restarts (lockstats)")
for T in (1, 16):
    for k, lab in (("base","baseline"),("stag","randomized")):
        rt = []
        for f in sorted(glob.glob(f"{R}/lockstats/{k}_t{T}_s*_r*.csv")):
            v = [l.split(',') for l in open(f) if l.startswith("LOCK,")]
            if v: rt.append(sum(int(x[6]) for x in v))
        if rt: print(f"  {T:>2} thr  {lab:<12} retries {st.median(rt):>14.0f}")
print("\n(at 1 thread the counter is zero: with one writer nothing restarts)")
