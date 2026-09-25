#!/usr/bin/env python3
"""Numbers the paper quotes from the reruns on the final binary (results/raw/r13).

  python3 analysis/waves/mk_r13.py

Same definitions as the original scripts: a run's burst ratio is its worst batch p99.9 over its
median batch p99.9; cells are medians over runs; "predicted" batches are those containing a burst
position computed from the constants (first at N0*8/7, then x4/3).
"""
import glob, os, statistics as st

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
T5 = (1, 2, 4, 8, 16)


def rows(f):
    xs, l2 = [], []
    for l in open(f):
        p = l.strip().split(",")
        if p[0] == "x" and len(p) == 23:
            xs.append(p)
        elif p[0] == "LOCK2":
            l2.append(p)
    return xs, l2


def predicted(xs, n0):
    keys = [int(r[5]) for r in xs]
    bs = keys[1] - keys[0]
    tot = keys[-1]
    w, x = [], n0 * 8 / 7
    while x < tot:
        w.append(x)
        x *= 4 / 3
    return {i for i in range(len(keys)) for u in w if keys[i] - bs < u <= keys[i]}


def stats(f, n0):
    xs, l2 = rows(f)
    if len(xs) < 20:
        return None
    p = sorted(int(r[8]) / 1000 for r in xs)
    g = [int(r[12]) for r in xs]
    wi = predicted(xs, n0)
    d = dict(ratio=p[-1] / st.median(p), floor=st.median(p), slow=sum(g),
             share=sum(g[i] for i in wi) / max(sum(g), 1),
             quiet=st.median([g[i] for i in range(len(g)) if i not in wi]),
             wave=st.median([g[i] for i in wi]), worst999=p[-1])
    if l2:
        rt = [int(r[6]) for r in l2]
        d["retry_total"] = sum(rt)
        d["retry_med"] = st.median(rt)
        d["retry_first"] = rt[min(wi)] if wi else 0
    return d


def cell(pat, n0=1_000_000):
    v = [s for s in (stats(f, n0) for f in sorted(glob.glob(os.path.join(R, pat)))) if s]
    return v


def m(v, k):
    return st.median(x[k] for x in v) if v else float("nan")


print("== 4M thread sweep: burst ratio (worst/median batch p99.9) and floor (us), median of runs")
for idx in ("alexol", "lippol", "sali", "btreeolc", "artolc", "alexolstag"):
    rs = [cell(f"r13/sweep/{idx}_t{T}_s*_r*.csv") for T in T5]
    print(f"  {idx:10s} n={[len(c) for c in rs]} ratio={[round(m(c,'ratio'),1) for c in rs]} "
          f"floor={[round(m(c,'floor'),2) for c in rs]}")
print("== predicted-batch share of high-latency inserts, 4M (released vs randomized)")
for T in (1, 16):
    a, b = cell(f"r13/sweep/alexol_t{T}_s*_r*.csv"), cell(f"r13/sweep/alexolstag_t{T}_s*_r*.csv")
    print(f"  t{T}: released {m(a,'share'):.0%} randomized {m(b,'share'):.0%} "
          f"| slow {m(a,'slow'):.0f} vs {m(b,'slow'):.0f} | quiet {m(a,'quiet'):.0f} vs {m(b,'quiet'):.0f}"
          f" | worst p99.9 {m(a,'worst999'):.1f} vs {m(b,'worst999'):.1f} us")
print("== restarts (lockstats), 4M")
for T in (1, 16):
    a, b = cell(f"r13/ls/base_t{T}_s*_r*.csv"), cell(f"r13/ls/stag_t{T}_s*_r*.csv")
    if a and b and "retry_total" in a[0]:
        print(f"  t{T}: base per-batch median {m(a,'retry_med'):.3g} first-burst {m(a,'retry_first'):.4g} "
              f"total {m(a,'retry_total'):.3g} | stag total {m(b,'retry_total'):.3g} "
              f"ratio {m(b,'retry_total')/max(m(a,'retry_total'),1):.2f} | slow ratio {m(b,'slow')/max(m(a,'slow'),1):.2f}")
print("== randomized-density variants, 4M (high-latency inserts)")
for T in (1, 4, 8, 16):
    print(f"  t{T}: " + " ".join(f"{k}={m(cell(f'r13/var/{k}_t{T}_s*_r*.csv'),'slow'):.0f}"
                               for k in ("base", "d017", "d008", "low017")))
print("== NUMA-pinned (OMP_PROC_BIND=close), 4M")
for T in T5:
    a, b = cell(f"r13/pin/base_t{T}_s*_r*.csv"), cell(f"r13/pin/stag_t{T}_s*_r*.csv")
    print(f"  t{T}: base {m(a,'slow'):.0f} randomized {m(b,'slow'):.0f}")
print("== B+-tree fill, burst ratio")
for k in ("unif07", "rand07", "unif09", "rand09"):
    print(f"  {k}: " + " ".join(f"t{T}={m(cell(f'r13/bt/{k}_t{T}_s*_r*.csv'),'ratio'):.1f}" for T in (1, 4, 8, 16)))
print("== node-size sweep, 4M (high-latency inserts)")
for T in (1, 16):
    print(f"  t{T}: " + " ".join(f"{nb//16}e={m(cell(f'r13/ns/nb{nb}_t{T}_s*_r*.csv'),'slow'):.0f}"
                               for nb in (512, 4096, 32768, 262144, 524288, 2097152)))
print("== 400M books, released vs randomized density (same server)")
for T in (1, 16):
    a = cell(f"r13/stag400/base_books_t{T}_s*.csv", 100_000_000)
    b = cell(f"r13/stag400/stag_books_t{T}_s*.csv", 100_000_000)
    print(f"  t{T} n={len(a)},{len(b)}: slow {m(a,'slow'):.0f} vs {m(b,'slow'):.0f} "
          f"({(m(b,'slow')/m(a,'slow')-1)*100:+.1f}%) share {m(a,'share'):.0%} vs {m(b,'share'):.0%} "
          f"quiet {m(a,'quiet'):.0f} vs {m(b,'quiet'):.0f} wave {m(a,'wave'):.0f}")
print("== densities right after bulk load")
for f in sorted(glob.glob(os.path.join(R, "r13/dens/*.txt"))):
    print("  ", os.path.basename(f), open(f).read().strip())
