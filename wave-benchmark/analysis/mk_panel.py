#!/usr/bin/env python3
"""Summaries of the experiments the 2026-09-24 review panel asked for (results/raw/r7, r8).

  python3 analysis/waves/mk_panel.py

cpu     CPU time over the timed interval (background threads included) at equal and
        unequal core budgets, 400M books.
recent  50/50 mix whose lookups target keys the same thread just inserted, so they reach
        nodes under rebuild and their side buffers. Every such lookup must hit.
skew    Zipfian and sorted insert orders, 4M synthetic and 400M books.
r8      ALEX-OL with a permanent per-node buffer (sidealways), if present.
Medians over seeds; n in brackets. Failed runs (.failN) are listed, never averaged.
"""
import glob, os, re, statistics as st

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))


def run(f):
    xs, cpu, ls2 = [], 0, []
    for l in open(f):
        p = l.strip().split(",")
        if p[0] == "x" and len(p) == 23:
            xs.append(p)
        elif p[0] == "CPU":
            cpu += int(p[5])
        elif p[0] == "LOCK2":
            ls2.append(p)
    if not xs:
        return None
    d = {"slow": sum(int(r[12]) for r in xs), "wall": sum(int(r[21]) for r in xs) / 1e9,
         "cpu": cpu / 1e9, "n10": sum(int(r[11]) for r in xs), "n1ms": sum(int(r[13]) for r in xs),
         "rp50": st.median(int(r[15]) for r in xs) if int(xs[0][20]) else 0,
         "rp99": st.median(int(r[16]) for r in xs) if int(xs[0][20]) else 0,
         "rslow": sum(int(r[19]) for r in xs), "mem": int(xs[-1][22]) / 1e9, "batches": len(xs)}
    if ls2:  # side_ins, side_read are fields 15-16 of LOCK2
        d["side_ins"] = sum(int(r[15]) for r in ls2)
        d["side_read"] = sum(int(r[16]) for r in ls2)
    return d


def summarize(pattern, keys=("slow", "wall", "cpu", "rp50", "rslow", "mem")):
    fs = sorted(glob.glob(os.path.join(R, pattern)))
    rs = [r for r in (run(f) for f in fs) if r]
    if not rs:
        return None
    out = {k: st.median(r[k] for r in rs if k in r) for k in keys if any(k in r for r in rs)}
    out["n"] = len(rs)
    return out


def show(label, pattern, keys=("slow", "wall", "cpu", "rp50", "rslow", "mem")):
    s = summarize(pattern, keys)
    if s is None:
        print(f"  {label:34s} (no runs)")
        return
    body = " ".join(f"{k}={s[k]:.2f}" if isinstance(s[k], float) and s[k] < 100 else f"{k}={s[k]:,.0f}"
                    for k in keys if k in s)
    print(f"  {label:34s} [n={s['n']}] {body}")


print("== CPU time, 400M books (seconds of CPU over the timed interval)")
for T, a in ((16, "base"), (16, "bgside4"), (12, "bgside4"), (1, "base"), (1, "bgside")):
    show(f"{a} t{T}", f"r7/cpu/{a}_books_t{T}_s*.csv")
print("== Equal cores (r6/eq: 12 foreground + 4 background vs 16 foreground)")
show("bgside4 t12 (r6/eq)", "r6/eq/bgside4_books_t12_s*.csv")
show("base t16 (r6/b400)", "r6/b400/base_books_t16_s*.csv")
show("bgside4 t16 (r6/b400)", "r6/b400/bgside4_books_t16_s*.csv")
print("== No spare cores (r6/sat: 32 foreground threads)")
for a in ("base", "bgside4"):
    show(f"{a} t32", f"r6/sat/{a}_books_t32_s*.csv")
print("== Lookups of just-inserted keys, 50/50, 400M books, 16 threads")
for a in ("base", "side", "bgside", "bgside4"):
    show(a, f"r7/recent/{a}_books_t16_s*.csv")
for a in ("sidels", "bgside4ls"):
    show(a + " (lockstats)", f"r7/recent/{a}_books_t16_s*.csv", ("slow", "rp50", "side_ins", "side_read"))
print("== Skewed and sorted inserts, 4M")
for o in ("zipf", "sorted"):
    for T in (1, 16):
        for a in ("base", "bgside", "bgside4"):
            if T == 1 and a == "bgside4":
                continue
            show(f"{o} {a} t{T}", f"r7/skew4m/{a}_{o}_t{T}_s*.csv", ("slow", "wall", "cpu"))
print("== Skewed and sorted inserts, 400M books")
for o, T, arms in (("zipf", 16, ("base", "bgside4")), ("zipf", 1, ("base", "bgside")),
                   ("sorted", 16, ("base", "bgside4")), ("sorted", 1, ("base", "bgside"))):
    for a in arms:
        show(f"{o} {a} t{T}", f"r7/skew/{a}_{o}_t{T}_s*.csv")
print("== Permanent buffer (sidealways), if run")
for p in sorted(glob.glob(os.path.join(R, "r8", "*"))):
    if os.path.isdir(p):
        arms = sorted({re.sub(r"_s\d+(_r\d+)?\.csv$", "", os.path.basename(f))
                       for f in glob.glob(os.path.join(p, "*.csv"))})
        for a in arms:
            show(f"{os.path.basename(p)}/{a}", f"r8/{os.path.basename(p)}/{a}_s*.csv")
print("== Failed runs")
for f in sorted(glob.glob(os.path.join(R, "r7", "*", "*.fail*")) +
                glob.glob(os.path.join(R, "r8", "*", "*.fail*"))):
    print("  ", os.path.relpath(f, R))
