#!/usr/bin/env python3
"""Comparison with XIndex and FINEdex at 400M books keys (paper Table: tab_cmp).

  python3 analysis/waves/mk_cmp.py [outdir]

Every cell is the median over three seeds (1866, 5, 72), all runs on the final binary:
  r6/b400   ALEX-OL arms, insert-only, 1 and 16 threads (final rerun, bgx6)
  r7/cmp    XIndex and FINEdex, insert-only, 16 threads (and 1 thread, seeds 5 and 72)
  final_cmp XIndex and FINEdex, insert-only, 1 thread, seed 1866 (their code is unchanged)
  r7/cmpmix all four, 50/50 mix, 16 threads
Memory is peak resident set size from the verify logs, which includes about 5 GB of driver
arrays common to all. A run whose exhaustive check failed has no .csv (only .failN), so it
cannot enter a median; failed runs are listed at the end.
"""
import glob, os, re, statistics as st, sys

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/waves"
os.makedirs(OUT, exist_ok=True)
SEEDS = (1866, 5, 72)


def run(f):
    xs = [l.strip().split(",") for l in open(f) if l.startswith("x,") and l.count(",") == 22]
    if len(xs) != 24:
        return None
    return {"slow": sum(int(r[12]) for r in xs), "wall": sum(int(r[21]) for r in xs) / 1e9,
            "rp50": st.median(int(r[15]) for r in xs)}


def rss_map():
    """Result path (relative to its run directory, e.g. r6/b400/x.csv) -> peak RSS in GB."""
    out = {}
    for log in ("r6/mq6_verify.log", "r7/mq7_verify.log", "final_cmp_verify.log"):
        p = os.path.join(R, log)
        if not os.path.exists(p):
            continue
        last = None
        for line in open(p):
            m = re.match(r"RSS \S+ t\d+ seed\d+: end_kb=\d+ peak_kb=(\d+)", line)
            if m:
                last = int(m.group(1)) / 1e6
            m = re.match(r"RC (\d+) (\S+)", line)
            if m:
                if last is not None and m.group(1) == "0":
                    out[m.group(2)] = last
                last = None
    return out


RSS = rss_map()


def cell(paths, key):
    """Median of key over the runs that exist; returns (value, n)."""
    vals = []
    for p in paths:
        f = os.path.join(R, p)
        if os.path.exists(f):
            r = run(f)
            if r:
                vals.append(r[key])
    return (st.median(vals), len(vals)) if vals else (None, 0)


def rss_cell(paths):
    # Verify logs name results relative to the bgx6/bgx7 directory (r6/..., r7/..., r4/cmp/...).
    vals = [RSS[p] for p in paths if p in RSS]
    return (st.median(vals), len(vals)) if vals else (None, 0)


def t1_paths(arm):
    if arm in ("xindex", "finedex"):
        return [f"final_cmp/{arm}_books_t1_s1866.csv"] + \
               [f"r7/cmp/{arm}_books_t1_s{s}.csv" for s in (5, 72)]
    return [f"r6/b400/{arm}_books_t1_s{s}.csv" for s in SEEDS]


def t16_paths(arm):
    d = "r7/cmp" if arm in ("xindex", "finedex") else "r6/b400"
    return [f"{d}/{arm}_books_t16_s{s}.csv" for s in SEEDS]


def mix_paths(arm):
    return [f"r7/cmpmix/{arm}_books_t16_s{s}.csv" for s in SEEDS]


ROWS = [("base", "base", "ALEX-OL"), ("bgside", "bgside4", "\\quad + side buffer, background"),
        ("xindex", "xindex", "XIndex"), ("finedex", "finedex", "FINEdex")]
fmt = lambda v, f: "--" if v is None else format(v, f)
print("== 400M books, medians of three seeds (n per cell in brackets)")
with open(f"{OUT}/tab_cmp.tex", "w") as o:
    o.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    o.write("& \\multicolumn{2}{c}{1 thread} & \\multicolumn{3}{c}{16 threads} & "
            "\\multicolumn{2}{c}{16 threads, 50/50} \\\\\n")
    o.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n")
    o.write("Index & slow & time & slow & time & RSS & slow & lookup \\\\\n\\midrule\n")
    for a1, a16, lab in ROWS:
        s1, n1 = cell(t1_paths(a1), "slow"); w1, _ = cell(t1_paths(a1), "wall")
        s16, n16 = cell(t16_paths(a16), "slow"); w16, _ = cell(t16_paths(a16), "wall")
        g, ng = rss_cell(t16_paths(a16))
        sm, nm = cell(mix_paths(a16), "slow"); lk, _ = cell(mix_paths(a16), "rp50")
        cells = [fmt(s1, ",.0f"), fmt(w1, ".0f"), fmt(s16, ",.0f"), fmt(w16, ".1f"),
                 fmt(g, ".1f"), fmt(sm, ",.0f"), fmt(lk, ".0f")]
        o.write(f"{lab} & " + " & ".join(c.replace(",", "{,}") for c in cells) + " \\\\\n")
        print(f"  {a16:8s} t1 slow={s1} time={w1} [n={n1}] | t16 slow={s16} time={w16} [n={n16}] "
              f"rss={g} [n={ng}] | mix slow={sm} lookup_p50={lk} [n={nm}]")
    o.write("\\bottomrule\n\\end{tabular}\n")
print("failed runs (excluded):")
for d in ("r7/cmp", "r7/cmpmix", "final_cmp", "final_cmpmix"):
    for f in sorted(glob.glob(os.path.join(R, d, "*.fail*"))):
        print("  ", os.path.relpath(f, R))
print("wrote", f"{OUT}/tab_cmp.tex")
