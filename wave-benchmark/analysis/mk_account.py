#!/usr/bin/env python3
"""Where each index pays for inserts: every concurrent index on one 400M workload (paper Table:
tab_account).

  python3 analysis/mk_account.py [outdir]

400M SOSD books keys, bulk 100M then 300M operations in 24 batches, 16 threads, final driver.
All from one server (r9, r10, r11, r14/acct were measured on the same machine).
  insert-only: r9/b400 (ALEX-OL arms), r14/acct/ins (XIndex, FINEdex), r10/ins + r11/acct/ins (others); 3 seeds
  50/50:       r14/acct/mix (ALEX-OL arms, XIndex, FINEdex), r10/mix + r11/acct/mix (others); 3 seeds
  90/10:       r10/read90 (seed 1866) and r11/acct/read90 (seeds 5, 72), all indexes
Cells are medians over the runs present. Memory is peak resident set size from the verify logs
(about 5 GB of it is the driver's key arrays, common to all). XIndex and FINEdex runs that
failed only the key check (.fail2) keep their timings; runs that hung (.fail124) have none.
"""
import glob, os, re, statistics as st, sys

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
SEEDS = (1866, 5, 72)
LOGS = ("r9/mq9_verify.log", "r10/mq10_verify.log", "r11/mq11_verify.log", "r14/mq14v6_verify.log")


def run(f):
    xs = [l.strip().split(",") for l in open(f) if l.startswith("x,") and l.count(",") == 22]
    if len(xs) != 24:
        return None
    return {"slow": sum(int(r[12]) for r in xs), "wall": sum(int(r[21]) for r in xs) / 1e9,
            "rp50": st.median(int(r[15]) for r in xs)}


RSS, RC = {}, {}
for log in LOGS:
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
            RC[m.group(2)] = int(m.group(1))
            if last is not None:
                RSS[m.group(2)] = last
            last = None


def files(paths):
    out = []
    for p in paths:
        f = os.path.join(R, p)
        if os.path.exists(f):
            out.append((p, f))
        elif os.path.exists(f + ".fail2"):
            out.append((p, f + ".fail2"))
    return out


def med(paths, key):
    v = [r[key] for _, f in files(paths) for r in [run(f)] if r]
    return (st.median(v), len(v)) if v else (None, 0)


def rss(paths):
    v = [RSS[p] for p in paths if p in RSS and RC.get(p) != 124]
    return st.median(v) if v else None


def outcome(paths):
    """Key-check outcome over every run of this index in the table's run sets."""
    rcs = [RC[p] for p in paths if p in RC]
    bad = sum(1 for c in rcs if c == 2)
    hung = sum(1 for c in rcs if c == 124)
    if hung:
        return f"hung {hung}/{len(rcs)}"
    if bad:
        return f"lost keys {bad}/{len(rcs)}"
    return "passed" if rcs else "--"


def src(arm):
    if arm in ("base", "bgside4"):
        ins = [f"r9/b400/{arm}_books_t16_s{s}.csv" for s in SEEDS]
        mix = [f"r14/acct/mix/{arm}_books_t16_s{s}.csv" for s in SEEDS]
    elif arm in ("xindex", "finedex"):
        ins = [f"r14/acct/ins/{arm}_books_t16_s{s}.csv" for s in SEEDS]
        mix = [f"r14/acct/mix/{arm}_books_t16_s{s}.csv" for s in SEEDS]
    else:
        ins = [f"r10/ins/{arm}_books_t16_s1866.csv"] + [f"r11/acct/ins/{arm}_books_t16_s{s}.csv" for s in (5, 72)]
        mix = [f"r10/mix/{arm}_books_t16_s1866.csv"] + [f"r11/acct/mix/{arm}_books_t16_s{s}.csv" for s in (5, 72)]
    rd = [f"r10/read90/{arm}_books_t16_s1866.csv"] + [f"r11/acct/read90/{arm}_books_t16_s{s}.csv" for s in (5, 72)]
    return ins, mix, rd


ROWS = [("base", "ALEX-OL"), ("bgside4", "\\quad + side buffer, background"),
        ("lippol", "LIPP-OL"), ("sali", "SALI"), ("xindex", "XIndex"), ("finedex", "FINEdex"),
        ("btreebulk", "B$^+$-tree-OLC"), ("artolc", "ART-OLC")]
KIND = {"base": "gapped array", "bgside4": "gapped array", "lippol": "model placement",
        "sali": "model placement", "xindex": "buffered", "finedex": "buffered",
        "btreebulk": "classical", "artolc": "classical"}
f0 = lambda v, fmt: "--" if v is None else format(v, fmt).replace(",", "{,}")
print("== 400M books, 16 threads")
with open(f"{OUT}/tab_account.tex", "w") as o:
    o.write("\\begin{tabular}{llrrrrrl}\n\\toprule\n")
    o.write("& & \\multicolumn{3}{c}{insert-only} & \\multicolumn{2}{c}{lookup (ns)} & \\\\\n")
    o.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n")
    o.write("Index & design & $>$100\\,\\textmu s & time (s) & RSS (GB) & 50/50 & 90/10 & key check \\\\\n\\midrule\n")
    for arm, lab in ROWS:
        ins, mix, rd = src(arm)
        s, n = med(ins, "slow"); w, _ = med(ins, "wall"); g = rss(ins)
        l50, n50 = med(mix, "rp50"); l90, n90 = med(rd, "rp50")
        chk = outcome(ins + mix + rd)
        o.write(f"{lab} & {KIND[arm]} & {f0(s, ',.0f')} & {f0(w, '.1f')} & {f0(g, '.1f')} & "
                f"{f0(l50, '.0f')} & {f0(l90, '.0f')} & {chk} \\\\\n")
        print(f"  {arm:9s} slow={s} [n={n}] time={w} rss={g} l50={l50} [n={n50}] l90={l90} [n={n90}] check={chk}")
    o.write("\\bottomrule\n\\end{tabular}\n")
print("wrote", f"{OUT}/tab_account.tex")
