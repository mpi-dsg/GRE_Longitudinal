#!/usr/bin/env python3
"""Tables and text numbers for the side buffer and background expansion (Section 5 / Q5),
and the node-size remedy at 400M. Generated from the CSVs; nothing is typed by hand.

  python3 analysis/waves/mk_mech.py [outdir]

Reads results/raw/r6 (final rerun on the final binary) and {ns400,seeds400,mix400,thr}. Prints every number the text quotes.
"""
import glob, os, statistics as st, sys

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/waves"
os.makedirs(OUT, exist_ok=True)

X = ("tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops "
     "r_p50 r_p99 r_p999 r_max r_n100 r_ops batch_ns mem").split()
LK = ("tag seed T batch keys retry retry_smo retry_ins parentspin fg_smo bg_smo bg_skip "
      "smo_ns smo_max_ns side_ins side_read side_full smo_split retry_noside split_ns").split()
L3 = ("tag seed T batch keys slow_fgsmo slow_sealed slow_rsmo slow_rins slow_none root_expand "
      "boundary_hit").split()


def rows(f):
    xs, ls = [], []
    global _l3
    _l3 = []
    for line in open(f):
        p = line.strip().split(",")
        if p[0] == "x" and len(p) == len(X):
            xs.append({k: (v if k in ("tag", "index") else float(v)) for k, v in zip(X, p)})
        elif p[0] == "LOCK2" and len(p) == len(LK) + 1:
            ls.append({k: float(v) for k, v in zip(LK, p[1:]) if k != "tag"})
        elif p[0] == "LOCK3" and len(p) in (len(L3), len(L3) + 1):
            _l3.append({k: float(v) for k, v in zip(L3, p[1:]) if k != "tag"})
    return xs, ls


def run_stats(f):
    xs, ls = rows(f)
    if not xs:
        return None
    s = {
        "slow": sum(r["n100"] for r in xs),
        "maxb": max(r["n100"] for r in xs),
        "wall": sum(r["batch_ns"] for r in xs) / 1e9,
        "mem": xs[-1]["mem"] / 1e9,
        "rp50": st.median(r["r_p50"] for r in xs),
        "rp99": st.median(r["r_p99"] for r in xs),
        "rslow": sum(r["r_n100"] for r in xs),
        "batches": len(xs),
    }
    if ls:
        for k in ("retry", "retry_smo", "retry_ins", "fg_smo", "bg_smo", "bg_skip", "smo_ns",
                  "side_ins", "side_read", "side_full", "smo_split"):
            s[k] = sum(r[k] for r in ls)
        s["smo_max_ns"] = max(r["smo_max_ns"] for r in ls)
    for k in ("slow_fgsmo", "slow_sealed", "slow_rsmo", "slow_rins", "slow_none", "root_expand"):
        if _l3:
            s[k] = sum(r[k] for r in _l3)
    s["n10"] = sum(r["n10"] for r in xs)
    s["n1ms"] = sum(r["n1ms"] for r in xs)
    return s


def cell(pattern, need=None):
    out = [run_stats(f) for f in sorted(glob.glob(os.path.join(R, pattern)))]
    out = [o for o in out if o and (need is None or o["batches"] == need)]
    return out


def med(runs, k):
    return st.median(r[k] for r in runs) if runs else float("nan")


ARMS = [("base", "ALEX-OL"), ("ns2048", "\\quad 2048-entry nodes"), ("bg", "\\quad background only"),
        ("side", "\\quad side buffer only"), ("bgside", "\\quad side buffer + background"),
        ("bgside2", "\\quad side buffer + 2 background"), ("bgside4", "\\quad side buffer + 4 background")]
THREADS = [1, 2, 4, 8, 16]

# ---- 4M: slow inserts across thread counts, and wall time at 1 and 16 threads ----
print("== 4M, insert-only, median of runs")
m4 = {(a, T): cell(f"r6/m4/{a}_t{T}_s*_r*.csv", 30) for a, _ in ARMS for T in THREADS}
with open(f"{OUT}/tab_mech4m.tex", "w") as o:
    o.write("\\begin{tabular}{l" + "r" * len(THREADS) + "rr}\n\\toprule\n")
    o.write("& \\multicolumn{%d}{c}{inserts $>100\\,\\mu$s, by threads} & \\multicolumn{2}{c}{time (s)} \\\\\n" % len(THREADS))
    o.write("\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n" % (len(THREADS) + 1, len(THREADS) + 2, len(THREADS) + 3))
    o.write("Configuration & " + " & ".join(str(T) for T in THREADS) + " & 1 & 16 \\\\\n\\midrule\n")
    for a, lab in ARMS:
        vals = [f"{med(m4[(a, T)], 'slow'):.0f}" for T in THREADS]
        o.write(f"{lab} & " + " & ".join(vals) +
                f" & {med(m4[(a, 1)], 'wall'):.2f} & {med(m4[(a, 16)], 'wall'):.2f} \\\\\n")
        print(f"  {a:8s} n={[len(m4[(a, T)]) for T in THREADS]} slow={vals} "
              f"wall1={med(m4[(a, 1)], 'wall'):.3f} wall16={med(m4[(a, 16)], 'wall'):.3f}")
    o.write("\\bottomrule\n\\end{tabular}\n")

# ---- 4M mixed: lookup cost of each remedy ----
print("== 4M, 50/50 mix: read p50 / p99 (ns), slow inserts, slow reads")
for T in (1, 16):
    for a in ("base", "ns2048", "side", "bgside", "bgside4"):
        c = cell(f"r6/m4mix/{a}_t{T}_s*_r*.csv")
        if c:
            print(f"  T{T:<2} {a:8s} n={len(c)} rp50={med(c, 'rp50'):.0f} rp99={med(c, 'rp99'):.0f} "
                  f"slow={med(c, 'slow'):.0f} rslow={med(c, 'rslow'):.0f} wall={med(c, 'wall'):.3f}")

# ---- restart attribution ----
print("== restart attribution (lockstats runs)")
for pat in ("r6/m4ls/{a}_t{T}_s*.csv", "r6/b400ls/{a}_books_t{T}_s*.csv"):
    for T in (1, 16):
        for a in ("base", "side", "bgside", "bgside2", "bgside4", "bg"):
            c = cell(pat.format(a=a, T=T))
            if not c or "retry" not in c[0]:
                continue
            smo = sum(r["smo_ns"] for r in c) / max(1, sum(r["fg_smo"] + r["bg_smo"] for r in c))
            print(f"  {pat.split('/')[0]:11s} T{T:<2} {a:7s} n={len(c)} retry={med(c, 'retry'):.3g} "
                  f"on_smo={100 * sum(r['retry_smo'] for r in c) / max(1, sum(r['retry'] for r in c)):.1f}% "
                  f"fg_smo={med(c, 'fg_smo'):.0f} bg_smo={med(c, 'bg_smo'):.0f} "
                  f"mean_smo_us={smo / 1e3:.0f} max_smo_us={max(r['smo_max_ns'] for r in c) / 1e3:.0f} "
                  f"side_ins={med(c, 'side_ins'):.0f} side_read={med(c, 'side_read'):.0f} "
                  f"side_full={med(c, 'side_full'):.0f} splits={med(c, 'smo_split'):.0f}")
            if "slow_fgsmo" in c[0]:
                print("      slow by cause: " + " ".join(f"{k[5:]}={med(c, k):.0f}" for k in
                      ("slow_fgsmo", "slow_sealed", "slow_rsmo", "slow_rins", "slow_none")) +
                      f" root_expand={med(c, 'root_expand'):.0f}")

# ---- soft threshold ----
print("== soft threshold (bgside), slow inserts")
for T in (1, 16):
    line = []
    for sf in ("0.85", "0.9", "0.97"):
        c = cell(f"r6/m4soft/sf{sf}_t{T}_s*.csv")
        line.append(f"{sf}:{med(c, 'slow'):.0f}")
    line.append(f"0.9375:{med(m4[('bgside', T)], 'slow'):.0f}")
    print(f"  T{T}: " + "  ".join(line))

# ---- 400M books ----
print("== 400M books")
B4 = {}
ARM4 = ("base", "ns2048", "side", "bgside", "bgside2", "bgside4")
for T in (1, 16):
    for a in ARM4:
        B4[(a, T)] = cell(f"r6/b400/{a}_books_t{T}_s*.csv", 24)
        c = B4[(a, T)]
        if not c:
            continue
        print(f"  T{T:<2} {a:7s} n={len(c)} slow={med(c, 'slow'):.0f} "
              f"[{min(r['slow'] for r in c):.0f}-{max(r['slow'] for r in c):.0f}] "
              f"maxbatch={med(c, 'maxb'):.0f} wall={med(c, 'wall'):.1f}s mem={med(c, 'mem'):.2f}GB "
              f">10us={med(c, 'n10'):.0f} >1ms={med(c, 'n1ms'):.0f}")
LAB4 = {"base": "ALEX-OL", "ns2048": "\\quad 2048-entry nodes", "side": "\\quad side buffer only",
        "bgside": "\\quad side buffer + background", "bgside2": "\\quad side buffer + 2 background",
        "bgside4": "\\quad side buffer + 4 background"}
def f0(c, k, fmt):
    return (fmt.format(med(c, k)).replace(",", "{,}")) if c else "--"
with open(f"{OUT}/tab_mech400.tex", "w") as o:
    o.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
    o.write("& \\multicolumn{4}{c}{1 thread} & \\multicolumn{4}{c}{16 threads} \\\\\n")
    o.write("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n")
    o.write("Configuration & slow & worst & time & mem & slow & worst & time & mem \\\\\n\\midrule\n")
    for a in ARM4:
        p, q = B4[(a, 1)], B4[(a, 16)]
        o.write(f"{LAB4[a]} & {f0(p, 'slow', '{:,.0f}')} & {f0(p, 'maxb', '{:,.0f}')} & {f0(p, 'wall', '{:.0f}')} & "
                f"{f0(p, 'mem', '{:.2f}')} & {f0(q, 'slow', '{:,.0f}')} & {f0(q, 'maxb', '{:,.0f}')} & "
                f"{f0(q, 'wall', '{:.1f}')} & {f0(q, 'mem', '{:.2f}')} \\\\\n")
    o.write("\\bottomrule\n\\end{tabular}\n")

# ---- 400M osm and mixed ----
for pat, lab in (("r6/o400/{a}_osm_t{T}_s*.csv", "osm"), ("r6/x400/{a}_books_t{T}_s*.csv", "mix50"),
                 ("mix400/nb{a}_books_t{T}_s*.csv", "mix50 nodesize")):
    print(f"== 400M {lab}")
    arms = ("524288", "32768") if "nodesize" in lab else ("base", "side", "bgside", "bgside2", "bgside4", "ns2048")
    for T in (1, 16):
        for a in arms:
            c = cell(pat.format(a=a, T=T), 24)
            if c:
                print(f"  T{T:<2} {a:7s} n={len(c)} slow={med(c, 'slow'):.0f} "
                      f"[{min(r['slow'] for r in c):.0f}-{max(r['slow'] for r in c):.0f}] maxbatch={med(c, 'maxb'):.0f} "
                      f"wall={med(c, 'wall'):.1f}s rp50={med(c, 'rp50'):.0f} rp99={med(c, 'rp99'):.0f} "
                      f"rslow={med(c, 'rslow'):.0f} mem={med(c, 'mem'):.2f}")

# ---- 400M headline replication and threads past 16 ----
print("== 400M seeds (alexol, alexolstag)")
for a in ("alexol", "alexolstag"):
    for T in (1, 16):
        c = cell(f"seeds400/{a}_books_t{T}_s*.csv", 24)
        if c:
            print(f"  {a:10s} T{T:<2} n={len(c)} slow={[round(r['slow']) for r in c]}")
print("== 4M, 24 and 32 threads")
for a in ("alexol", "alexolstag", "nb32768"):
    for T in (24, 32):
        c = cell(f"thr/{a}_t{T}_s*_r*.csv", 30)
        if c:
            print(f"  {a:10s} T{T} n={len(c)} slow={med(c, 'slow'):.0f} wall={med(c, 'wall'):.3f}")
print("wrote", f"{OUT}/tab_mech4m.tex", f"{OUT}/tab_mech400.tex")
