#!/usr/bin/env python3
"""RQ2 at scale: concentration and predicted-batch share for every index at 400M keys, one thread.

  python3 analysis/mk_s400.py [dir ...]

Concentration is the median high-latency count (inserts over 100 us) of the batches that contain a
predicted ALEX burst position over the median of the other batches; share is the fraction of all
high-latency inserts in those batches. Cells are medians over seeds. Reads the final driver's
23-column rows and the earlier driver's 17-column rows.
"""
import glob, os, statistics as st, sys

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
BULK, BS = 100_000_000, 12_500_000


def rows(f):
    out = []
    for l in open(f):
        p = l.strip().split(",")
        if p[0] == "x" and len(p) in (17, 23):
            out.append((int(p[5]), int(p[12])))
    return out


def run(f):
    x = rows(f)
    if len(x) < 24:
        return None
    keys, g = [k for k, _ in x], [n for _, n in x]
    w, v = [], BULK * 8 / 7
    while v < keys[-1]:
        w.append(v); v *= 4 / 3
    wi = {i for i, k in enumerate(keys) for u in w if k - BS < u <= k}
    wave = st.median(g[i] for i in wi)
    quiet = st.median(g[i] for i in range(len(g)) if i not in wi)
    return dict(conc=wave / max(quiet, 1), share=100 * sum(g[i] for i in wi) / max(sum(g), 1),
                total=sum(g), wave=wave, quiet=quiet)


def show(label, pat):
    v = [r for r in (run(f) for f in sorted(glob.glob(os.path.join(R, pat)))) if r]
    if not v:
        print(f"  {label:28s} (no runs)"); return
    m = {k: st.median(r[k] for r in v) for k in v[0]}
    print(f"  {label:28s} n={len(v)} conc={m['conc']:.2f} "
          f"(range {min(r['conc'] for r in v):.2f}-{max(r['conc'] for r in v):.2f}) "
          f"share={m['share']:.0f}% total={m['total']:.0f} wave={m['wave']:.0f} quiet={m['quiet']:.0f}")


for d in sys.argv[1:] or ["r14/s400"]:
    print(f"== {d}")
    for idx in ("lippol", "sali", "btreebulk", "artolc"):
        show(f"{idx} books t1", f"{d}/{idx}_books_t1_s*.csv")
    show("sali osm t1", f"{d}/sali_osm_t1_s*.csv")
    show("sali osm t1 (earlier naming)", f"{d}/sali_t1_s*.csv")
