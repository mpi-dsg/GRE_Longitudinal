#!/usr/bin/env python3
"""What a burst costs a workload (RQ2 "What a burst costs a workload", RQ4 "At scale").

Per run: median over predicted-burst batches and over quiet batches of batch time, inserts over
1 ms, and the slowest insert; cells are medians over the three seeds. 400M books, insert-only.
  python3 analysis/mk_burstcost.py
"""
import glob, os, statistics as st

R = os.environ.get("WAVE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw"))
BULK, BS = 100_000_000, 12_500_000


def predicted(keys):
    w, x = [], BULK * 8 / 7
    while x < keys[-1]:
        w.append(x); x *= 4 / 3
    return {i for i, k in enumerate(keys) for u in w if k - BS < u <= k}


for label, pat in (("ALEX-OL, 1 thread", "r6/b400/base_books_t1_s*.csv"),
                   ("ALEX-OL, 16 threads", "r14/b400/base_books_t16_s*.csv"),
                   ("modified, 1 background, 1 thread", "r6/b400/bgside_books_t1_s*.csv"),
                   ("modified, 4 background, 16 threads", "r14/b400/bgside4_books_t16_s*.csv")):
    runs = []
    for f in sorted(glob.glob(os.path.join(R, pat))):
        xs = [l.strip().split(",") for l in open(f) if l.startswith("x,") and l.count(",") == 22]
        wi = predicted([int(r[5]) for r in xs])
        qi = [i for i in range(len(xs)) if i not in wi]
        t = [int(r[21]) / 1e9 for r in xs]; n1ms = [int(r[13]) for r in xs]; mx = [int(r[10]) / 1e6 for r in xs]
        runs.append(dict(t_burst=st.median(t[i] for i in wi), t_quiet=st.median(t[i] for i in qi),
                         over1ms_burst=st.median(n1ms[i] for i in wi), over1ms_quiet=st.median(n1ms[i] for i in qi),
                         slowest_ms_burst=st.median(mx[i] for i in wi), slowest_ms_quiet=st.median(mx[i] for i in qi)))
    print(f"{label:36s}", {k: round(st.median(r[k] for r in runs), 2) for k in runs[0]})
