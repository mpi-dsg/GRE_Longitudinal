#!/usr/bin/env python3
"""Analysis for the wave benchmark. Reads the CSVs, prints the headline table.

  python3 analyze.py <results-dir> <bulk> <total>

Wave positions come from ALEX's density constants and the bulk-load size, never from the
latency data: the first expansion follows an x8/7 growth in key count, each later one x4/3.
"""
import glob, os, statistics as st, sys

C = ("tag index seed threads batch keys p50 p99 p999 p9999 max n10 n100 n1ms nops "
     "r_p50 r_p99 r_p999 r_max r_gt100 r_ops batch_ns mem").split()

def L(f):
    out = []
    for l in open(f):
        p = l.strip().split(',')
        if len(p) == len(C) and p[0] != 'tag':
            out.append(dict(zip(C, p)))
    return out

def wave_idx(keys, bulk, total):
    w, x = [], bulk * 8 / 7
    while x < total:
        w.append(x); x *= 4 / 3
    bs = keys[1] - keys[0]
    return {min(range(len(keys)), key=lambda i: abs(keys[i] - bs / 2 - v)) for v in w}, w

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results"
    bulk = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000_000
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000_000
    files = sorted(glob.glob(f"{d}/*.csv"))
    if not files:
        print(f"no CSVs in {d}"); return
    print(f"bulk {bulk/1e6:g}M -> {total/1e6:g}M\n")
    print(f"{'run':<34}{'wave':>8}{'quiet':>8}{'ratio':>8}{'total':>9}{'in-wave':>9}"
          f"{'rd>100us':>10}{'mem GB':>9}")
    print("-" * 95)
    for f in files:
        x = L(f)
        if len(x) < 5:
            print(f"{os.path.basename(f)[:33]:<34}  (incomplete: {len(x)} batches)"); continue
        keys = [int(r['keys']) for r in x]
        wi, _ = wave_idx(keys, bulk, total)
        g = [int(r['n100']) for r in x]
        tot = sum(g)
        wv = st.median([g[i] for i in wi])
        qt = st.median([g[i] for i in range(len(x)) if i not in wi])
        rd = sum(int(r['r_gt100']) for r in x)
        mem = float(x[-1]['mem']) / 1e9
        print(f"{os.path.basename(f)[:-4][:33]:<34}{wv:>8.0f}{qt:>8.0f}{wv/max(qt,1):>7.0f}x"
              f"{tot:>9}{100*sum(g[i] for i in wi)/max(tot,1):>8.0f}%{rd:>10}"
              f"{(f'{mem:.1f}' if mem > 0 else '--'):>9}")
    print("\nwave/quiet are median counts per batch of inserts over 100us.")
    print("Uniform distribution puts (number of wave batches)/(total batches) of slow inserts")
    print("in wave batches; compare 'in-wave' against that, not against 0.")

main()
