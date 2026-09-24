#!/usr/bin/env python3
"""Thread-sweep analysis over five indexes.

Two metrics, deliberately separated:

  burstiness  = worst-batch p999 / median-batch p999, per run.
                Schedule-agnostic. Makes no assumption about where restructuring
                happens, so it is fair to indexes whose trigger differs from ALEX's
                (LIPP's ladder, SALI's Bernoulli draw) and to those with no
                restructuring schedule at all (the classical controls).

  wave/quiet  = p999 at ALEX's predicted wave batches vs everywhere else.
                Only meaningful for ALEX; the positions come from the density
                constants, never from the latency data.
"""
import glob, os, statistics as st, sys, math

COL = "tag index seed threads batch keys p50 p99 p999 max batch_ns mem".split()
BULK, TOTAL = 1_000_000, 4_000_000

def load(f):
    out = []
    for l in open(f):
        p = l.strip().split(',')
        if len(p) == 12 and p[0] != 'tag':
            out.append(dict(zip(COL, p)))
    return out

def predicted(idx):
    """Wave positions from each index's own source constants. Never from the data.

    ALEX-OL   kInitDensity_ 0.7, kMaxDensity_ 0.8, kMinDensity_ 0.6
              (alex_nodes.h:369-374) -> first expansion at x8/7, then x4/3.
    LIPP-OL   node->size >= node->build_size * 2 (lipp.h:1167; upstream LIPP uses *4)
              -> x2 ladder.
    SALI      the same counters are commented out (sali.h:214,286,992,1327) and replaced
              by std::bernoulli_distribution draws (sali.h:997,1141-1142,1179).
              Predicts NO waves. This is the natural experiment.
    classical no restructuring schedule of this kind.
    """
    w = []
    if idx.startswith("alexol"):
        x = BULK * 8 / 7
        while x < TOTAL: w.append(x); x *= 4 / 3
    elif idx == "lippol":
        x = BULK * 2
        while x < TOTAL: w.append(x); x *= 2
    return w

def wave_idx(keys, idx):
    w = predicted(idx)
    if not w: return set()
    bs = keys[1] - keys[0]
    return {min(range(len(keys)), key=lambda i: abs(keys[i] - bs / 2 - v)) for v in w}

def mwu(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3: return float('nan')
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    r, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        for k in range(i, j + 1): r[k] = (i + j) / 2 + 1
        i = j + 1
    R1 = sum(r[k] for k, (v, g) in enumerate(allv) if g == 0)
    u1 = R1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return math.erfc(abs((u - mu) / sd) / math.sqrt(2))

D = sys.argv[1] if len(sys.argv) > 1 else "res"
INDEXES = ["alexol", "alexolstag", "lippol", "sali", "btreeolc", "artolc"]
LABEL = {"alexol": "ALEX-OL", "alexolstag": "ALEX-OL +stagger", "lippol": "LIPP-OL",
         "sali": "SALI", "btreeolc": "BTree-OLC", "artolc": "ART-OLC"}
KIND = {"alexol": "learned", "alexolstag": "learned", "lippol": "learned",
        "sali": "learned", "btreeolc": "classical", "artolc": "classical"}
THREADS = [1, 2, 4, 8, 16]

burst = {}
for idx in INDEXES:
    for T in THREADS:
        vals, med, worst, robust = [], [], [], []
        for f in sorted(glob.glob(f"{D}/{idx}_t{T}_s*_r*.csv")):
            x = load(f)
            if len(x) < 25: continue
            p = sorted(float(r['p999']) / 1000 for r in x)
            m = st.median(p)
            vals.append(p[-1] / m); med.append(m); worst.append(p[-1])
            # p90 of batches over median of batches: same shape, immune to one outlier batch
            robust.append(p[int(0.9 * (len(p) - 1))] / m)
        if vals: burst[(idx, T)] = (vals, med, worst, robust)

print("BURSTINESS  = worst-batch p999 / median-batch p999   (median over runs [min-max], n runs)")
print("Schedule-agnostic: assumes nothing about where restructuring happens.\n")
hdr = f"{'index':<18}{'kind':<10}" + "".join(f"{'t'+str(T):>17}" for T in THREADS)
print(hdr); print("-" * len(hdr))
for idx in INDEXES:
    if not any((idx, T) in burst for T in THREADS): continue
    row = f"{LABEL[idx]:<18}{KIND[idx]:<10}"
    for T in THREADS:
        if (idx, T) in burst:
            v = burst[(idx, T)][0]
            row += f"{st.median(v):8.1f}x[{min(v):.0f}-{max(v):.0f}]".rjust(17)
        else: row += f"{'-':>17}"
    print(row)

print("\nWORST-BATCH p999 (us, absolute) — burstiness is scale-free, this is not\n")
print(hdr); print("-" * len(hdr))
for idx in INDEXES:
    if not any((idx, T) in burst for T in THREADS): continue
    row = f"{LABEL[idx]:<18}{KIND[idx]:<10}"
    for T in THREADS:
        row += (f"{st.median(burst[(idx,T)][2]):16.1f} " if (idx, T) in burst else f"{'-':>17}")
    print(row)

print("\nROBUST BURSTINESS = p90-of-batches / median-of-batches (immune to one outlier batch)\n")
print(hdr); print("-" * len(hdr))
for idx in INDEXES:
    if not any((idx, T) in burst for T in THREADS): continue
    row = f"{LABEL[idx]:<18}{KIND[idx]:<10}"
    for T in THREADS:
        row += (f"{st.median(burst[(idx,T)][3]):15.1f}x " if (idx, T) in burst else f"{'-':>17}")
    print(row)

print("\nMEDIAN-BATCH p999 (us) — the ordinary floor, not the tail\n")
print(hdr); print("-" * len(hdr))
for idx in INDEXES:
    if not any((idx, T) in burst for T in THREADS): continue
    row = f"{LABEL[idx]:<18}{KIND[idx]:<10}"
    for T in THREADS:
        row += (f"{st.median(burst[(idx,T)][1]):16.2f} " if (idx, T) in burst else f"{'-':>17}")
    print(row)

print("\nAMPLIFICATION 1 -> 16 threads (burstiness ratio, and significance)\n")
for idx in INDEXES:
    if (idx, 1) in burst and (idx, 16) in burst:
        a, b = burst[(idx, 1)][0], burst[(idx, 16)][0]
        p = mwu(a, b)
        print(f"  {LABEL[idx]:<18}{KIND[idx]:<10} {st.median(a):7.1f}x -> {st.median(b):7.1f}x "
              f"({st.median(b)/st.median(a):5.2f}x)   p={p:.4f}")

print("\nWAVE POSITIONS — each index's own predicted schedule, from source constants\n")
for idx in ("alexol", "alexolstag", "lippol"):
    pw = predicted(idx)
    print(f"  {LABEL[idx]}: predicted at " + ", ".join(f"{v/1e6:.2f}M" for v in pw))
    for T in THREADS:
        W, Q = [], []
        for f in sorted(glob.glob(f"{D}/{idx}_t{T}_s*_r*.csv")):
            x = load(f)
            if len(x) < 25: continue
            keys = [int(r['keys']) for r in x]; wi = wave_idx(keys, idx)
            if not wi: continue
            W.append(st.median([float(x[i]['p999']) / 1000 for i in wi]))
            Q.append(st.median([float(x[i]['p999']) / 1000 for i in range(len(x)) if i not in wi]))
        if W:
            print(f"  {LABEL[idx]:<18} t{T:<3} wave {st.median(W):9.1f}us  quiet {st.median(Q):7.2f}us  "
                  f"concentration {st.median(W)/st.median(Q):7.1f}x")

print("\nREMEDY: baseline vs staggered bulk-load density, per thread count")
print("  (burstiness median over runs, Mann-Whitney on the run distributions)\n")
print(f"  {'threads':<9}{'baseline':>14}{'staggered':>14}{'change':>10}{'p':>10}"
      f"{'worst base':>13}{'worst stag':>13}")
for T in THREADS:
    if ("alexol", T) not in burst or ("alexolstag", T) not in burst: continue
    a, b = burst[("alexol", T)], burst[("alexolstag", T)]
    ma, mb = st.median(a[0]), st.median(b[0])
    print(f"  t{T:<8}{ma:13.1f}x{mb:13.1f}x{ma/mb:9.2f}x{mwu(a[0], b[0]):10.4f}"
          f"{st.median(a[2]):12.1f}u{st.median(b[2]):12.1f}u")

print("\nTHREAD SCALING of burstiness, per index (is the amplification monotone?)\n")
for idx in INDEXES:
    row = [(T, st.median(burst[(idx, T)][0])) for T in THREADS if (idx, T) in burst]
    if len(row) < 2: continue
    mono = all(row[i][1] <= row[i + 1][1] * 1.15 for i in range(len(row) - 1))
    trend = " ".join(f"{v:.1f}" for _, v in row)
    print(f"  {LABEL[idx]:<18}{KIND[idx]:<10} {trend:<45} "
          f"{'monotone' if mono else 'not monotone'}")
