# wave-benchmark

Harness, measurements and analysis for synchronized restructuring in bulk-loaded updatable
indexes. Everything needed to check the numbers, re-run the experiments, or analyze the data
differently is in this directory.

```
bench.cpp btree_bulk.h build.sh run.sh patches/ shim/ shim_mkl/   the harness
runs/                 the scripts that produced the final run sets (mq6, mq7, mq8)
results/raw/          CSVs, one directory per experiment, each backing a claim
results/README.md     layout, column meanings, how to read them
analysis/             generators for the paper's tables and figure
analysis/summarize.py prints every headline number from the data
```

## What this measures

Bulk loading ALEX sets every data node to `kInitDensity_` (0.7). Nodes expand at
`kMaxDensity_` (0.8) and reset to `kMinDensity_` (0.6), so the whole population crosses its
threshold together: restructuring arrives in waves at key counts `N0 * 8/7`, then successive
multiples of `4/3`, where `N0` is the number of keys bulk loaded.

On 400M SOSD books keys those predicted batches carry a median of 8284 inserts slower than
100 us against 10 in the others (medians over three seeds), with 89% of all slow inserts in five of twenty-four batches.
LIPP-OL, SALI, a bulk-loaded B+tree and ART-OLC show nothing comparable on the same workload.

Randomizing the bulk-load density, which is the remedy the B-tree literature established,
decorrelates the cohort but does not reduce the number of slow operations, and under
concurrency increases it (147M insert restarts against 283M at 16 threads; zero at one thread,
in both arms). Shrinking data nodes from 32768 to 2048 entries removes the waves at 4M keys but
at 400M makes the run 37-38% slower and median lookups 36-58% slower.

Instrumenting the insert path shows why: about 97% of restarts meet a node whose lock is held by
a rebuild. `patches/alexol-sidebuf.patch` removes that blocking. A side buffer lets inserts and
lookups proceed while a node is rebuilt, and background threads perform rebuilds at a soft
threshold. Both are off by default (`side`, `bg`, `bgthreads=N`, `bgsoft=F`). On 400M books keys
(final binary, medians of three seeds) the design takes inserts slower than 100 us from 57,450 to
249 at one thread (`results/raw/r6/b400`) and, with four background threads, from 70,928 to 4,029
at sixteen at an unchanged run time (`results/raw/r9/b400`). Median lookup latency under a
balanced mix rises by 6-10%. Changing the three density constants moves the bursts exactly to
positions recorded before the runs (`results/raw/r11/consts`, `results/raw/predictions/`), and a
variant that buffers every insert at all times takes about 17 times as long as the baseline
(`results/raw/r8`).

`alexol-sidebuf.patch` also fixes defects in released ALEX-OL, in every configuration, the
baseline included: root expansion left children's depths stale (later rebuilds overwrote sibling
subtrees), mutated the live root in place and freed its child array under readers, and took no
lock on the outermost leaf; a rebuild could copy a stale depth before locking its parent; and a
lookup could miss a key on a floating-point split boundary (lookups that miss now check the
adjacent leaves). An unflagged run is therefore ALEX-OL with these fixes and no design changes.

`patches/alexol-sidealways.patch` (optional, see `patches/README-sidealways.md`) adds an ablation
with a permanent per-node buffer, as XIndex keeps: every insert goes to the node's buffer and a
full buffer is merged into the node. It isolates the cost of buffering all the time against
buffering only during a rebuild.

## Check the numbers without running anything

```sh
python3 analysis/summarize.py
```

No arguments, no dependencies beyond Python 3. Reads `results/raw` and prints the wave
concentrations, the index comparison, the remedy result, the node-size sweep and the retry
counters.

## Build the harness

```sh
sh build.sh                                  # from inside a GRE checkout
GRE=/path/to/GRE_Longitudinal sh build.sh    # or point it elsewhere
XF=1 sh build.sh                             # also XIndex and FINEdex
XF=1 SIDEALWAYS=1 sh build.sh                # also the permanent-buffer ablation
```

The measured binaries were built with `XF=1` (runs in `r6`, `r7`) and `XF=1 SIDEALWAYS=1`
(runs in `r8`). With its flag off, the ablation patch leaves every other arm unchanged.
XIndex and FINEdex need MKL's `LAPACKE_dgels`; `shim_mkl/` supplies a least-squares stand-in so
no MKL installation is needed.

Requires g++ with OpenMP, oneTBB headers and `libtbb`. It copies the competitor sources into
`_c/`, patches the copy, and compiles; your GRE checkout is never modified. The build ends with
a smoke test that must print `VERIFY ... 100.0000%`.

Only `alexol` is vendored in GRE's tree. `lippol`, `sali`, `btreeolc` and `artsync` are
submodules, so a fresh clone has the wrappers but no implementations; `build.sh` checks and
prints the exact `git submodule update` command if any are missing.

oneTBB removed `tbb/mutex.h` and `tbb/reader_writer_lock.h` in 2021. `shim/tbb/` supplies
three-line replacements. Both are named only by typedefs these indexes never instantiate.

## Re-run the experiments

```sh
DATA=/path/to/sosd sh run.sh            # all tiers
DATA=/path/to/sosd TIER=1 sh run.sh     # one tier
python3 analysis/analyze.py results 100000000 400000000
```

`DATA` holds the SOSD binaries (`books_800M_uint64`, `osm_cellids_800M_uint64`, from
https://github.com/learnedsystems/SOSD). Completed runs are skipped, so interrupting and
restarting is safe.

| Tier | Time | What it is |
|---|---|---|
| 1 | ~90 min | The headline comparison: six indexes, 400M keys, 1 and 16 threads |
| 2 | ~60 min | Balanced read/write. Reads are timed separately (`r_*` columns) |
| 3 | ~90 min | OSM, where heterogeneous key density weakens the effect |
| 4 | — | Additional seeds |
| 5 | ~2 h | The design: side buffer and background expansion against ALEX-OL and 2048-entry nodes |

## Running it directly

```
./bench <index> <bulk> <total> <batch> <seed> <threads> <tag> [options...]
```

`<index>` is one of `alexol`, `lippol`, `sali`, `btreebulk`, `artolc`, `alexsized`, and with
`XF=1` also `xindex`, `finedex`.
Options: `stagger` (randomized bulk-load density), `lowonly`, `delta=X`, `nodebytes=N`
(with `alexsized`), `bfill=F` and `bspread=S` (with `btreebulk`), `readpct=N`,
`data=<sosd file>`, `lockstats` (restart and rebuild counters, printed as `LOCK2` rows),
`side` (side buffer), `bg` (background expansion), `bgthreads=N`, `bgsoft=F` (soft threshold as
a fraction of the hard one, default 0.9375), `sidealways` (permanent buffer, `SIDEALWAYS=1`
builds only), `recent` (with `readpct`, lookups target keys the same thread just inserted, so
they reach nodes under rebuild and their buffers; every such lookup must hit or the run fails),
`order=shuffle|zipf|sorted` (insert order after the bulk load; `zipf` concentrates inserts in
1024 key ranges with Zipfian weights, `ztheta=` sets the skew, default 0.99). Each batch also
prints a `CPU` row: CPU seconds of the whole process over the timed interval, background threads
included.

## Two things that will bite you

**Report counts, not percentiles.** A wave produces roughly one slow insert per expanding node,
and whether a percentile sees them depends on how that count compares with its rank cutoff,
which depends on batch size. The same 400M runs report a wave-to-quiet ratio of 1.8x measured
by p99.9 over 12.5M-operation batches and 828x measured by counting inserts over 100 us.
Percentiles also flatter any change that merely spreads the same slow operations more thinly:
that produced a 237-fold "improvement" that does not exist. Use `n_gt10us`, `n_gt100us`,
`n_gt1ms`. `results/README.md` explains this at length and it is the single most important
thing to understand before interpreting any of these files.

**The driver uses OpenMP, not `std::thread`.** LIPP-OL and SALI index their per-thread node
pools with `omp_get_thread_num()`, which returns 0 for every raw thread, so they share one
unsynchronized pool. LIPP-OL crashes an internal assertion. SALI does not crash: it races
silently and returns clean, flat numbers. If you adapt this driver, keep the parallel region.

Every run ends with a 100k-probe recall check and an exhaustive check of every inserted key
and payload (`FULLVERIFY ... 0/N missing or wrong`). A failing run exits with status 2, and the
run scripts keep its output only as `.failN`. FINEdex loses one key in some 400M runs; XIndex
makes no progress on some seeds (kept as `.fail124`, a timeout). With a 100k-key bulk load at 32 threads, released ALEX-OL itself crashes or loses
keys in some runs; the exhaustive check is what shows it.

## Analysis scripts

| Script | Output |
|---|---|
| `summarize.py` | every headline number, plain text, no arguments |
| `mktables.py` | thread-sweep tables |
| `mk400b.py` | the 400M comparison table |
| `mk_remedy.py` | the randomized-density table, from counts in `r13/sweep` |
| `mk_r13.py` | every number from the reruns in `r13`: 4M burst ratios, restarts, variants, pinning, B+-tree fill, node sizes, 400M randomized density |
| `mk_account.py` | the account table (memory, run time, latency, key check per index) |
| `mkfig_mech4m.py` | the 4M design figure |
| `mkfig.py` | the 400M trace figure |
| `mk_mech.py` | the design tables; prints every number the design evaluation quotes |
| `mkfig_mech.py` | the design figure |
| `mk_cmp.py` | the comparison with XIndex and FINEdex |
| `mk_panel.py` | CPU time, lookups of just-inserted keys, skewed and sorted inserts, the permanent-buffer ablation |
| `analyze.py` | per-run analysis for arbitrary result directories |

Set `WAVE_RESULTS` to point them at a different results root. `mk_remedy.py` is separate from
`mktables.py` because the latter once built that table from percentiles and reported an
improvement that had not occurred.

## Measurement hygiene

Everything here was measured on otherwise idle dual-socket Xeon Gold 6134M servers (16 physical
cores, 755 GB, oneTBB); every comparison uses runs from one server. Do not measure this on a shared or loaded machine: on a laptop our
quiet-batch p99.9 ranged 44-1347 us across identical runs, which is larger than most of the
effects reported, and it concealed rather than created them.
