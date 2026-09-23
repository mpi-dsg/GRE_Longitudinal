# wave-benchmark

Harness, measurements and analysis for synchronized restructuring in bulk-loaded updatable
indexes. Everything needed to check the numbers, re-run the experiments, or analyze the data
differently is in this directory.

```
bench.cpp btree_bulk.h build.sh run.sh patches/ shim/   the harness
results/raw/          903 CSVs, ten sets, each backing a claim
results/README.md     layout, column meanings, how to read them
analysis/             generators for the paper's tables and figure
analysis/summarize.py prints every headline number from the data
```

## What this measures

Bulk loading ALEX sets every data node to `kInitDensity_` (0.7). Nodes expand at
`kMaxDensity_` (0.8) and reset to `kMinDensity_` (0.6), so the whole population crosses its
threshold together: restructuring arrives in waves at key counts `N0 * 8/7`, then successive
multiples of `4/3`, where `N0` is the number of keys bulk loaded.

On 400M SOSD books keys those predicted batches carry a median of 8334 inserts slower than
100 us against 9 in the others, with 89% of all slow inserts in five of twenty-four batches.
LIPP-OL, SALI, a bulk-loaded B+tree and ART-OLC show nothing comparable on the same workload.

Randomizing the bulk-load density, which is the remedy the B-tree literature established,
decorrelates the cohort but does not reduce the number of slow operations, and under
concurrency increases it: inserts restart when they meet a modification in progress, and
spreading expansions across the workload means more of them do (147M restarts against 283M at
16 threads; zero at one thread, in both arms). Reducing `max_data_node_size` does reduce the
cost: 32768 to 2048 entries takes slow inserts from 649 to 10 at one thread and 4563 to 998 at
sixteen, for 10-14% on median lookup latency.

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
```

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

## Running it directly

```
./bench <index> <bulk> <total> <batch> <seed> <threads> <tag> [options...]
```

`<index>` is one of `alexol`, `lippol`, `sali`, `btreebulk`, `artolc`, `alexsized`.
Options: `stagger` (randomized bulk-load density), `lowonly`, `delta=X`, `nodebytes=N`
(with `alexsized`), `bfill=F` and `bspread=S` (with `btreebulk`), `readpct=N`,
`data=<sosd file>`, `lockstats` (needs a `-DLOCK_STATS` build).

## Two things that will bite you

**Report counts, not percentiles.** A wave produces roughly one slow insert per expanding node,
and whether a percentile sees them depends on how that count compares with its rank cutoff,
which depends on batch size. The same 400M run reports a wave-to-quiet ratio of 1.8x measured
by p99.9 over 12.5M-operation batches and 926x measured by counting inserts over 100 us.
Percentiles also flatter any change that merely spreads the same slow operations more thinly:
that produced a 240-fold "improvement" that does not exist. Use `n_gt10us`, `n_gt100us`,
`n_gt1ms`. `results/README.md` explains this at length and it is the single most important
thing to understand before interpreting any of these files.

**The driver uses OpenMP, not `std::thread`.** LIPP-OL and SALI index their per-thread node
pools with `omp_get_thread_num()`, which returns 0 for every raw thread, so they share one
unsynchronized pool. LIPP-OL crashes an internal assertion. SALI does not crash: it races
silently and returns clean, flat numbers. If you adapt this driver, keep the parallel region.

Every run ends with a 100k-probe recall check. Treat a missing `VERIFY ... 100.0000%` as a
failed run.

## Analysis scripts

| Script | Output |
|---|---|
| `summarize.py` | every headline number, plain text, no arguments |
| `mktables.py` | thread-sweep tables |
| `mk400b.py` | the 400M comparison table |
| `mk_remedy.py` | the remedy table, from counts |
| `mkfig.py` | the 400M trace figure |
| `analyze.py` | per-run analysis for arbitrary result directories |

Set `WAVE_RESULTS` to point them at a different results root. `mk_remedy.py` is separate from
`mktables.py` because the latter once built that table from percentiles and reported an
improvement that had not occurred.

## Measurement hygiene

Everything here was measured on an otherwise idle dual-socket Xeon Gold 6134M (16 physical
cores, 755 GB, oneTBB). Do not measure this on a shared or loaded machine: on a laptop our
quiet-batch p99.9 ranged 44-1347 us across identical runs, which is larger than most of the
effects reported, and it concealed rather than created them.
