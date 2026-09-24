# sidealways: permanent delta buffer (XIndex-style ablation)

`alexol-sidealways.patch` applies on top of the chain `alexol-stagger.patch` (-p0),
`alexol-contention-stats.patch` (-p1), and `alexol-sidebuf.patch` (-p0):

    patch -p0 -d alexol/src -i alexol-sidealways.patch

The driver needs one line next to the `side` flag. This line is in
volta07:/local/bindsch/wave/bgx8/bench.cpp, which is the bgx7 driver plus this line. The local
`testbed/scale/bench.cpp` is older than the bgx7 driver and does not have it.

    else if (a == "sidealways") { alexol::sidebuf::enabled() = true; alexol::sidebuf::always() = true; }

## What it does

The arm isolates one difference from the side buffer. In `side`, a buffer exists only while a
node is rebuilt. In `sidealways`, every data node has a buffer all the time, as XIndex's
per-group delta buffer does.

- **Buffer.** It is the existing `SideBuf` (kCap 1024, spinlock, `sealed`). A node creates it on
  the first insert and publishes it with a CAS from null, so each node has exactly one. It is
  freed with the node through epoch reclamation.
- **Insert.** A new key is appended to the buffer without taking the node lock. The array of a
  live node is never written in place in this mode. The duplicate check reads the array, then
  takes the spinlock and checks the seal, the same order as the side-buffer insert path.
- **Lookup.** A lookup searches the array first and the buffer only on a miss (XIndex read
  path). An unlocked node is validated by its version. A locked node is read as in `side`.
- **Merge.** When the buffer is full or sealed, the inserting thread takes the node lock
  (`AlexDataNode::insert` returns the new code 7). It then runs `run_smo` with the choice ALEX's
  insert path would make if the buffered keys were in the array. It checks catastrophic cost
  first. At the expansion threshold, it applies the usual split, retrain, or expand choice.
  Below the threshold, it copies the node at its current capacity (code 7 in `run_smo_body`).
  Expansions are sized for array plus buffer, so the density after draining is 0.6. The buffer
  is drained into the replacement, which is published and retired through the existing paths.
  The thread then retries its insert against the replacement's empty buffer.
- **Other rebuilds.** Foreground, background (`bg`), and split rebuilds use the permanent buffer
  as their side buffer (`open_side` returns it). With `bg`, a node is queued when array plus
  buffer reaches the soft threshold. `bg_due` and `bg_smo_code` count buffered keys.
- **Root expansion.** This is the only in-place change to a live node's array. It seals the
  outermost node's buffer and sets a new `excl` flag before modifying anything. Readers and
  inserters that see `excl` restart. Buffered keys count toward the new key domain. Buffered
  keys that the new root routes to a new node move into that node before the root is
  published. The rest stay in the sealed buffer, and the next insert into the node merges it.
- **Memory.** `Alex::data_size` adds `sizeof(SideBuf)` (about 16 KiB) for every allocated buffer.

With the flag off, every new branch is skipped. At one thread, default and `side` runs report
the same memory for every batch as the unpatched binary.

Not supported in this mode: `update`, `erase`, and range scans ignore the buffer. The bench
does not use them.

## Correctness tests (volta07, 2026-09-24)

All runs passed FULLVERIFY with 0 missing or wrong. No run with `recent` reads reported READ_MISS.

- **Sanity, 4M keys.** Default, `side` and `bg side bgthreads=4`, at t1 and t16. At t1, default
  and `side` report the same memory and key counts for all 30 batches as the unpatched binary.
- **4M keys.** Seeds 1866, 5 and 72; threads 1, 4, 16 and 32. Arms: `sidealways` and
  `sidealways bg bgthreads=4`, each insert-only and with `readpct=50 recent` (96 runs).
- **Stress.** Bulk load 100k, 8M keys, t32, 8 seeds, the same four variants (32 runs). Root
  expansion runs thousands of times per run here.
- **Other orders and seeds.** `order=sorted` and `order=zipf` at t16 and t32, 3 seeds, plus
  8 more stress seeds with recent reads (48 runs).
- **AddressSanitizer.** Six runs at t32. They need `ASAN_OPTIONS=alloc_dealloc_mismatch=0`,
  because upstream ALEX-OL frees `posix_memalign`'d nodes with `operator delete`. No memory
  errors. One run printed a LeakSanitizer report at exit. It covers the live index, which the
  bench never frees: the reported bytes equal that run's final `mem_bytes`.
- **400M books.** 100M bulk load, t16, `sidealways bg bgthreads=4`: 0 of 400M keys missing.
