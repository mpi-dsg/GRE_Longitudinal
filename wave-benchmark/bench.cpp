// Multi-index wave/concurrency benchmark.
//
// One driver, five indexes, through GRE's own indexInterface so the code path matches
// the harness that produced the existing 800M data.
//
// Per batch it reports insert-latency percentiles across all threads, wall time, and the
// index's self-reported memory. Wave batches are identified from ALEX's density constants,
// never from the latency data.
#include <cstdint>
#include <iostream>
#include "indexInterface.h"
#include "alexol/alex.h"
#include "lippol/lippol.h"
#include "sali/sali.h"
#include "btreeolc/btreeolc.h"
#include "artsync/artolc.h"
#include "btree_bulk.h"

// BTree-OLC with a real bulk load. GRE's wrapper builds the tree by ordinary inserts,
// which never creates the uniform page cohort that waves of misery concerns, so it cannot
// be used to test the randomized-fill remedy.
// ALEX-OL with a settable data-node size. GRE's wrapper hardcodes 1<<19 bytes, which is
// 32768 entries per data node against 31 in a 512-byte B+tree page -- a factor of ~1000.
// If node size is what makes a synchronized expansion cohort catastrophic rather than
// invisible, shrinking it should collapse the burstiness toward the B+tree's.
inline int &alex_node_bytes() { static int b = 1 << 19; return b; }

template <class K_, class P_>
class AlexSizedInterface : public indexInterface<K_, P_> {
 public:
  void init(Param * = nullptr) override {}
  void bulk_load(std::pair<K_, P_> *kv, size_t n, Param * = nullptr) override {
    idx.set_max_model_node_size(1 << 24);
    idx.set_max_data_node_size(alex_node_bytes());
    idx.bulk_load(kv, (int)n);
  }
  bool get(K_ k, P_ &v, Param * = nullptr) override { return idx.get_payload(k, &v); }
  bool put(K_ k, P_ v, Param * = nullptr) override { return idx.insert(k, v); }
  bool update(K_ k, P_ v, Param * = nullptr) override { return idx.update(k, v); }
  bool remove(K_ k, Param * = nullptr) override { return idx.erase_one(k) > 0; }
  size_t scan(K_, size_t, std::pair<K_, P_> *, Param * = nullptr) override { return 0; }
  long long memory_consumption() override {
    return idx.model_size() + idx.data_size();
  }
 private:
  alexol::Alex<K_, P_> idx;
};

template <class K_, class P_>
class BTreeBulkInterface : public indexInterface<K_, P_> {
 public:
  void init(Param * = nullptr) override {}
  void bulk_load(std::pair<K_, P_> *kv, size_t n, Param * = nullptr) override {
    btree_bulk::load(idx, kv, n, btree_bulk::seed());
  }
  bool get(K_ k, P_ &v, Param * = nullptr) override { return idx.lookup(k, v); }
  bool put(K_ k, P_ v, Param * = nullptr) override { idx.insert(k, v); return true; }
  bool update(K_ k, P_ v, Param * = nullptr) override { idx.insert(k, v); return true; }
  bool remove(K_, Param * = nullptr) override { return false; }
  size_t scan(K_ lo, size_t n, std::pair<K_, P_> *r, Param * = nullptr) override {
    return idx.scan(lo, n, r);
  }
  long long memory_consumption() override { return 0; }
 private:
  btreeolc::BTree<K_, P_> idx;
};

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <omp.h>
#include <sys/resource.h>
#include <unistd.h>
#if defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm>
#define PSORT __gnu_parallel::sort
#else
#define PSORT std::sort
#endif
#include <vector>

#ifdef WITH_XF
// XIndex and FINEdex, as vendored in GRE. Built only with -DWITH_XF, since they need the MKL
// stand-in in shim_mkl/ and are compared only in Section 6.
#include "xindex/xindex.h"
#include "finedex/finedex.h"
#endif

using K = uint64_t;
using P = uint64_t;
using Clock = std::chrono::steady_clock;
using Index = indexInterface<K, P>;

static Index *make(const std::string &n) {
  if (n == "alexol")   return new alexolInterface<K, P>;
  if (n == "lippol")   return new LIPPOLInterface<K, P>;
  if (n == "sali")     return new SALIInterface<K, P>;
  if (n == "btreeolc") return new BTreeOLCInterface<K, P>;
  if (n == "artolc")   return new ARTOLCInterface<K, P>;
  if (n == "btreebulk") return new BTreeBulkInterface<K, P>;
  if (n == "alexsized") return new AlexSizedInterface<K, P>;
#ifdef WITH_XF
  if (n == "xindex")   return new xindexInterface<K, P>;
  if (n == "finedex")  return new finedexInterface<K, P>;
#endif
  return nullptr;
}

// SOSD format: uint64 count, then that many uint64 keys.
static std::vector<K> load_sosd(const std::string &path, size_t want) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  uint64_t n = 0;
  f.read(reinterpret_cast<char *>(&n), 8);
  if (want && want < n) n = want;
  std::vector<K> k(n);
  f.read(reinterpret_cast<char *>(k.data()), (std::streamsize)(n * 8));
  return k;
}

int main(int argc, char **argv) {
  if (argc < 8) {
    fprintf(stderr,
      "usage: %s <index> <bulk> <total> <batch> <seed> <threads> <tag>"
      " [stagger] [data=<sosd file>] [readpct=<0-100>]\n", argv[0]);
    return 1;
  }
  std::string name = argv[1];
  size_t bulk = strtoull(argv[2], nullptr, 10);
  size_t total = strtoull(argv[3], nullptr, 10);
  size_t batch = strtoull(argv[4], nullptr, 10);
  unsigned seed = (unsigned)atoi(argv[5]);
  int T = atoi(argv[6]);
  const char *tag = argv[7];

  std::string data;
  int readpct = 0;
  bool destroy = false;  // delete the index at the end, with background threads running
  bool bulkext = false;  // diagnostic: put the smallest and largest keys in the bulk load
  int failed = 0;        // any verification failure makes the run exit nonzero
  // recent: reads look up a key this thread inserted earlier in the same batch, mostly one of
  // its last 256, so they reach nodes being rebuilt and the side buffers of those nodes. Every
  // such read must hit, so a miss fails the run.
  bool recent = false;
  // order: insert order after the bulk load. "shuffle" (default) is uniform over the key
  // space; "sorted" inserts in key order (append-heavy); "zipf" concentrates inserts in hot
  // key ranges, drawing 1024 contiguous ranges with Zipfian weights.
  std::string order = "shuffle";
  double ztheta = 0.99;
  for (int i = 8; i < argc; i++) {
    std::string a = argv[i];
    if (a == "stagger") alexol::alex_stagger::enabled() = true;
    else if (a.rfind("data=", 0) == 0) data = a.substr(5);
    else if (a.rfind("readpct=", 0) == 0) readpct = atoi(a.c_str() + 8);
    else if (a == "destroy") destroy = true;
    else if (a == "bulkext") bulkext = true;
    else if (a == "recent") recent = true;
    else if (a.rfind("order=", 0) == 0) order = a.substr(6);
    else if (a.rfind("ztheta=", 0) == 0) ztheta = atof(a.c_str() + 7);
    else if (a.rfind("delta=", 0) == 0) alexol::alex_stagger::delta() = atof(a.c_str() + 6);
    else if (a == "lowonly") alexol::alex_stagger::low_only() = true;
#ifdef LOCK_STATS
    else if (a == "lockstats") alexol::lock_stats::enabled() = true;
#endif
    else if (a == "bg") alexol::bgexp::enabled() = true;
    else if (a == "side") alexol::sidebuf::enabled() = true;
#ifdef SIDEALWAYS
    // Ablation: a permanent per-node buffer, as XIndex keeps (patches/alexol-sidealways.patch).
    else if (a == "sidealways") { alexol::sidebuf::enabled() = true; alexol::sidebuf::always() = true; }
#endif
    else if (a.rfind("bgsoft=", 0) == 0) alexol::bgexp::soft() = atof(a.c_str() + 7);
    else if (a.rfind("bgthreads=", 0) == 0) alexol::bgexp::threads() = atoi(a.c_str() + 10);
    else if (a.rfind("nodebytes=", 0) == 0) alex_node_bytes() = atoi(a.c_str() + 10);
    else if (a.rfind("bfill=", 0) == 0) btree_bulk::fill() = atof(a.c_str() + 6);
    else if (a.rfind("bspread=", 0) == 0) {
      btree_bulk::spread() = atof(a.c_str() + 8);
      btree_bulk::randomized() = btree_bulk::spread() > 0.0;
    }
  }

  // Keys: real SOSD trace, or a shuffled dense odd sequence.
  std::vector<K> keys;
  if (!data.empty()) {
    keys = load_sosd(data, total);
    if (keys.size() < total) total = keys.size();
    PSORT(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    total = std::min(total, keys.size());
    keys.resize(total);
    std::shuffle(keys.begin(), keys.end(), std::mt19937_64(seed));
  } else {
    keys.resize(total);
    for (size_t i = 0; i < total; i++) keys[i] = (K)(i * 2 + 1);
    std::shuffle(keys.begin(), keys.end(), std::mt19937_64(seed));
  }

  if (bulkext && bulk >= 2) {
    // With both extremes in the bulk load, no insert falls outside the key domain, so the
    // index never expands its root. Used to isolate root expansion as a cause.
    auto mn = std::min_element(keys.begin(), keys.end()) - keys.begin();
    std::swap(keys[0], keys[mn]);
    auto mx = std::max_element(keys.begin() + 1, keys.end()) - keys.begin();
    std::swap(keys[1], keys[mx]);
  }
  if (order == "sorted") {
    PSORT(keys.begin() + bulk, keys.end());
  } else if (order == "zipf") {
    // Sort the inserted keys into R contiguous ranges, give each range a Zipfian weight (ranks
    // assigned to ranges at random), and emit keys by drawing a range by weight each time.
    const size_t R = 1024;
    std::vector<K> rest(keys.begin() + bulk, keys.end());
    PSORT(rest.begin(), rest.end());
    size_t n = rest.size();
    std::mt19937_64 zr(seed ^ 0x9e3779b97f4a7c15ULL);
    std::vector<size_t> rank(R);
    for (size_t r = 0; r < R; r++) rank[r] = r + 1;
    std::shuffle(rank.begin(), rank.end(), zr);
    std::vector<size_t> pos(R), end(R);
    std::vector<double> w(R);
    for (size_t r = 0; r < R; r++) {
      pos[r] = n * r / R; end[r] = n * (r + 1) / R;
      std::shuffle(rest.begin() + pos[r], rest.begin() + end[r], zr);
      w[r] = 1.0 / std::pow((double)rank[r], ztheta);
    }
    std::discrete_distribution<size_t> dd(w.begin(), w.end());
    for (size_t i = bulk; i < total; i++) {
      size_t r = dd(zr);
      if (pos[r] == end[r]) {  // range exhausted: drop it and redraw
        w[r] = 0; dd = std::discrete_distribution<size_t>(w.begin(), w.end()); i--; continue;
      }
      keys[i] = rest[pos[r]++];
    }
  } else if (order != "shuffle") {
    fprintf(stderr, "unknown order %s\n", order.c_str()); return 1;
  }
  fprintf(stderr, "ORDER %s recent=%d\n", order.c_str(), (int)recent);

  btree_bulk::seed() = seed;
  Index *idx = make(name);
  if (!idx) { fprintf(stderr, "unknown index %s\n", name.c_str()); return 1; }
  // Per-thread parameters. XIndex keys its per-worker state by thread_id and sizes it by
  // worker_num; the other wrappers ignore the argument.
  std::vector<Param> params;
  for (int t = 0; t < std::max(T, 1); t++) params.emplace_back((size_t)T, (uint32_t)t);
  idx->init(&params[0]);

  std::vector<std::pair<K, P>> init(bulk);
  for (size_t i = 0; i < bulk; i++) init[i] = {keys[i], (P)keys[i]};
  PSORT(init.begin(), init.end());
  idx->bulk_load(init.data(), bulk, &params[0]);

  // Slow-operation counts are the primary tail metric. Percentiles depend on batch size:
  // a wave produces roughly one slow insert per expanding data node, and when that count
  // falls below the percentile's rank cutoff the wave vanishes from the measurement. Counts
  // do not have that failure mode, so they are comparable across scales.
  printf("tag,index,seed,threads,batch,keys,p50_ns,p99_ns,p999_ns,p9999_ns,max_ns,"
         "n_gt10us,n_gt100us,n_gt1ms,n_ops,"
         "r_p50_ns,r_p99_ns,r_p999_ns,r_max_ns,r_gt100us,r_ops,"
         "batch_ns,mem_bytes\n");
  fflush(stdout);

  // Which keys were actually inserted, so the exhaustive check also covers mixed runs.
  std::vector<uint8_t> written(total, 0);
  for (size_t i = 0; i < bulk; i++) written[i] = 1;
  std::atomic<size_t> put_fail{0};
  std::atomic<size_t> read_miss{0};
  for (size_t off = bulk; off + batch <= total; off += batch) {
    std::vector<std::vector<long long>> lat(T), rlat(T);
    // CPU time of the whole process over the timed interval, background threads included.
    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    auto t0 = Clock::now();
    // Background rebuilds run only inside the timed interval: resumed after the clock
    // starts, and paused (waiting for any rebuild in flight) before it stops.
    if (alexol::bgexp::pause_hook()) alexol::bgexp::pause_hook()(false);
    // OpenMP, not std::thread. LIPP-OL and SALI index their per-thread node pool by
    // omp_get_thread_num() (lipp.h:612-624, sali.h:217). Outside an OpenMP parallel region
    // that returns 0 for every thread, so all threads would share one unsynchronized
    // std::stack -- LIPP-OL crashes its own RT_ASSERT and SALI races silently. GRE's own
    // harness uses OpenMP, so this also matches the code path that produced the published data.
#pragma omp parallel num_threads(T)
    {
      int t = omp_get_thread_num();
      size_t lo = off + (size_t)t * batch / T, hi = off + (size_t)(t + 1) * batch / T;
      // Thread-local buffer, moved into the shared vector only after timing stops.
      // Writing lat[t] directly during the loop would false-share: the vector objects
      // are 24 bytes apart, so several threads' size pointers live on one cache line
      // and ping-pong -- a confound that grows with thread count, which is the axis
      // under measurement.
      std::vector<long long> L, R;
      size_t misses = 0;
      L.reserve(hi - lo);
      std::mt19937_64 rng(seed * 1000 + t);
      for (size_t i = lo; i < hi; i++) {
        if (readpct && (int)(rng() % 100) < readpct) {
          // Reads are timed separately: whether a restructuring wave is visible on the
          // read path is a separate question from how it hits writes.
          //
          // Sample only from the bulk-loaded prefix, which is certainly present.
          // Sampling keys[rng() % i] draws keys that earlier read operations skipped and
          // keys owned by threads that have not inserted them yet, so a large share of
          // the "lookups" become misses timed as though they were hits. Indexing by loop
          // position rather than by insert count has the same defect, because a thread's
          // inserted keys are not contiguous once some iterations are reads.
          K rk = keys[rng() % bulk];
          if (recent && i > lo) {
            size_t win = std::min<size_t>(i - lo, 256);
            size_t j = (rng() % 4 == 0) ? lo + rng() % (i - lo) : i - 1 - rng() % win;
            if (written[j]) rk = keys[j];
          }
          P v;
          auto ra = Clock::now();
          bool hit = idx->get(rk, v, &params[t]);
          auto rb = Clock::now();
          if (!hit) misses++;
          R.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(rb - ra).count());
          continue;
        }
        auto a = Clock::now();
        bool ok = idx->put(keys[i], (P)keys[i], &params[t]);
        auto b = Clock::now();
        written[i] = 1;
        if (!ok) put_fail++;
        L.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
      }
      lat[t] = std::move(L);
      rlat[t] = std::move(R);
      if (misses) {
        fprintf(stderr, "READ_MISS t%d batch %zu: %zu misses\n", t,
                (off - bulk) / batch + 1, misses);
        read_miss += misses;
      }
    }
    if (alexol::bgexp::pause_hook()) alexol::bgexp::pause_hook()(true);
    auto t1 = Clock::now();
    getrusage(RUSAGE_SELF, &ru1);
    auto tv = [](const timeval &a) { return (long long)a.tv_sec * 1000000000LL + a.tv_usec * 1000LL; };
    long long cpu_ns = tv(ru1.ru_utime) - tv(ru0.ru_utime) + tv(ru1.ru_stime) - tv(ru0.ru_stime);

    std::vector<long long> all, rall;
    for (auto &v : lat) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    for (auto &v : rlat) rall.insert(rall.end(), v.begin(), v.end());
    std::sort(rall.begin(), rall.end());
    auto q = [&](double p) {
      return all.empty() ? 0LL : all[std::min(all.size() - 1, (size_t)(p * all.size()))];
    };
    auto above = [&](long long ns) {
      return (long long)(all.end() - std::lower_bound(all.begin(), all.end(), ns));
    };
    auto rq = [&](double p) {
      return rall.empty() ? 0LL : rall[std::min(rall.size() - 1, (size_t)(p * rall.size()))];
    };
    long long r_gt100 =
        (long long)(rall.end() - std::lower_bound(rall.begin(), rall.end(), 100000LL));
    printf("%s,%s,%u,%d,%zu,%zu,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%zu,"
           "%lld,%lld,%lld,%lld,%lld,%zu,%lld,%lld\n",
           tag, name.c_str(), seed, T, (off - bulk) / batch + 1, off + batch,
           q(0.50), q(0.99), q(0.999), q(0.9999), all.empty() ? 0LL : all.back(),
           above(10000), above(100000), above(1000000), all.size(),
           rq(0.50), rq(0.99), rq(0.999), rall.empty() ? 0LL : rall.back(),
           r_gt100, rall.size(),
           (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
           (long long)idx->memory_consumption());
    printf("CPU,%s,%u,%d,%zu,%lld\n", tag, seed, T, (off - bulk) / batch + 1, cpu_ns);
#ifdef LOCK_STATS
    if (alexol::lock_stats::enabled()) {
      static alexol::lock_stats::Counters prev;
      auto now = alexol::lock_stats::total();
      printf("LOCK,%s,%u,%d,%zu,%zu,%llu,%llu,%llu,%llu\n", tag, seed, T,
             (off - bulk) / batch + 1, off + batch,
             (unsigned long long)(now.lock_busy - prev.lock_busy),
             (unsigned long long)(now.validate_fail - prev.validate_fail),
             (unsigned long long)(now.read_denied - prev.read_denied),
             (unsigned long long)(now.write_denied - prev.write_denied));
      printf("LOCK2,%s,%u,%d,%zu,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", tag, seed, T,
             (off - bulk) / batch + 1, off + batch,
             (unsigned long long)(now.retry - prev.retry),
             (unsigned long long)(now.retry_smo - prev.retry_smo),
             (unsigned long long)(now.retry_ins - prev.retry_ins),
             (unsigned long long)(now.parentspin - prev.parentspin),
             (unsigned long long)(now.fg_smo - prev.fg_smo),
             (unsigned long long)(now.bg_smo - prev.bg_smo),
             (unsigned long long)(now.bg_skip - prev.bg_skip),
             (unsigned long long)(now.smo_ns - prev.smo_ns),
             (unsigned long long)now.smo_max_ns,
             (unsigned long long)(now.side_ins - prev.side_ins),
             (unsigned long long)(now.side_read - prev.side_read),
             (unsigned long long)(now.side_full - prev.side_full),
             (unsigned long long)(now.smo_split - prev.smo_split),
             (unsigned long long)(now.retry_noside - prev.retry_noside),
             (unsigned long long)(now.split_ns - prev.split_ns));
      printf("LOCK3,%s,%u,%d,%zu,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", tag, seed, T,
             (off - bulk) / batch + 1, off + batch,
             (unsigned long long)(now.slow_fgsmo - prev.slow_fgsmo),
             (unsigned long long)(now.slow_sealed - prev.slow_sealed),
             (unsigned long long)(now.slow_rsmo - prev.slow_rsmo),
             (unsigned long long)(now.slow_rins - prev.slow_rins),
             (unsigned long long)(now.slow_none - prev.slow_none),
             (unsigned long long)(now.root_expand - prev.root_expand),
             (unsigned long long)(now.boundary_hit - prev.boundary_hit));
      prev = now;
    }
#endif
    fflush(stdout);
  }

  // Memory as the operating system sees it, for indexes whose own accounting is missing
  // (XIndex and FINEdex report 0). Includes the driver's key arrays, which are the same size
  // for every index, so differences between indexes are meaningful and absolute values are not.
  {
    long rss_pages = 0, tmp = 0;
    FILE *sf = fopen("/proc/self/statm", "r");
    if (sf) { if (fscanf(sf, "%ld %ld", &tmp, &rss_pages) != 2) rss_pages = 0; fclose(sf); }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "RSS %s t%d seed%u: end_kb=%ld peak_kb=%ld\n", name.c_str(), T, seed,
            rss_pages * (sysconf(_SC_PAGESIZE) / 1024), (long)ru.ru_maxrss);
  }

  // Correctness gate. A silently failing put() would make every latency number above
  // a measurement of nothing, so probe a sample of what should be present.
  {
    // With a read fraction, only the bulk-loaded prefix is guaranteed present: the rest
    // of the key range is split between inserts and reads, so probing it would report a
    // recall below 100% that reflects the workload mix rather than a fault.
    size_t done = readpct ? bulk : bulk + ((total - bulk) / batch) * batch;
    std::mt19937_64 rng(12345);
    size_t n = std::min<size_t>(100000, done), hit = 0;
    for (size_t i = 0; i < n; i++) {
      P v = 0;
      K k = keys[rng() % done];
      if (idx->get(k, v, &params[0]) && v == (P)k) hit++;
    }
    fprintf(stderr, "VERIFY %s t%d seed%u: %zu/%zu probes found (%.4f%%)\n",
            name.c_str(), T, seed, hit, n, 100.0 * hit / n);
    if (hit < n) failed = 1;
    if (hit * 100 < n * 99)
      fprintf(stderr, "VERIFY_FAIL %s: recall %.4f%% below 99%%\n", name.c_str(), 100.0 * hit / n);
  }
  // Exhaustive check: every key that should be present, with its payload. The side buffer
  // routes entries during splits, and a routing error would lose keys without crashing,
  // so the sampled check above is not enough to trust it.
  {
    if (put_fail) fprintf(stderr, "PUT_FAIL %s: %zu inserts returned false\n", name.c_str(),
                          put_fail.load());
    size_t done = bulk + ((total - bulk) / batch) * batch, bad = 0, checked = 0;
    // XIndex's per-worker state is sized by the thread count it was built with.
    int vt = (name == "xindex") ? T : 16;
#pragma omp parallel for reduction(+ : bad, checked) num_threads(vt)
    for (size_t i = 0; i < done; i++) {
      if (!written[i]) continue;
      checked++;
      P v = 0;
      if (!idx->get(keys[i], v, &params[omp_get_thread_num() % params.size()]) || v != (P)keys[i]) {
        bad++;
        fprintf(stderr, "MISSING key=%llu index=%zu got=%llu\n", (unsigned long long)keys[i], i,
                (unsigned long long)v);
      }
    }
    done = checked;
    fprintf(stderr, "FULLVERIFY %s t%d seed%u: %zu/%zu missing or wrong\n", name.c_str(), T,
            seed, bad, done);
    if (bad) { fprintf(stderr, "FULLVERIFY_FAIL %s\n", name.c_str()); failed = 1; }
  }
  if (read_miss) { fprintf(stderr, "READ_MISS_FAIL %s: %zu\n", name.c_str(), read_miss.load()); failed = 1; }
  if (destroy) {
    // Teardown with background work pending: resume the threads, then destroy the index.
    if (alexol::bgexp::pause_hook()) alexol::bgexp::pause_hook()(false);
    // indexInterface has no virtual destructor, so delete through the concrete type.
    if (name == "alexol") delete static_cast<alexolInterface<K, P> *>(idx);
    else if (name == "alexsized") delete static_cast<AlexSizedInterface<K, P> *>(idx);
    fprintf(stderr, "DESTROYED %s\n", name.c_str());
  }
  return failed ? 2 : 0;
}
