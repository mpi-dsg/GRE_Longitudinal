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
#if defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm>
#define PSORT __gnu_parallel::sort
#else
#define PSORT std::sort
#endif
#include <vector>

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
  for (int i = 8; i < argc; i++) {
    std::string a = argv[i];
    if (a == "stagger") alexol::alex_stagger::enabled() = true;
    else if (a.rfind("data=", 0) == 0) data = a.substr(5);
    else if (a.rfind("readpct=", 0) == 0) readpct = atoi(a.c_str() + 8);
    else if (a.rfind("delta=", 0) == 0) alexol::alex_stagger::delta() = atof(a.c_str() + 6);
    else if (a == "lowonly") alexol::alex_stagger::low_only() = true;
#ifdef LOCK_STATS
    else if (a == "lockstats") alexol::lock_stats::enabled() = true;
#endif
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

  btree_bulk::seed() = seed;
  Index *idx = make(name);
  if (!idx) { fprintf(stderr, "unknown index %s\n", name.c_str()); return 1; }
  idx->init(nullptr);

  std::vector<std::pair<K, P>> init(bulk);
  for (size_t i = 0; i < bulk; i++) init[i] = {keys[i], (P)keys[i]};
  PSORT(init.begin(), init.end());
  idx->bulk_load(init.data(), bulk, nullptr);

  // Slow-operation counts are the primary tail metric. Percentiles depend on batch size:
  // a wave produces roughly one slow insert per expanding data node, and when that count
  // falls below the percentile's rank cutoff the wave vanishes from the measurement. Counts
  // do not have that failure mode, so they are comparable across scales.
  printf("tag,index,seed,threads,batch,keys,p50_ns,p99_ns,p999_ns,p9999_ns,max_ns,"
         "n_gt10us,n_gt100us,n_gt1ms,n_ops,"
         "r_p50_ns,r_p99_ns,r_p999_ns,r_max_ns,r_gt100us,r_ops,"
         "batch_ns,mem_bytes\n");
  fflush(stdout);

  for (size_t off = bulk; off + batch <= total; off += batch) {
    std::vector<std::vector<long long>> lat(T), rlat(T);
    auto t0 = Clock::now();
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
      L.reserve(hi - lo);
      std::mt19937_64 rng(seed * 1000 + t);
      for (size_t i = lo; i < hi; i++) {
        if (readpct && (int)(rng() % 100) < readpct) {
          // Reads are timed separately. Whether a restructuring wave is visible on the
          // read path at all is a separate question from how it hits writes.
          P v;
          auto ra = Clock::now();
          idx->get(keys[rng() % i], v, nullptr);
          auto rb = Clock::now();
          R.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(rb - ra).count());
          continue;
        }
        auto a = Clock::now();
        idx->put(keys[i], (P)keys[i], nullptr);
        auto b = Clock::now();
        L.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
      }
      lat[t] = std::move(L);
      rlat[t] = std::move(R);
    }
    auto t1 = Clock::now();

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
      prev = now;
    }
#endif
    fflush(stdout);
  }

  // Correctness gate. A silently failing put() would make every latency number above
  // a measurement of nothing, so probe a sample of what should be present.
  {
    size_t done = bulk + ((total - bulk) / batch) * batch;
    std::mt19937_64 rng(12345);
    size_t n = std::min<size_t>(100000, done), hit = 0;
    for (size_t i = 0; i < n; i++) {
      P v = 0;
      K k = keys[rng() % done];
      if (idx->get(k, v, nullptr) && v == (P)k) hit++;
    }
    fprintf(stderr, "VERIFY %s t%d seed%u: %zu/%zu probes found (%.4f%%)\n",
            name.c_str(), T, seed, hit, n, 100.0 * hit / n);
    if (hit * 100 < n * 99)
      fprintf(stderr, "VERIFY_FAIL %s: recall %.4f%% below 99%%\n", name.c_str(), 100.0 * hit / n);
  }
  return 0;
}
