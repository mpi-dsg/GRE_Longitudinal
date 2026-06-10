#pragma once

#include <algorithm>
#include <atomic>
#include <ctime>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../competitor/competitor.h"
#include "../competitor/indexInterface.h"
#include "../tscns.h"
#include "flags.h"
#include "omp.h"
#include "pgm_metric.h"
#include "tbb/parallel_sort.h"
#include "utils.h"
#include <jemalloc/jemalloc.h>

// Longitudinal benchmark. All batch operations are generated once, before any
// index runs, and every index is evaluated against the same sequence, so
// throughput differences reflect index behaviour and not sampling differences.
// Use --save_ops=<path> to write the generated batches to a binary file and
// --load_ops=<path> to replay a previously saved file instead of generating.

template <typename KEY_TYPE, typename PAYLOAD_TYPE>
class BenchmarkLongbench {

  typedef indexInterface<KEY_TYPE, PAYLOAD_TYPE> index_t;

  enum Operation : uint8_t { READ = 0, INSERT, DELETE, SCAN, UPDATE };

  enum OperationOrder {
    ITERATE_SORT_UNSORTED = 0,
    SHUFFLE_ALL,
    SHUFFLE_THEN_SORT
  };

  static const uint32_t OPS_FILE_MAGIC = 0x4C424F50;
  static const uint32_t OPS_FILE_VERSION = 1;

  double read_ratio   = 1;
  double insert_ratio = 0;
  double delete_ratio = 0;
  double update_ratio = 0;
  double scan_ratio   = 0;
  size_t scan_num     = 100;
  size_t operations_num;
  long long table_size     = -1;
  size_t    init_table_size;
  double    init_table_ratio;
  double    del_table_ratio;
  size_t    thread_num = 1;

  std::string              index_type;
  std::vector<std::string> index_types;   // populated by --indexes=a,b,c
  std::string keys_file_path;
  std::string keys_file_type;
  std::string sample_distribution;
  bool   latency_sample       = false;
  double latency_sample_ratio = 0.01;
  int    error_bound;
  std::string output_path;
  size_t      random_seed;
  bool memory_record    = false;
  bool dataset_statistic = false;
  bool data_shift        = false;

  std::string save_ops_path;
  std::string load_ops_path;

  OperationOrder operation_order    = ITERATE_SORT_UNSORTED;

  std::vector<KEY_TYPE> active_keys;
  std::vector<KEY_TYPE> init_keys;
  KEY_TYPE *keys = nullptr;
  std::pair<KEY_TYPE, PAYLOAD_TYPE> *init_key_values = nullptr;
  std::mt19937 gen;
  int iteration_num = 0;

  // running key count, updated after every batch and reported in the CSV
  size_t current_table_size = 0;

  // all_batch_ops[i] holds the full operation vector for batch i
  std::vector<std::vector<std::pair<Operation, KEY_TYPE>>> all_batch_ops;

  struct Stat {
    std::vector<double> read_latency;
    std::vector<double> write_latency;
    uint64_t  throughput          = 0;
    size_t    fitness_of_dataset  = 0;
    long long memory_consumption  = 0;
    uint64_t  success_insert      = 0;
    uint64_t  success_read        = 0;
    uint64_t  success_update      = 0;
    uint64_t  success_remove      = 0;
    uint64_t  scan_not_enough     = 0;

    void clear() {
      read_latency.clear();
      write_latency.clear();
      throughput = fitness_of_dataset = memory_consumption = 0;
      success_insert = success_read = success_update =
        success_remove = scan_not_enough = 0;
    }
  } stat;

  struct alignas(64) ThreadParam {
    std::vector<std::pair<uint64_t, uint64_t>> read_latency;
    std::vector<std::pair<uint64_t, uint64_t>> write_latency;
    uint64_t success_insert    = 0;
    uint64_t success_read      = 0;
    uint64_t success_update    = 0;
    uint64_t success_remove    = 0;
    uint64_t scan_not_enough   = 0;
  };
  typedef ThreadParam param_t;

public:
  BenchmarkLongbench() {}

  KEY_TYPE *load_keys() {
    COUT_THIS("Reading data from file.");
    if (table_size > 0) keys = new KEY_TYPE[table_size];

    if (keys_file_type == "binary")
      table_size = load_binary_data(keys, table_size, keys_file_path);
    else if (keys_file_type == "text")
      table_size = load_text_data(keys, table_size, keys_file_path);
    else {
      COUT_THIS("Unknown key file type."); exit(1);
    }
    if (table_size <= 0) { COUT_THIS("Failed to load keys."); exit(1); }

    if (!data_shift) {
      tbb::parallel_sort(keys, keys + table_size);
      auto last = std::unique(keys, keys + table_size);
      table_size = last - keys;
      std::shuffle(keys, keys + table_size, gen);
    }

    init_table_size = static_cast<size_t>(init_table_ratio * table_size);
    std::cout << "table_size=" << table_size
              << "  init_table_size=" << init_table_size << "\n";

    init_keys.resize(init_table_size);
    #pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i < init_table_size; ++i) init_keys[i] = keys[i];
    tbb::parallel_sort(init_keys.begin(), init_keys.end());

    init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_table_size];
    #pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i < init_table_size; ++i)
      init_key_values[i] = {init_keys[i], 123456789};

    active_keys.clear();
    active_keys.insert(active_keys.end(), init_keys.begin(), init_keys.end());
    current_table_size = init_table_size;
    return keys;
  }

  inline void prepare(index_t *&index) {
    index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);
    Param param(thread_num, 0);
    index->init(&param);
    thread_num = param.worker_num;
    COUT_THIS("bulk loading");
    index->bulk_load(init_key_values, init_keys.size(), &param);
    COUT_THIS("bulk load done");
  }

  inline void parse_args(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);
    keys_file_path      = get_required(flags, "keys_file");
    keys_file_type      = get_with_default(flags, "keys_file_type", "binary");
    read_ratio          = stod(get_required(flags, "read"));
    insert_ratio        = stod(get_with_default(flags, "insert",        "0"));
    delete_ratio        = stod(get_with_default(flags, "delete",        "0"));
    update_ratio        = stod(get_with_default(flags, "update",        "0"));
    scan_ratio          = stod(get_with_default(flags, "scan",          "0"));
    scan_num            = stoi(get_with_default(flags, "scan_num",      "100"));
    operations_num      = stoi(get_with_default(flags, "operations_num","1000000000"));
    table_size          = stoi(get_with_default(flags, "table_size",    "-1"));
    init_table_ratio    = stod(get_with_default(flags, "init_table_ratio","0.5"));
    del_table_ratio     = stod(get_with_default(flags, "del_table_ratio", "0.5"));
    thread_num          = stoi(get_with_default(flags, "thread_num",    "1"));
    index_type          = get_with_default(flags, "index",              "alexol");
    sample_distribution = get_with_default(flags, "sample_distribution","uniform");
    latency_sample      = get_boolean_flag(flags,  "latency_sample");
    latency_sample_ratio= stod(get_with_default(flags, "latency_sample_ratio","0.01"));
    error_bound         = stoi(get_with_default(flags, "error_bound",   "64"));
    output_path         = get_with_default(flags, "output_path",        "./out.csv");
    random_seed         = stoul(get_with_default(flags, "seed",         "1866"));
    memory_record       = get_boolean_flag(flags,  "memory");
    dataset_statistic   = get_boolean_flag(flags,  "dataset_statistic");
    data_shift          = get_boolean_flag(flags,  "data_shift");
    save_ops_path       = get_with_default(flags, "save_ops",           "");
    load_ops_path       = get_with_default(flags, "load_ops",           "");

    // --indexes=alex,lipp,dili runs all listed indexes against the same ops
    {
      std::string idx_list = get_with_default(flags, "indexes", "");
      if (!idx_list.empty()) {
        std::stringstream ss(idx_list);
        std::string tok;
        while (std::getline(ss, tok, ','))
          if (!tok.empty()) index_types.push_back(tok);
      }
    }

    gen.seed(random_seed);
    init_table_size = 0;

    std::string order_str = get_with_default(flags, "operation_order", "iterate");
    if      (order_str == "iterate")       operation_order = ITERATE_SORT_UNSORTED;
    else if (order_str == "shuffle")       operation_order = SHUFFLE_ALL;
    else if (order_str == "shuffle_sort")  operation_order = SHUFFLE_THEN_SORT;

    double ratio_sum = read_ratio + insert_ratio + delete_ratio +
                       update_ratio + scan_ratio;
    INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);
    INVARIANT(thread_num > 0);

    COUT_THIS("Read:Insert:Update:Scan:Delete = "
      << read_ratio <<":"<< insert_ratio <<":"<< update_ratio
      <<":"<< scan_ratio <<":"<< delete_ratio);
  }

  // Generates one batch. All mutable state is passed in explicitly so this
  // can run during pre-generation without touching the members used while
  // the benchmark itself runs.
  void generate_single_batch(
      std::vector<std::pair<Operation, KEY_TYPE>> &ops,
      std::vector<KEY_TYPE>                       &act_keys,
      size_t                                      &next_insert_pos,
      bool                                        &batch_sorted,
      std::mt19937                                &rng,
      size_t                                      &search_seed)
  {
    ops.clear();
    ops.reserve(operations_num);

    const size_t n_read   = static_cast<size_t>(operations_num * read_ratio);
    const size_t n_insert = static_cast<size_t>(operations_num * insert_ratio);
    const size_t n_update = static_cast<size_t>(operations_num * update_ratio);
    const size_t n_scan   = static_cast<size_t>(operations_num * scan_ratio);
    const size_t n_delete = static_cast<size_t>(operations_num * delete_ratio);

    std::vector<std::pair<Operation, KEY_TYPE>> temp;
    temp.reserve(operations_num);

    // deletes first: remove from the active set so they cannot be read later
    for (size_t i = 0; i < n_delete; ++i) {
      if (act_keys.empty()) continue;
      std::uniform_int_distribution<size_t> dist(0, act_keys.size() - 1);
      size_t idx  = dist(rng);
      KEY_TYPE key = act_keys[idx];
      act_keys[idx] = act_keys.back();
      act_keys.pop_back();
      temp.emplace_back(DELETE, key);
    }

    // reads/updates/scans are sampled from the surviving active keys, before
    // this batch's inserts are appended, so no operation in a batch depends
    // on another operation in the same batch
    KEY_TYPE *sample_ptr = nullptr;
    if (!act_keys.empty()) {
      if (sample_distribution == "uniform")
        sample_ptr = get_search_keys(
            act_keys.data(), act_keys.size(), operations_num, &search_seed);
      else
        sample_ptr = get_search_keys_zipf(
            act_keys.data(), act_keys.size(), operations_num, &search_seed);
    }

    size_t sc = 0;
    for (size_t i = 0; i < n_read + n_update + n_scan; ++i) {
      if (!sample_ptr) continue;
      KEY_TYPE key = sample_ptr[sc++];
      Operation op;
      if      (i < n_read)             op = READ;
      else if (i < n_read + n_update)  op = UPDATE;
      else                             op = SCAN;
      temp.emplace_back(op, key);
    }
    if (sample_ptr) { delete[] sample_ptr; sample_ptr = nullptr; }

    // inserts advance through the shuffled key array and join the active set
    for (size_t i = 0; i < n_insert; ++i) {
      if (next_insert_pos >= static_cast<size_t>(table_size)) continue;
      KEY_TYPE key = keys[next_insert_pos++];
      temp.emplace_back(INSERT, key);
      act_keys.push_back(key);
    }

    auto cmp = [](const std::pair<Operation,KEY_TYPE> &a,
                  const std::pair<Operation,KEY_TYPE> &b) {
      return a.first != b.first ? a.first < b.first : a.second < b.second;
    };
    switch (operation_order) {
      case ITERATE_SORT_UNSORTED:
        if (batch_sorted) std::sort(temp.begin(), temp.end(), cmp);
        else              std::shuffle(temp.begin(), temp.end(), rng);
        batch_sorted = !batch_sorted;
        break;
      case SHUFFLE_ALL:
        std::shuffle(temp.begin(), temp.end(), rng);
        break;
      case SHUFFLE_THEN_SORT:
        std::shuffle(temp.begin(), temp.end(), rng);
        std::sort(temp.begin(), temp.end(), cmp);
        break;
    }
    ops = std::move(temp);
  }

  void pregenerate_all_batches() {
    int n_runs = static_cast<int>(
        (table_size - static_cast<long long>(init_table_size)) /
        (operations_num * insert_ratio));
    if (read_ratio == 1.0) n_runs = 2;
    if (n_runs < 1)        n_runs = 1;

    COUT_THIS("Pre-generating " << n_runs << " batches ("
              << operations_num << " ops each)...");

    // work on copies so the class members stay untouched
    std::vector<KEY_TYPE> sim_active  = active_keys;
    size_t                sim_insert  = init_table_size;
    bool                  sim_sorted  = false;
    std::mt19937          sim_rng     = gen;
    size_t                sim_search  = random_seed;

    all_batch_ops.resize(n_runs);
    for (int i = 0; i < n_runs; ++i) {
      generate_single_batch(all_batch_ops[i],
                            sim_active, sim_insert,
                            sim_sorted, sim_rng, sim_search);
      COUT_THIS("  batch " << i+1 << "/" << n_runs
                << "  ops=" << all_batch_ops[i].size()
                << "  active_keys=" << sim_active.size());
    }
    COUT_THIS("Pre-generation complete.");
  }

  // File layout: magic, version, sizeof(KEY_TYPE), n_batches, then for each
  // batch a uint64 op count followed by (uint8 op, KEY_TYPE key) pairs.
  void save_ops_to_file(const std::string &path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << " for writing\n"; return; }

    uint32_t magic = OPS_FILE_MAGIC;
    uint32_t version = OPS_FILE_VERSION;
    uint32_t key_size = sizeof(KEY_TYPE);
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    f.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));

    uint32_t n = static_cast<uint32_t>(all_batch_ops.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (auto &batch : all_batch_ops) {
      uint64_t sz = batch.size();
      f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
      for (auto &op : batch) {
        uint8_t  t = static_cast<uint8_t>(op.first);
        KEY_TYPE k = op.second;
        f.write(reinterpret_cast<const char*>(&t), sizeof(t));
        f.write(reinterpret_cast<const char*>(&k), sizeof(k));
      }
    }
    COUT_THIS("Saved " << n << " batches to " << path);
  }

  bool load_ops_from_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = 0, version = 0, key_size = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
    if (!f || magic != OPS_FILE_MAGIC || version != OPS_FILE_VERSION ||
        key_size != sizeof(KEY_TYPE)) {
      std::cerr << "Ops file " << path
                << " has wrong magic/version/key size, regenerating\n";
      return false;
    }

    uint32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    all_batch_ops.resize(n);

    for (uint32_t i = 0; i < n; ++i) {
      uint64_t sz;
      f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
      all_batch_ops[i].resize(sz);
      for (uint64_t j = 0; j < sz; ++j) {
        uint8_t  t; KEY_TYPE k;
        f.read(reinterpret_cast<char*>(&t), sizeof(t));
        f.read(reinterpret_cast<char*>(&k), sizeof(k));
        all_batch_ops[i][j] = {static_cast<Operation>(t), k};
      }
    }
    COUT_THIS("Loaded " << n << " batches from " << path);
    return true;
  }

  void run_batch(index_t *index,
                 std::vector<std::pair<Operation, KEY_TYPE>> &operations) {
    std::vector<param_t> params(thread_num);
    // the last batch may be short if the insert pool ran out, so the loop
    // bound must come from the batch itself
    const int n_ops = static_cast<int>(operations.size());
    TSCNS tn; tn.init();
    auto start_time = tn.rdtsc();
    auto end_time   = tn.rdtsc();

    int lat_interval = static_cast<int>(
        operations_num / (operations_num * latency_sample_ratio + 1));
    if (lat_interval < 1) lat_interval = 1;

    #pragma omp parallel num_threads(thread_num)
    {
      auto tid    = omp_get_thread_num();
      auto paramI = Param(thread_num, tid);
      param_t &tp = params[tid];
      tp.read_latency.reserve(operations_num / (lat_interval + 1));
      tp.write_latency.reserve(operations_num / (lat_interval + 1));

      PAYLOAD_TYPE val;
      std::pair<KEY_TYPE, PAYLOAD_TYPE> *scan_result =
          new std::pair<KEY_TYPE, PAYLOAD_TYPE>[scan_num];

      #pragma omp barrier
      #pragma omp master
      start_time = tn.rdtsc();

      #pragma omp for schedule(dynamic, 10000)
      for (int i = 0; i < n_ops; ++i) {
        auto op  = operations[i].first;
        auto key = operations[i].second;
        uint64_t ts0 = (latency_sample && i % lat_interval == 0) ? tn.rdtsc() : 0;

        switch (op) {
          case READ:
            tp.success_read += index->get(key, val, &paramI); break;
          case INSERT:
            tp.success_insert += index->put(key, 123456789, &paramI); break;
          case UPDATE:
            tp.success_update += index->update(key, 234567891, &paramI); break;
          case SCAN: {
            auto len = index->scan(key, scan_num, scan_result, &paramI);
            if (len != scan_num) tp.scan_not_enough++;
            break;
          }
          case DELETE:
            tp.success_remove += index->remove(key, &paramI); break;
        }

        if (latency_sample && i % lat_interval == 0) {
          uint64_t ts1 = tn.rdtsc();
          if (op == READ) tp.read_latency.emplace_back(ts0, ts1);
          else            tp.write_latency.emplace_back(ts0, ts1);
        }
      }

      #pragma omp master
      end_time = tn.rdtsc();
      delete[] scan_result;
    }

    double elapsed_ns = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
    stat.throughput =
        static_cast<uint64_t>(operations.size() / (elapsed_ns / 1e9));

    for (auto &p : params) {
      if (latency_sample) {
        for (auto e : p.read_latency)
          stat.read_latency.push_back(tn.tsc2ns(e.second) - tn.tsc2ns(e.first));
        for (auto e : p.write_latency)
          stat.write_latency.push_back(tn.tsc2ns(e.second) - tn.tsc2ns(e.first));
      }
      stat.success_read   += p.success_read;
      stat.success_insert += p.success_insert;
      stat.success_update += p.success_update;
      stat.success_remove += p.success_remove;
      stat.scan_not_enough += p.scan_not_enough;
    }

    if (dataset_statistic) {
      tbb::parallel_sort(init_keys.begin(), init_keys.end());
      stat.fitness_of_dataset =
          pgmMetric::PGM_metric(init_keys.data(), init_keys.size(), error_bound);
    }
    if (memory_record)
      stat.memory_consumption = index->memory_consumption();

    // advance the running key count by the net change in this batch, counted
    // from the generated ops so it stays exact even if operations fail
    size_t n_ins = 0, n_del = 0;
    for (auto &op : operations) {
      if (op.first == INSERT) ++n_ins;
      else if (op.first == DELETE) ++n_del;
    }
    current_table_size += n_ins;
    if (current_table_size >= n_del) current_table_size -= n_del;

    print_stat();
  }

  void print_stat(bool clear_flag = true) {
    // sort once so the min/max/percentile reads below are all correct
    std::sort(stat.read_latency.begin(), stat.read_latency.end());
    std::sort(stat.write_latency.begin(), stat.write_latency.end());

    auto percentile = [](const std::vector<double> &v, double p) -> double {
      if (v.empty()) return 0;
      size_t i = static_cast<size_t>(p * v.size());
      if (i >= v.size()) i = v.size() - 1;
      return v[i];
    };

    double avg_r = 0, avg_w = 0, var_r = 0, var_w = 0;
    if (!stat.read_latency.empty()) {
      for (auto t : stat.read_latency) avg_r += t;
      avg_r /= stat.read_latency.size();
      for (auto t : stat.read_latency) var_r += (t-avg_r)*(t-avg_r);
      var_r /= stat.read_latency.size();
    }
    if (!stat.write_latency.empty()) {
      for (auto t : stat.write_latency) avg_w += t;
      avg_w /= stat.write_latency.size();
      for (auto t : stat.write_latency) var_w += (t-avg_w)*(t-avg_w);
      var_w /= stat.write_latency.size();
    }

    printf("Throughput=%llu  Memory=%lld  reads=%llu  inserts=%llu\n",
           stat.throughput, stat.memory_consumption,
           stat.success_read, stat.success_insert);

    if (!file_exists(output_path)) {
      std::ofstream h(output_path, std::ios::app);
      h << "timestamp,read_ratio,insert_ratio,update_ratio,scan_ratio,"
           "delete_ratio,key_path,index_type,throughput,table_size,"
           "memory_consumption,thread_num,"
           "r_min,r_p50,r_p90,r_p99,r_p999,r_p9999,r_max,r_avg,"
           "w_min,w_p50,w_p90,w_p99,w_p999,w_p9999,w_max,w_avg,"
           "seed,scan_num,r_var,w_var,latency_sample,data_shift,"
           "pgm,error_bound,file_table_size\n";
    }

    std::time_t t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d%H%M%S", std::localtime(&t));

    std::ofstream o(output_path, std::ios::app);
    o << ts << ","
      << read_ratio   << "," << insert_ratio << ","
      << update_ratio << "," << scan_ratio   << "," << delete_ratio << ","
      << keys_file_path << "," << index_type << ","
      << stat.throughput << "," << current_table_size << ","
      << stat.memory_consumption << "," << thread_num << ",";

    if (latency_sample) {
      auto &r = stat.read_latency;
      auto &w = stat.write_latency;
      o << (r.empty()?0:r.front())                 << ","
        << percentile(stat.read_latency,  0.50)    << ","
        << percentile(stat.read_latency,  0.90)    << ","
        << percentile(stat.read_latency,  0.99)    << ","
        << percentile(stat.read_latency,  0.999)   << ","
        << percentile(stat.read_latency,  0.9999)  << ","
        << (r.empty()?0:r.back())                  << "," << avg_r << ","
        << (w.empty()?0:w.front())                 << ","
        << percentile(stat.write_latency, 0.50)    << ","
        << percentile(stat.write_latency, 0.90)    << ","
        << percentile(stat.write_latency, 0.99)    << ","
        << percentile(stat.write_latency, 0.999)   << ","
        << percentile(stat.write_latency, 0.9999)  << ","
        << (w.empty()?0:w.back())                  << "," << avg_w << ",";
    } else {
      for (int i = 0; i < 16; ++i) o << "0,";
    }

    o << random_seed << "," << scan_num << ","
      << var_r << "," << var_w << ","
      << latency_sample << "," << data_shift << ","
      << stat.fitness_of_dataset << "," << error_bound << ","
      << table_size << "\n";
    o.close();

    if (clear_flag) stat.clear();
  }

  void run_benchmark() {
    load_keys();

    // resolve operations once; they are shared by all indexes
    bool loaded = false;
    if (!load_ops_path.empty()) {
      loaded = load_ops_from_file(load_ops_path);
      if (!loaded)
        COUT_THIS("Warning: could not load ops from " << load_ops_path
                  << ", falling back to in-memory generation.");
    }
    if (!loaded) {
      pregenerate_all_batches();
      if (!save_ops_path.empty())
        save_ops_to_file(save_ops_path);
    }
    COUT_THIS("Total batches to replay: " << all_batch_ops.size());

    // --indexes=alex,lipp,dili takes priority over single --index=x
    std::vector<std::string> to_run =
        index_types.empty() ? std::vector<std::string>{index_type} : index_types;

    COUT_THIS("Indexes to benchmark (" << to_run.size() << "): ");
    for (auto &n : to_run) COUT_THIS("  " << n);

    for (const auto &idx_name : to_run) {
      index_type = idx_name;   // print_stat() reads index_type for CSV output
      current_table_size = init_table_size;  // reset for each index
      COUT_THIS("\n=== Index: " << idx_name << " ===");

      index_t *index = nullptr;
      prepare(index);

      for (size_t b = 0; b < all_batch_ops.size(); ++b) {
        COUT_THIS("  batch " << b+1 << "/" << all_batch_ops.size());
        run_batch(index, all_batch_ops[b]);
      }

      delete index;
      index = nullptr;
    }
  }
};
