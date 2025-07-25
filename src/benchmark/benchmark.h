#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <ctime>
#include <unordered_set>
#include <stdexcept>
#include <sstream>

#include "../tscns.h"
#include "omp.h"
#include "tbb/parallel_sort.h"
#include "flags.h"
#include "utils.h"
#include "../competitor/competitor.h"
#include "../competitor/indexInterface.h"
#include "pgm_metric.h"
#include <jemalloc/jemalloc.h>

template<typename KEY_TYPE, typename PAYLOAD_TYPE>
class Benchmark {
    typedef indexInterface<KEY_TYPE, PAYLOAD_TYPE> index_t;

    enum Operation {
        READ = 0, INSERT, DELETE, SCAN, UPDATE
    };

    enum OperationOrder {
        ITERATE_SORT_UNSORTED = 0,
        SHUFFLE_ALL,
        SHUFFLE_THEN_SORT
    };

    // parameters
    double read_ratio = 1;
    double insert_ratio = 0;
    double delete_ratio = 0;
    double update_ratio = 0;
    double scan_ratio = 0;
    size_t scan_num = 100;
    size_t operations_num;
    long long table_size = -1;
    size_t init_table_size;
    double init_table_ratio;
    double del_table_ratio;
    size_t thread_num = 1;
    size_t sample_counter = 0;
    size_t delete_counter = table_size * (1 - del_table_ratio);
    std::vector<std::string> all_index_type;
    std::vector<std::string> all_thread_num;
    std::string index_type;
    std::string keys_file_path;
    std::string keys_file_type;
    std::string sample_distribution;
    bool latency_sample = false;
    double latency_sample_ratio = 0.01;
    int error_bound;
    std::string output_path;
    size_t random_seed;
    bool memory_record;
    bool dataset_statistic;
    bool data_shift = false;
    bool is_first_call = true;
    OperationOrder operation_order = ITERATE_SORT_UNSORTED;
    bool current_batch_sorted = false;

    std::unordered_set<KEY_TYPE> inserted_keys;
    std::unordered_set<KEY_TYPE> deleted_keys;
    std::vector<KEY_TYPE> init_keys;
    KEY_TYPE* keys;
    std::pair<KEY_TYPE, PAYLOAD_TYPE>* init_key_values;
    std::mt19937 gen;
    int iteration_num = 0;
    
    // New: Store all operations organized by iteration
    std::vector<std::vector<std::pair<Operation, KEY_TYPE>>> all_iterations_operations;
    bool all_operations_generated = false;

    struct Stat {
        std::vector<double> read_latency;
        std::vector<double> write_latency;
        uint64_t throughput = 0;
        size_t fitness_of_dataset = 0;
        long long memory_consumption = 0;
        uint64_t success_insert = 0;
        uint64_t success_read = 0;
        uint64_t success_update = 0;
        uint64_t success_remove = 0;
        uint64_t scan_not_enough = 0;

        void clear() {
            read_latency.clear();
            write_latency.clear();
            throughput = 0;
            fitness_of_dataset = 0;
            memory_consumption = 0;
            success_insert = 0;
            success_read = 0;
            success_update = 0;
            success_remove = 0;
            scan_not_enough = 0;
        }
    } stat;

    struct alignas(CACHELINE_SIZE) ThreadParam {
        std::vector<std::pair<uint64_t, uint64_t>> read_latency;
        std::vector<std::pair<uint64_t, uint64_t>> write_latency;
        uint64_t success_insert = 0;
        uint64_t success_read = 0;
        uint64_t success_update = 0;
        uint64_t success_remove = 0;
        uint64_t scan_not_enough = 0;
    };
    typedef ThreadParam param_t;

public:
    Benchmark() {
        init_key_values = nullptr;
    }

    KEY_TYPE* load_keys() {
        COUT_THIS("Reading data from file.");

        if (table_size > 0) keys = new KEY_TYPE[table_size];

        if (keys_file_type == "binary") {
            table_size = load_binary_data(keys, table_size, keys_file_path);
            if (table_size <= 0) {
                COUT_THIS("Could not open key file, please check the path of key file.");
                exit(0);
            }
        } else if (keys_file_type == "text") {
            table_size = load_text_data(keys, table_size, keys_file_path);
            if (table_size <= 0) {
                COUT_THIS("Could not open key file, please check the path of key file.");
                exit(0);
            }
        } else {
            COUT_THIS("Could not open key file, please check the path of key file.");
            exit(0);
        }

        if (!data_shift) {
            tbb::parallel_sort(keys, keys + table_size);
            auto last = std::unique(keys, keys + table_size);
            table_size = last - keys;
            std::shuffle(keys, keys + table_size, gen);
        }

        init_table_size = init_table_ratio * table_size;
        std::cout << "Table size is " << table_size << ", Init table size is " << init_table_size << std::endl;

        for (auto j = 0; j < 10; j++) {
            std::cout << keys[j] << " ";
        }
        std::cout << std::endl;

        COUT_THIS("prepare init keys.");
        init_keys.resize(init_table_size);
#pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < init_table_size; ++i) {
            init_keys[i] = (keys[i]);
        }
        tbb::parallel_sort(init_keys.begin(), init_keys.end());

        init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_keys.size()];
#pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < init_keys.size(); i++) {
            init_key_values[i].first = init_keys[i];
            init_key_values[i].second = 123456789;
        }
        inserted_keys.clear();
        inserted_keys.insert(init_keys.begin(), init_keys.end());

        COUT_VAR(table_size);
        COUT_VAR(init_keys.size());

        return keys;
    }

    inline void prepare(index_t*& index, const KEY_TYPE* keys) {
        index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);
        Param param = Param(thread_num, 0);
        index->init(&param);
        thread_num = param.worker_num;

        COUT_THIS("bulk loading");
        index->bulk_load(init_key_values, init_keys.size(), &param);
        COUT_THIS("finished bulkload");
    }

    inline void parse_args(int argc, char** argv) {
        auto flags = parse_flags(argc, argv);
        keys_file_path = get_required(flags, "keys_file");
        keys_file_type = get_with_default(flags, "keys_file_type", "binary");
        read_ratio = stod(get_required(flags, "read"));
        insert_ratio = stod(get_with_default(flags, "insert", "0"));
        delete_ratio = stod(get_with_default(flags, "delete", "0"));
        update_ratio = stod(get_with_default(flags, "update", "0"));
        scan_ratio = stod(get_with_default(flags, "scan", "0"));
        scan_num = stoi(get_with_default(flags, "scan_num", "100"));
        operations_num = stoi(get_with_default(flags, "operations_num", "1000000000"));
        table_size = stoi(get_with_default(flags, "table_size", "-1"));
        init_table_ratio = stod(get_with_default(flags, "init_table_ratio", "0.5"));
        del_table_ratio = stod(get_with_default(flags, "del_table_ratio", "0.5"));
        init_table_size = 0;
        thread_num = stoi(get_with_default(flags, "thread_num", "1"));
        index_type = get_with_default(flags, "index", "alexol");
        
        // Parse multiple index types and thread counts for nested loops
        std::string all_indices_str = get_with_default(flags, "all_index", "");
        std::string all_threads_str = get_with_default(flags, "all_thread", "");
        
        // Check if we have comma-separated values using get_comma_separated (if available)
        // Otherwise fall back to manual parsing
        try {
            if (!all_indices_str.empty()) {
                // Parse comma-separated index types
                std::stringstream ss(all_indices_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    all_index_type.push_back(item);
                }
            } else {
                // Try get_comma_separated function if available, otherwise use single index
                // all_index_type = get_comma_separated(flags, "index");
                // For now, use single index type
                all_index_type.push_back(index_type);
            }
            
            if (!all_threads_str.empty()) {
                // Parse comma-separated thread counts
                std::stringstream ss(all_threads_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    all_thread_num.push_back(item);
                }
            } else {
                // Try get_comma_separated function if available, otherwise use single thread count
                // all_thread_num = get_comma_separated(flags, "thread_num");
                // For now, use single thread count
                all_thread_num.push_back(std::to_string(thread_num));
            }
        } catch (...) {
            // Fallback to single values if parsing fails
            all_index_type.push_back(index_type);
            all_thread_num.push_back(std::to_string(thread_num));
        }
        sample_distribution = get_with_default(flags, "sample_distribution", "uniform");
        latency_sample = get_boolean_flag(flags, "latency_sample");
        latency_sample_ratio = stod(get_with_default(flags, "latency_sample_ratio", "0.01"));
        error_bound = stoi(get_with_default(flags, "error_bound", "64"));
        output_path = get_with_default(flags, "output_path", "./out.csv");
        random_seed = stoul(get_with_default(flags, "seed", "1866"));
        gen.seed(random_seed);
        memory_record = get_boolean_flag(flags, "memory");
        dataset_statistic = get_boolean_flag(flags, "dataset_statistic");
        data_shift = get_boolean_flag(flags, "data_shift");

        std::string order_str = get_with_default(flags, "operation_order", "iterate");
        if (order_str == "iterate") {
            operation_order = ITERATE_SORT_UNSORTED;
        } else if (order_str == "shuffle") {
            operation_order = SHUFFLE_ALL;
        } else if (order_str == "shuffle_sort") {
            operation_order = SHUFFLE_THEN_SORT;
        } else {
            COUT_THIS("Unknown operation order: " << order_str << ". Using default (iterate).");
        }

        COUT_THIS("[micro] Read:Insert:Update:Scan:Delete= " << read_ratio << ":" << insert_ratio << ":" << update_ratio << ":"
                  << scan_ratio << ":" << delete_ratio);
        double ratio_sum = read_ratio + insert_ratio + delete_ratio + update_ratio + scan_ratio;
        double insert_delete = insert_ratio + delete_ratio;
        INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);
        INVARIANT(sample_distribution == "zipf" || sample_distribution == "uniform");
        INVARIANT(thread_num > 0);
        INVARIANT(all_index_type.size() > 0);
        INVARIANT(all_thread_num.size() > 0);
        
        COUT_THIS("Configured for " << all_index_type.size() << " index types and " 
                  << all_thread_num.size() << " thread configurations");
        
        std::cout << "Index types: ";
        for (const auto& idx : all_index_type) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Thread counts: ";
        for (const auto& thr : all_thread_num) {
            std::cout << thr << " ";
        }
        std::cout << std::endl;
    }

    void generate_operations(KEY_TYPE* keys, std::vector<std::pair<Operation, KEY_TYPE>>& operations, 
                       const std::string& current_index_name) {
        COUT_THIS("Generating operations");
            INVARIANT(delete_ratio * operations_num < init_table_size);

        COUT_THIS("NOW GENERATING....")
        iteration_num++;
        operations.clear();
        operations.reserve(operations_num);
    
        const size_t total_ops = operations_num;
        const size_t read_count = static_cast<size_t>(total_ops * read_ratio);
        const size_t insert_count = static_cast<size_t>(total_ops * insert_ratio);
        const size_t update_count = static_cast<size_t>(total_ops * update_ratio);
        const size_t scan_count = static_cast<size_t>(total_ops * scan_ratio);
        const size_t delete_count = static_cast<size_t>(total_ops * delete_ratio);

        // Track available keys for operations
        std::vector<KEY_TYPE> available_keys_vec(inserted_keys.begin(), inserted_keys.end());
        size_t next_insert_pos = init_table_size;

        // First generate all operations with their keys
        std::vector<std::pair<Operation, KEY_TYPE>> temp_operations;
        temp_operations.reserve(total_ops);
    
    
        // Generate deletes (using existing keys)
        std::unordered_set<KEY_TYPE> keys_to_delete;
        for (size_t i = 0; i < delete_count; i++) {
            if (inserted_keys.empty()) continue;
        
            KEY_TYPE key;
            size_t attempts = 0;
            const size_t max_attempts = 100000000;
            do {
                std::uniform_int_distribution<size_t> dist(0, available_keys_vec.size() - 1);
                size_t idx = dist(gen);
                key = available_keys_vec[idx];
                attempts++;
            } while (keys_to_delete.count(key) && attempts < max_attempts);
        
            if (keys_to_delete.count(key)) continue;
        
            keys_to_delete.insert(key);
            inserted_keys.erase(key);
            deleted_keys.insert(key);
            temp_operations.emplace_back(DELETE, key);
        }
    
        // Create a set of remaining keys (not marked for deletion)
        std::unordered_set<KEY_TYPE> remaining_keys;
        for (const auto& key : available_keys_vec) {
            if (!keys_to_delete.count(key)) {
                remaining_keys.insert(key);
            }
        }
    
        // Convert remaining keys to vector for sampling
        std::vector<KEY_TYPE> remaining_keys_vec(remaining_keys.begin(), remaining_keys.end());

    
        // Pre-generate sample keys using only available keys
        KEY_TYPE* sample_ptr = nullptr;
        if (sample_distribution == "uniform") {
            sample_ptr = get_search_keys(remaining_keys_vec.data(), remaining_keys_vec.size(), 
                                   remaining_keys_vec.size(), &random_seed);
        } else if (sample_distribution == "zipf") {
            sample_ptr = get_search_keys_zipf(remaining_keys_vec.data(), remaining_keys_vec.size(), 
                                        remaining_keys_vec.size(), &random_seed);
        }

        // Generate reads, updates, and scans (using existing keys)
        for (size_t i = 0; i < read_count + update_count + scan_count; i++) {
            if (remaining_keys_vec.empty() || sample_ptr == nullptr) continue;
        
            KEY_TYPE key;
            size_t attempts = 0;
            const size_t max_attempts = 10000000;
            do {
                if (sample_counter >= remaining_keys_vec.size()) sample_counter = 0;
                key = sample_ptr[sample_counter++];
                attempts++;
            } while (deleted_keys.count(key) && attempts < max_attempts);

            if (deleted_keys.count(key)) continue;

            Operation op;
            if (i < read_count) {
                op = READ;
            } else if (i < read_count + update_count) {
                op = UPDATE;
            } else {
                op = SCAN;
            }
            temp_operations.emplace_back(op, key);
        }

        // Generate inserts (using new keys)
        for (size_t i = 0; i < insert_count; i++) {
            if (next_insert_pos >= table_size) continue;
            KEY_TYPE key = keys[next_insert_pos++];
            temp_operations.emplace_back(INSERT, key);
            available_keys_vec.push_back(key);
            inserted_keys.insert(key);
        }



        // Define the comparison function
        auto operation_comp = [](const std::pair<Operation, KEY_TYPE>& a, 
                            const std::pair<Operation, KEY_TYPE>& b) -> bool {
            if (a.first == b.first) {
                return a.second < b.second;
            }
            return a.first < b.first;
        };

        // Now handle the ordering based on the selected option
        switch (operation_order) {
            case ITERATE_SORT_UNSORTED:
                if (current_batch_sorted) {
                    std::sort(temp_operations.begin(), temp_operations.end(), operation_comp);
                } else {
                    std::shuffle(temp_operations.begin(), temp_operations.end(), gen);
                }
                current_batch_sorted = !current_batch_sorted;
                break;
            
            case SHUFFLE_ALL:
                std::shuffle(temp_operations.begin(), temp_operations.end(), gen);
                break;
            
            case SHUFFLE_THEN_SORT:
                std::shuffle(temp_operations.begin(), temp_operations.end(), gen);
                std::sort(temp_operations.begin(), temp_operations.end(), operation_comp);
                break;
        }

        // Copy to final operations vector
        operations = std::move(temp_operations);

        // Create a file to store operations
        std::string filename = std::string("operations_") + current_index_name + "_" + 
                          std::to_string(iteration_num) + ".txt";
        std::ofstream ops_file(filename);
        if (ops_file.is_open()) {
            for (const auto& op : operations) {
                std::string op_str;
                switch(op.first) {
                    case READ: op_str = "READ"; break;
                    case INSERT: op_str = "INSERT"; break;
                    case UPDATE: op_str = "UPDATE"; break;
                    case SCAN: op_str = "SCAN"; break;
                    case DELETE: op_str = "DELETE"; break;
                }
                ops_file << op_str << " " << op.second << "\n";
            }
            ops_file.close();
        }

        // Update state for next iteration
        init_table_size = next_insert_pos;
        if (sample_ptr) delete[] sample_ptr;

        // Verify operation counts
        size_t actual_reads = std::count_if(operations.begin(), operations.end(),
            [](const auto& op) { return op.first == READ; });

        COUT_THIS("Generated operations: " << operations.size());
        COUT_THIS("Reads: " << actual_reads << "/" << read_count);
        COUT_THIS("Inserts: " << insert_count);
        COUT_THIS("Deletes: " << std::count_if(operations.begin(), operations.end(),
            [](const auto& op) { return op.first == DELETE; }) << "/" << delete_count);
        COUT_THIS("UPDATES: " << std::count_if(operations.begin(), operations.end(),
            [](const auto& op) { return op.first == UPDATE; }) << "/" << update_count);
    
        std::string order_used;
        switch (operation_order) {
            case ITERATE_SORT_UNSORTED: 
                order_used = current_batch_sorted ? "sorted" : "shuffled";
                break;
            case SHUFFLE_ALL: 
                order_used = "shuffled";
                break;
            case SHUFFLE_THEN_SORT: 
                order_used = "shuffled_then_sorted";
                break;
        }
        COUT_THIS("Operation order for this batch: " << order_used);
   
        if (!is_first_call) {
            init_keys.clear();
            init_keys.resize(inserted_keys.size());
            std::vector<KEY_TYPE> inserted_keys_vec(inserted_keys.begin(), inserted_keys.end());
            for (size_t i = 0; i < inserted_keys.size(); ++i) {
                init_keys[i] = inserted_keys_vec[i];
            }
            tbb::parallel_sort(init_keys.begin(), init_keys.end());

            init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_keys.size()];
            for (size_t i = 0; i < init_keys.size(); ++i) {
                init_key_values[i].first = init_keys[i];
                init_key_values[i].second = 123456789;
            }
            is_first_call = false;

        }                        
    
        COUT_THIS("done with generate operations function");
    }

    void update_init_keys_and_values_threaded(KEY_TYPE *keys, size_t init_table_size, int thread_num) {
            // Update init_keys with new keys
		    init_keys.clear();
		    init_keys.resize(inserted_keys.size());
		    std::vector<KEY_TYPE> inserted_keys_vec(inserted_keys.begin(), inserted_keys.end());
           //#pragma omp parallel for num_threads(thread_num)
		    for (size_t i = 0; i < inserted_keys.size(); ++i) {
		        init_keys[i] = inserted_keys_vec[i];
		    }

		    tbb::parallel_sort(init_keys.begin(), init_keys.end());

		    init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_keys.size()];

            //#pragma omp parallel for num_threads(thread_num)
		    for (size_t i = 0; i < init_keys.size(); ++i) {
		        init_key_values[i].first = init_keys[i];
		        init_key_values[i].second = 123456789;
		    }
	}

    void run(index_t* index, std::vector<std::pair<Operation, KEY_TYPE>>& operations) {
        std::thread* thread_array = new std::thread[thread_num];
        param_t params[thread_num];
        TSCNS tn;
        tn.init();
        printf("Begin running\n");
        auto start_time = tn.rdtsc();
        auto end_time = tn.rdtsc();

        #pragma omp parallel num_threads(thread_num)
        {
            auto thread_id = omp_get_thread_num();
            auto paramI = Param(thread_num, thread_id);
            int latency_sample_interval = operations_num / (operations_num * latency_sample_ratio);
            auto latency_sample_start_time = tn.rdtsc();
            auto latency_sample_end_time = tn.rdtsc();
            param_t& thread_param = params[thread_id];
            thread_param.read_latency.reserve(operations_num / latency_sample_interval);
            thread_param.write_latency.reserve(operations_num / latency_sample_interval);
            PAYLOAD_TYPE val;
            std::pair<KEY_TYPE, PAYLOAD_TYPE>* scan_result = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[scan_num];

            #pragma omp barrier
            #pragma omp master
            start_time = tn.rdtsc();

            #pragma omp for schedule(dynamic, 10000)
            for (auto i = 0; i < operations_num; i++) {
                auto op = operations[i].first;
                auto key = operations[i].second;

                if (latency_sample && i % latency_sample_interval == 0)
                    latency_sample_start_time = tn.rdtsc();

                if (op == READ) {
                    auto ret = index->get(key, val, &paramI);
                    thread_param.success_read += ret;
                    if (latency_sample && i % latency_sample_interval == 0) {
                        latency_sample_end_time = tn.rdtsc();
                        thread_param.read_latency.push_back(std::make_pair(latency_sample_start_time, latency_sample_end_time));
                    }
                } else if (op == INSERT) {
                    auto ret = index->put(key, 123456789, &paramI);
                    thread_param.success_insert += ret;
                    if (latency_sample && i % latency_sample_interval == 0) {
                        latency_sample_end_time = tn.rdtsc();
                        thread_param.write_latency.push_back(std::make_pair(latency_sample_start_time, latency_sample_end_time));
                    }
                } else if (op == UPDATE) {
                    auto ret = index->update(key, 234567891, &paramI);
                    thread_param.success_update += ret;
                     if (latency_sample && i % latency_sample_interval == 0) {
                        latency_sample_end_time = tn.rdtsc();
                        thread_param.write_latency.push_back(std::make_pair(latency_sample_start_time, latency_sample_end_time));
                    }
                } else if (op == SCAN) {
                    auto scan_len = index->scan(key, scan_num, scan_result, &paramI);
                    if (scan_len != scan_num) {
                        thread_param.scan_not_enough++;
                    }
                } else if (op == DELETE) {
                    auto ret = index->remove(key, &paramI);
                    thread_param.success_remove += ret;
                     if (latency_sample && i % latency_sample_interval == 0) {
                        latency_sample_end_time = tn.rdtsc();
                        thread_param.write_latency.push_back(std::make_pair(latency_sample_start_time, latency_sample_end_time));
                    }
                }
            }
            #pragma omp master
            end_time = tn.rdtsc();
        }

        auto diff = tn.tsc2ns(end_time) - tn.tsc2ns(start_time);
        printf("Finish running\n");

        for (auto& p : params) {
            if (latency_sample) {
                for (auto e : p.read_latency) {
                    stat.read_latency.push_back(tn.tsc2ns(e.second) - tn.tsc2ns(e.first));
                }
                for (auto e : p.write_latency) {
                    stat.write_latency.push_back(tn.tsc2ns(e.second) - tn.tsc2ns(e.first));
                }
            }
            stat.success_read += p.success_read;
            stat.success_insert += p.success_insert;
            stat.success_update += p.success_update;
            stat.success_remove += p.success_remove;
            stat.scan_not_enough += p.scan_not_enough;
        }

        stat.throughput = static_cast<uint64_t>(operations.size() / (diff/(double) 1000000000));

        if (dataset_statistic) {
            // Calculate PGM metric on all keys currently in the index (not just init_keys)
            std::vector<KEY_TYPE> current_keys(inserted_keys.begin(), inserted_keys.end());
            tbb::parallel_sort(current_keys.begin(), current_keys.end());
            stat.fitness_of_dataset = pgmMetric::PGM_metric(current_keys.data(), current_keys.size(), error_bound);
        }

        if (memory_record)
            stat.memory_consumption = index->memory_consumption();

        print_stat();
        COUT_THIS("done with print stat function");

        delete[] thread_array;
        COUT_THIS("done with run function");
    }

    void print_stat(bool header = false, bool clear_flag = true) {
        double avg_read_latency = 0;
        if (!stat.read_latency.empty()) {
        for (auto t : stat.read_latency) {
            avg_read_latency += t;
        }
        avg_read_latency /= stat.read_latency.size();
    }

    double avg_write_latency = 0;
    if (!stat.write_latency.empty()) {
        for (auto t : stat.write_latency) {
            avg_write_latency += t;
        }
        avg_write_latency /= stat.write_latency.size();
    }

        double read_latency_variance = 0;
        double write_latency_variance = 0;
        
        if (latency_sample) {
        if (!stat.read_latency.empty()) {
            for (auto t : stat.read_latency) {
                read_latency_variance += (t - avg_read_latency) * (t - avg_read_latency);
            }
            read_latency_variance /= stat.read_latency.size();
            std::sort(stat.read_latency.begin(), stat.read_latency.end());
        }

        if (!stat.write_latency.empty()) {
            for (auto t : stat.write_latency) {
                write_latency_variance += (t - avg_write_latency) * (t - avg_write_latency);
            }
            write_latency_variance /= stat.write_latency.size();
            std::sort(stat.write_latency.begin(), stat.write_latency.end());
        }
    }

        printf("Throughput = %llu\n", stat.throughput);
        printf("Memory: %lld\n", stat.memory_consumption);
        printf("success_read: %llu\n", stat.success_read);
        printf("success_insert: %llu\n", stat.success_insert);
        printf("success_update: %llu\n", stat.success_update);
        printf("success_remove: %llu\n", stat.success_remove);
        printf("scan_not_enough: %llu\n", stat.scan_not_enough);

        std::time_t t = std::time(nullptr);
        char time_str[100];

        if (!file_exists(output_path)) {
            std::ofstream ofile;
            ofile.open(output_path, std::ios::app);
            ofile << "id" << ",";
            ofile << "read_ratio" << "," << "insert_ratio" << "," << "update_ratio" << "," << "scan_ratio" << "," << "delete_ratio" << ",";
            ofile << "key_path" << ",";
            ofile << "index_type" << ",";
            ofile << "throughput" << ",";
            ofile << "init_table_size" << ",";
            ofile << "memory_consumption" << ",";
            ofile << "thread_num" << ",";
            ofile << "read_min" << ",";
            ofile << "read_50 percentile" << ",";
            ofile << "read_90 percentile" << ",";
            ofile << "read_99 percentile" << ",";
            ofile << "read_99.9 percentile" << ",";
            ofile << "read_99.99 percentile" << ",";
            ofile << "read_max" << ",";
            ofile << "read_avg" << ",";
            ofile << "write_min" << ",";
            ofile << "write_50 percentile" << ",";
            ofile << "write_90 percentile" << ",";
            ofile << "write_99 percentile" << ",";
            ofile << "write_99.9 percentile" << ",";
            ofile << "write_99.99 percentile" << ",";
            ofile << "write_max" << ",";
            ofile << "write_avg" << ",";
            ofile << "seed" << ",";
            ofile << "scan_num" << ",";
            ofile << "read_latency_variance" << ",";
            ofile << "write_latency_variance" << ",";
            ofile << "latency_sample" << ",";
            ofile << "data_shift" << ",";
            ofile << "pgm" << ",";
            ofile << "error_bound" ",";
            ofile << "table_size" << std::endl;
        }

        std::ofstream ofile;
        
        ofile.open(output_path, std::ios::app);
        if (!ofile) {
            std::cerr << "Error opening output file: " << output_path << std::endl;
            return;
        }
        if (std::strftime(time_str, sizeof(time_str), "%Y%m%d%H%M%S", std::localtime(&t))) {
            ofile << time_str << ',';
        }
        ofile << read_ratio << "," << insert_ratio << "," << update_ratio << "," << scan_ratio << "," << delete_ratio << ",";

        ofile << keys_file_path << ",";
        ofile << index_type << ",";
        ofile << stat.throughput << ",";
        ofile << init_table_size << ",";
        ofile << stat.memory_consumption << ",";
        ofile << thread_num << ",";
        if (latency_sample) {
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[0]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[0.5 * stat.read_latency.size()]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[0.9 * stat.read_latency.size()]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[0.99 * stat.read_latency.size()]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[0.999 * stat.read_latency.size()]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[0.9999 * stat.read_latency.size()]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : stat.read_latency[stat.read_latency.size() - 1]) << ",";
            ofile << (stat.read_latency.empty() ? 0 : avg_read_latency) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[0]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[0.5 * stat.write_latency.size()]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[0.9 * stat.write_latency.size()]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[0.99 * stat.write_latency.size()]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[0.999 * stat.write_latency.size()]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[0.9999 * stat.write_latency.size()]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : stat.write_latency[stat.write_latency.size() - 1]) << ",";
            ofile << (stat.write_latency.empty() ? 0 : avg_write_latency) << ",";
        } else {
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
        }
        ofile << random_seed << ",";
        ofile << scan_num << ",";
        ofile << (stat.read_latency.empty() ? 0 : read_latency_variance) << ",";
        ofile << (stat.write_latency.empty() ? 0 : write_latency_variance) << ",";
        ofile << latency_sample << ",";
        ofile << data_shift << ",";
        ofile << stat.fitness_of_dataset << ",";
        ofile << error_bound << ",";
        ofile << table_size << std::endl;
        ofile.close();

        if (clear_flag) stat.clear();
    }

    void run_benchmark() {
        load_keys();
        
        // Store original init state for resets
        std::vector<KEY_TYPE> original_init_keys = init_keys;
        std::unordered_set<KEY_TYPE> original_inserted_keys = inserted_keys;
        
        // Calculate number of runs (same for all index/thread combinations)
        int n_runs = (table_size - (table_size * init_table_ratio)) / (operations_num * insert_ratio);
        if (read_ratio == 1.0) n_runs = 2;
        
        // Generate all operations for all iterations ONCE before any index/thread loops
        // This ensures the same operations are used for all index types and thread counts
        COUT_THIS("Generating operations for " << n_runs << " iterations to be used across all index/thread combinations");
        generate_all_iterations_operations(keys, n_runs, "shared");
        
        // Nested loops over all index types and thread counts
        for (const std::string& current_index_type : all_index_type) {
            for (const std::string& current_thread_str : all_thread_num) {
                size_t current_thread_count = std::stoul(current_thread_str);
                
                COUT_THIS("Running benchmark for index: " << current_index_type 
                         << ", threads: " << current_thread_count << " (using shared pre-generated operations)");
                
                // Set current parameters
                index_type = current_index_type;
                thread_num = current_thread_count;
                
                // Reset state for this index/thread combination
                reset_state_for_new_index();
                
                // Restore original init state - THIS MUST COME AFTER RESET
                init_keys = original_init_keys;
                inserted_keys = original_inserted_keys;
                init_table_size = original_init_keys.size();
                
                // CRITICAL: Rebuild init_key_values to match the restored init_keys
                if (init_key_values != nullptr) {
                    delete[] init_key_values;  // Clean up previous allocation
                }
                init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_keys.size()];
                for (size_t i = 0; i < init_keys.size(); ++i) {
                    init_key_values[i].first = init_keys[i];
                    init_key_values[i].second = 123456789;
                }
                
                COUT_THIS("Restored original state: " << inserted_keys.size() << " inserted keys");
                COUT_THIS("Rebuilt init_key_values with " << init_keys.size() << " key-value pairs");
                
                // Run first iteration using pre-generated operations
                COUT_THIS("Starting iteration 1 of " << n_runs 
                         << " for " << index_type << " with " << thread_num << " threads (using shared pre-generated operations)");
                std::vector<std::pair<Operation, KEY_TYPE>>& operations = get_operations_for_iteration(0);
                
                COUT_THIS("First iteration operations summary:");
                COUT_THIS("  Total operations: " << operations.size());
                COUT_THIS("  Reads: " << std::count_if(operations.begin(), operations.end(), [](const auto& op) { return op.first == READ; }));
                COUT_THIS("  Inserts: " << std::count_if(operations.begin(), operations.end(), [](const auto& op) { return op.first == INSERT; }));
                COUT_THIS("  Updates: " << std::count_if(operations.begin(), operations.end(), [](const auto& op) { return op.first == UPDATE; }));
                COUT_THIS("  Deletes: " << std::count_if(operations.begin(), operations.end(), [](const auto& op) { return op.first == DELETE; }));
                COUT_THIS("  Current inserted_keys size: " << inserted_keys.size());
                COUT_THIS("  Current init_keys size: " << init_keys.size());
                
                index_t* index;
                prepare(index, keys);
                run(index, operations);
                
                // Update real state based on executed operations
                update_state_after_iteration(operations);
                
                // Update init_table_size to reflect newly inserted keys
                init_table_size = inserted_keys.size();
                COUT_THIS("Updated init_table_size to: " << init_table_size);
                
                update_init_keys_and_values_threaded(keys, init_table_size, thread_num);
                
                // Run remaining iterations using pre-generated operations
                for (int iter_idx = 1; iter_idx < n_runs; ++iter_idx) {
                    COUT_THIS("Starting iteration " << (iter_idx + 1) << " of " << n_runs 
                             << " for " << index_type << " with " << thread_num << " threads (using shared pre-generated operations)");
                    std::vector<std::pair<Operation, KEY_TYPE>>& operations_iter = get_operations_for_iteration(iter_idx);
                    run(index, operations_iter);
                    
                    // Update real state based on executed operations
                    update_state_after_iteration(operations_iter);
                    
                    // Update init_table_size to reflect newly inserted keys
                    init_table_size = inserted_keys.size();
                    COUT_THIS("Updated init_table_size to: " << init_table_size);
                }
                
                if (index != nullptr) {
                    delete index;
                    index = nullptr;
                }
                
                COUT_THIS("Completed benchmark for " << index_type << " with " << thread_num << " threads");
            }
        }
    }                

    // Function to update real state after executing operations for an iteration
    void update_state_after_iteration(const std::vector<std::pair<Operation, KEY_TYPE>>& executed_operations) {
        for (const auto& op : executed_operations) {
            const Operation operation = op.first;
            const KEY_TYPE key = op.second;
            
            switch (operation) {
                case INSERT:
                    inserted_keys.insert(key);
                    deleted_keys.erase(key); // Remove from deleted if it was there
                    break;
                case DELETE:
                    inserted_keys.erase(key);
                    deleted_keys.insert(key);
                    break;
                case READ:
                case UPDATE:
                case SCAN:
                    // These don't change the key state
                    break;
            }
        }
        
        // Update iteration number to match what the original logic would do
        iteration_num++;
        
        COUT_THIS("Updated real state after iteration " << iteration_num);
        COUT_THIS("Currently inserted keys: " << inserted_keys.size());
        COUT_THIS("Currently deleted keys: " << deleted_keys.size());
    }

    // Function to reset benchmark state between different index/thread combinations
    void reset_state_for_new_index() {
        // DO NOT reset operation generation state - we want to reuse the same operations
        // all_operations_generated = false;
        // all_iterations_operations.clear();
        
        // Reset key tracking state (restore to initial state after load_keys)
        inserted_keys.clear();
        deleted_keys.clear();
        
        // Reset counters and execution state to match the generation state
        sample_counter = 0;
        iteration_num = 0;
        is_first_call = true;
        current_batch_sorted = false;
        
        // Reset statistics
        stat.clear();
        
        COUT_THIS("Reset state for new index/thread combination");
        COUT_THIS("Keeping shared pre-generated operations intact");
    }

    void generate_all_iterations_operations(KEY_TYPE* keys, int n_runs, const std::string& current_index_name) {
        COUT_THIS("Generating all operations for all iterations at start");
        
        all_iterations_operations.clear();
        all_iterations_operations.resize(n_runs);
        
        // Simulate state for each iteration - create working copies
        std::unordered_set<KEY_TYPE> sim_inserted_keys = inserted_keys;
        std::unordered_set<KEY_TYPE> sim_deleted_keys = deleted_keys;
        std::vector<KEY_TYPE> sim_init_keys = init_keys;
        size_t sim_init_table_size = init_table_size;
        size_t sim_sample_counter = sample_counter;
        int sim_iteration_num = iteration_num;
        bool sim_is_first_call = is_first_call;
        bool sim_current_batch_sorted = current_batch_sorted;
        
        for (int iter = 0; iter < n_runs; ++iter) {
            COUT_THIS("Simulating operation generation for iteration " << (iter + 1) << " of " << n_runs);
            
            // Generate operations for this iteration using simulated state
            std::vector<std::pair<Operation, KEY_TYPE>>& iter_operations = all_iterations_operations[iter];
            
            INVARIANT(delete_ratio * operations_num < sim_init_table_size);
            
            sim_iteration_num++;
            iter_operations.clear();
            iter_operations.reserve(operations_num);
            
            const size_t total_ops = operations_num;
            const size_t read_count = static_cast<size_t>(total_ops * read_ratio);
            const size_t insert_count = static_cast<size_t>(total_ops * insert_ratio);
            const size_t update_count = static_cast<size_t>(total_ops * update_ratio);
            const size_t scan_count = static_cast<size_t>(total_ops * scan_ratio);
            const size_t delete_count = static_cast<size_t>(total_ops * delete_ratio);
            
            // Track available keys for operations using simulated state
            std::vector<KEY_TYPE> available_keys_vec(sim_inserted_keys.begin(), sim_inserted_keys.end());
            size_t next_insert_pos = sim_init_table_size;
            
            // First generate all operations with their keys
            std::vector<std::pair<Operation, KEY_TYPE>> temp_operations;
            temp_operations.reserve(total_ops);
            
            // Generate deletes (using existing keys)
            std::unordered_set<KEY_TYPE> keys_to_delete;
            for (size_t i = 0; i < delete_count; i++) {
                if (sim_inserted_keys.empty()) continue;
                
                KEY_TYPE key;
                size_t attempts = 0;
                const size_t max_attempts = 100000000;
                do {
                    std::uniform_int_distribution<size_t> dist(0, available_keys_vec.size() - 1);
                    size_t idx = dist(gen);
                    key = available_keys_vec[idx];
                    attempts++;
                } while (keys_to_delete.count(key) && attempts < max_attempts);
                
                if (keys_to_delete.count(key)) continue;
                
                keys_to_delete.insert(key);
                sim_inserted_keys.erase(key);
                sim_deleted_keys.insert(key);
                temp_operations.emplace_back(DELETE, key);
            }
            
            // Create a set of remaining keys (not marked for deletion)
            std::unordered_set<KEY_TYPE> remaining_keys;
            for (const auto& key : available_keys_vec) {
                if (!keys_to_delete.count(key)) {
                    remaining_keys.insert(key);
                }
            }
            
            // Convert remaining keys to vector for sampling
            std::vector<KEY_TYPE> remaining_keys_vec(remaining_keys.begin(), remaining_keys.end());
            
            // Pre-generate sample keys using only available keys
            KEY_TYPE* sample_ptr = nullptr;
            if (sample_distribution == "uniform") {
                sample_ptr = get_search_keys(remaining_keys_vec.data(), remaining_keys_vec.size(), 
                                       remaining_keys_vec.size(), &random_seed);
            } else if (sample_distribution == "zipf") {
                sample_ptr = get_search_keys_zipf(remaining_keys_vec.data(), remaining_keys_vec.size(), 
                                            remaining_keys_vec.size(), &random_seed);
            }
            
            // Generate reads, updates, and scans (using existing keys)
            for (size_t i = 0; i < read_count + update_count + scan_count; i++) {
                if (remaining_keys_vec.empty() || sample_ptr == nullptr) continue;
                
                KEY_TYPE key;
                size_t attempts = 0;
                const size_t max_attempts = 10000000;
                do {
                    if (sim_sample_counter >= remaining_keys_vec.size()) sim_sample_counter = 0;
                    key = sample_ptr[sim_sample_counter++];
                    attempts++;
                } while (sim_deleted_keys.count(key) && attempts < max_attempts);
                
                if (sim_deleted_keys.count(key)) continue;
                
                Operation op;
                if (i < read_count) {
                    op = READ;
                } else if (i < read_count + update_count) {
                    op = UPDATE;
                } else {
                    op = SCAN;
                }
                temp_operations.emplace_back(op, key);
            }
            
            // Generate inserts (using new keys)
            for (size_t i = 0; i < insert_count; i++) {
                if (next_insert_pos >= table_size) continue;
                KEY_TYPE key = keys[next_insert_pos++];
                temp_operations.emplace_back(INSERT, key);
                available_keys_vec.push_back(key);
                sim_inserted_keys.insert(key);
            }
            
            // Define the comparison function
            auto operation_comp = [](const std::pair<Operation, KEY_TYPE>& a, 
                                const std::pair<Operation, KEY_TYPE>& b) -> bool {
                if (a.first == b.first) {
                    return a.second < b.second;
                }
                return a.first < b.first;
            };
            
            // Now handle the ordering based on the selected option
            switch (operation_order) {
                case ITERATE_SORT_UNSORTED:
                    if (sim_current_batch_sorted) {
                        std::sort(temp_operations.begin(), temp_operations.end(), operation_comp);
                    } else {
                        std::shuffle(temp_operations.begin(), temp_operations.end(), gen);
                    }
                    sim_current_batch_sorted = !sim_current_batch_sorted;
                    break;
                
                case SHUFFLE_ALL:
                    std::shuffle(temp_operations.begin(), temp_operations.end(), gen);
                    break;
                
                case SHUFFLE_THEN_SORT:
                    std::shuffle(temp_operations.begin(), temp_operations.end(), gen);
                    std::sort(temp_operations.begin(), temp_operations.end(), operation_comp);
                    break;
            }
            
            // Copy to iteration operations
            iter_operations = std::move(temp_operations);
            
            // Create a file to store operations for this iteration
            std::string filename = std::string("operations_") + current_index_name + "_" + 
                              std::to_string(sim_iteration_num) + ".txt";
            std::ofstream ops_file(filename);
            if (ops_file.is_open()) {
                for (const auto& op : iter_operations) {
                    std::string op_str;
                    switch(op.first) {
                        case READ: op_str = "READ"; break;
                        case INSERT: op_str = "INSERT"; break;
                        case UPDATE: op_str = "UPDATE"; break;
                        case SCAN: op_str = "SCAN"; break;
                        case DELETE: op_str = "DELETE"; break;
                    }
                    ops_file << op_str << " " << op.second << "\n";
                }
                ops_file.close();
            }
            
            // Update simulated state for next iteration
            sim_init_table_size = next_insert_pos;
            if (sample_ptr) delete[] sample_ptr;
            
            // Verify operation counts
            size_t actual_reads = std::count_if(iter_operations.begin(), iter_operations.end(),
                [](const auto& op) { return op.first == READ; });
            
            COUT_THIS("Iteration " << sim_iteration_num << " generated operations: " << iter_operations.size());
            COUT_THIS("Reads: " << actual_reads << "/" << read_count);
            COUT_THIS("Inserts: " << insert_count);
            COUT_THIS("Deletes: " << std::count_if(iter_operations.begin(), iter_operations.end(),
                [](const auto& op) { return op.first == DELETE; }) << "/" << delete_count);
            COUT_THIS("UPDATES: " << std::count_if(iter_operations.begin(), iter_operations.end(),
                [](const auto& op) { return op.first == UPDATE; }) << "/" << update_count);
            
            std::string order_used;
            switch (operation_order) {
                case ITERATE_SORT_UNSORTED: 
                    order_used = sim_current_batch_sorted ? "sorted" : "shuffled";
                    break;
                case SHUFFLE_ALL: 
                    order_used = "shuffled";
                    break;
                case SHUFFLE_THEN_SORT: 
                    order_used = "shuffled_then_sorted";
                    break;
            }
            COUT_THIS("Operation order for iteration " << sim_iteration_num << ": " << order_used);
            
            // Update simulated init_keys for next iteration (like the original logic)
            if (!sim_is_first_call) {
                sim_init_keys.clear();
                sim_init_keys.resize(sim_inserted_keys.size());
                std::vector<KEY_TYPE> inserted_keys_vec(sim_inserted_keys.begin(), sim_inserted_keys.end());
                for (size_t i = 0; i < sim_inserted_keys.size(); ++i) {
                    sim_init_keys[i] = inserted_keys_vec[i];
                }
                tbb::parallel_sort(sim_init_keys.begin(), sim_init_keys.end());
                sim_is_first_call = false;
            }
        }
        
        all_operations_generated = true;
        COUT_THIS("Finished generating all operations for " << n_runs << " iterations");
    }

    // Function to get operations for a specific iteration
    std::vector<std::pair<Operation, KEY_TYPE>>& get_operations_for_iteration(int iteration_index) {
        if (!all_operations_generated) {
            throw std::runtime_error("Operations not generated. Call generate_all_iterations_operations first.");
        }
        if (iteration_index < 0 || iteration_index >= static_cast<int>(all_iterations_operations.size())) {
            std::stringstream ss;
            ss << "Invalid iteration index " << iteration_index 
               << ". Valid range: 0 to " << (all_iterations_operations.size() - 1)
               << " (total iterations: " << all_iterations_operations.size() << ")";
            throw std::runtime_error(ss.str());
        }
        return all_iterations_operations[iteration_index];
    }
};