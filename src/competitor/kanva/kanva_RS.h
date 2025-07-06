// #include "./src/osm/include/function.h"
#include "./src/Kanva_RS/kanva_RS.h"
#include "../indexInterface.h"
#include "./src/common/util.h"
#include <omp.h>
#include <atomic>
#include <memory>
#include <iostream>

template<class KEY_TYPE, class PAYLOAD_TYPE>
class kanvaInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  kanvaInterface() : index(nullptr), max_threads(1) {} // Use the single parameter constructor with 1 thread
  ~kanvaInterface() {
    if (index) {
      delete index;
    }
  }
  
  void init(Param *param = nullptr) {
    if (param != nullptr) {
      max_threads = param->worker_num;
    } else {
      max_threads = 64; // fallback default
    }

    // Clean up existing index if any
    if (index) {
      delete index;
    }

    // Create new index with the correct thread count
    index = new kanva_RS::Kanva_RS<KEY_TYPE, PAYLOAD_TYPE>(max_threads);

    // Initialize thread tracking array
    thread_initialized.reset(new std::atomic<bool>[max_threads]);
    for (int i = 0; i < max_threads; i++) {
      thread_initialized[i].store(false);
    }
   }

  void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr);

  long long memory_consumption() { return 0; }

private:
  kanva_RS::Kanva_RS<KEY_TYPE, PAYLOAD_TYPE>* index;
  std::unique_ptr<std::atomic<bool>[]> thread_initialized;
  int max_threads;
  inline int get_thread_id() {
    return omp_get_thread_num();
  }
  inline void ensure_thread_initialized() {
    int tid = get_thread_id();
    if (tid >= max_threads || tid < 0) {
      // This should not happen, but if it does, print error and abort
      std::cerr << "FATAL ERROR: Invalid thread ID " << tid << " (max_threads=" << max_threads << ")" << std::endl;
      std::cerr << "This indicates a serious threading issue. Aborting to prevent corruption." << std::endl;
      abort(); // Fail fast rather than corrupt memory
    }
    if (thread_initialized[tid].load()) {
      return; // Already initialized
    }
    //if (tid < max_threads && !thread_initialized[tid].load()) {
    bool expected = false;
    if (thread_initialized[tid].compare_exchange_strong(expected, true)) {
      // Initialize record managers for this thread
      index->llRecMgr->initThread(tid);
      index->vnodeRecMgr->initThread(tid);
      // Note: thread_initialized[tid] is already set to true by compare_exchange_strong
    }
  }
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
  ensure_thread_initialized();
  std::vector<KEY_TYPE> key_temp;
  std::vector<PAYLOAD_TYPE> val_temp;
  key_temp.reserve(num);
  val_temp.reserve(num);
  for (size_t i = 0; i < num; i++) {
      key_temp.push_back(key_value[i].first);
      val_temp.push_back(key_value[i].second);
  }
  index->train(key_temp, val_temp, 32, get_thread_id()); // Add thread_id_t parameter (0)
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  ensure_thread_initialized();
  kanva_RS::result_t res = index->find(key, val, get_thread_id());
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  ensure_thread_initialized();
  kanva_RS::result_t res = index->insert(key, value, get_thread_id());
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  ensure_thread_initialized();
  kanva_RS::result_t res = index->update(key, value, get_thread_id());
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
 ensure_thread_initialized();
 kanva_RS::result_t res = index->remove(key, get_thread_id());
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
  ensure_thread_initialized();
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> res;
  res.reserve(key_num);
  size_t scan_size = index->rangequery(key_low_bound, key_num, res, get_thread_id());
  return scan_size;
}
