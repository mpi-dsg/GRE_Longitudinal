#include "dili.h"

// Template specializations for the types used in the benchmark
template<class KEY_TYPE, class PAYLOAD_TYPE>
void diliInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
  // Convert to DILI's expected format
  std::vector<std::pair<long, long>> bulk_data;
  bulk_data.reserve(num);
  
  for (size_t i = 0; i < num; i++) {
    bulk_data.emplace_back(static_cast<long>(key_value[i].first), 
                          static_cast<long>(key_value[i].second));
  }
  
  index.bulk_load(bulk_data);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  long dili_key = static_cast<long>(key);
  long result = index.search(dili_key);
  
  if (result != -1) {
    val = static_cast<PAYLOAD_TYPE>(result);
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  long dili_key = static_cast<long>(key);
  long dili_value = static_cast<long>(value);
  return index.insert(dili_key, dili_value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  // DILI doesn't have explicit update, so we use insert which will overwrite
  long dili_key = static_cast<long>(key);
  long dili_value = static_cast<long>(value);
  return index.insert(dili_key, dili_value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  long dili_key = static_cast<long>(key);
  return index.erase(dili_key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t diliInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
  // DILI's range_query requires both lower and upper bounds
  // We'll estimate the upper bound based on the key range
  long dili_key_low = static_cast<long>(key_low_bound);
  long dili_key_high = dili_key_low + static_cast<long>(key_num * 100); // Heuristic upper bound
  
  // Allocate temporary storage for DILI's result format
  std::vector<long> temp_results;
  temp_results.resize(key_num);
  
  int actual_count = index.range_query(dili_key_low, dili_key_high, temp_results.data());
  
  // Convert results back to expected format
  size_t converted_count = 0;
  for (int i = 0; i < actual_count && converted_count < key_num; i++) {
    if (temp_results[i] != -1) {
      result[converted_count].first = static_cast<KEY_TYPE>(dili_key_low + i);
      result[converted_count].second = static_cast<PAYLOAD_TYPE>(temp_results[i]);
      converted_count++;
    }
  }
  
  return converted_count;
}

// Explicit template instantiation for the types used in the benchmark
template class diliInterface<uint64_t, uint64_t>;
template class diliInterface<int64_t, int64_t>;
