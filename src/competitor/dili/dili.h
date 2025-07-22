#pragma once

#include "../indexInterface.h"
#include "src/src/dili/DILI.h"

// Undefine DILI macros to prevent conflicts with benchmark framework
#ifdef KEY_TYPE
#undef KEY_TYPE
#endif
#ifdef PAYLOAD_TYPE
#undef PAYLOAD_TYPE
#endif

template<class KeyType, class PayloadType>
class diliInterface : public indexInterface<KeyType, PayloadType> {
public:
  void init(Param *param = nullptr) {
    // Set mirror directory for DILI
    index.set_mirror_dir("/tmp/dili_mirror");
  }

  void bulk_load(std::pair <KeyType, PayloadType> *key_value, size_t num, Param *param = nullptr);

  bool get(KeyType key, PayloadType &val, Param *param = nullptr);

  bool put(KeyType key, PayloadType value, Param *param = nullptr);

  bool update(KeyType key, PayloadType value, Param *param = nullptr);

  bool remove(KeyType key, Param *param = nullptr);

  size_t scan(KeyType key_low_bound, size_t key_num, std::pair<KeyType, PayloadType> *result,
              Param *param = nullptr);

  long long memory_consumption() { return static_cast<long long>(index.total_size()); }

private:
  DILI index;
};

template<class KeyType, class PayloadType>
void diliInterface<KeyType, PayloadType>::bulk_load(std::pair <KeyType, PayloadType> *key_value, size_t num,
                                                      Param *param) {
  // Convert to the format expected by DILI (long, long pairs)
  std::vector<std::pair<long, long>> data;
  data.reserve(num);
  for (size_t i = 0; i < num; i++) {
    data.emplace_back(static_cast<long>(key_value[i].first), static_cast<long>(key_value[i].second));
  }
  index.bulk_load(data);
}

template<class KeyType, class PayloadType>
bool diliInterface<KeyType, PayloadType>::get(KeyType key, PayloadType &val, Param *param) {
  long result = index.search(static_cast<long>(key));
  if (result >= 0) {  // DILI returns -1 for not found
    val = static_cast<PayloadType>(result);
    return true;
  }
  return false;
}

template<class KeyType, class PayloadType>
bool diliInterface<KeyType, PayloadType>::put(KeyType key, PayloadType value, Param *param) {
  return index.insert(static_cast<long>(key), static_cast<long>(value));
}

template<class KeyType, class PayloadType>
bool diliInterface<KeyType, PayloadType>::update(KeyType key, PayloadType value, Param *param) {
  // DILI doesn't have direct update, so do delete + insert
  long k = static_cast<long>(key);
  long v = static_cast<long>(value);
  recordPtr old_ptr = index.delete_key(k);
  if (old_ptr >= 0) {
    return index.insert(k, v);
  }
  return false;
}

template<class KeyType, class PayloadType>
bool diliInterface<KeyType, PayloadType>::remove(KeyType key, Param *param) {
  recordPtr deleted_ptr = index.delete_key(static_cast<long>(key));
  return deleted_ptr >= 0;
}

template<class KeyType, class PayloadType>
size_t diliInterface<KeyType, PayloadType>::scan(KeyType key_low_bound, size_t key_num,
                                                   std::pair<KeyType, PayloadType> *result,
                                                   Param *param) {
  // DILI range_query signature: range_query(k1, k2, recordPtr *ptrs)
  // We need to find appropriate upper bound and then convert results
  long k1 = static_cast<long>(key_low_bound);
  long k2 = k1 + static_cast<long>(key_num);  // Simple approach
  
  recordPtr* ptrs = new recordPtr[key_num];
  int actual_results = index.range_query(k1, k2, ptrs);
  
  // Convert results to the expected format
  size_t scan_size = std::min(static_cast<size_t>(actual_results), key_num);
  for (size_t i = 0; i < scan_size; i++) {
    result[i] = {static_cast<KeyType>(k1 + i), static_cast<PayloadType>(ptrs[i])}; // Approximate - DILI doesn't return keys directly
  }
  
  delete[] ptrs;
  return scan_size;
}

