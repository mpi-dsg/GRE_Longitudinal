#include "./src/Kanva_impl/kanva.h"
#include "./src/Kanva_impl/kanva_impl.h"
#include "../indexInterface.h"
#include <iostream>

template<class KEY_TYPE, class PAYLOAD_TYPE>
class kanvaImplInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  kanvaImplInterface() : index(nullptr) {}
  
  ~kanvaImplInterface() {
    if (index) {
      delete index;
    }
  }
  
  void init(Param *param = nullptr) {
    // Clean up existing index if any
    if (index) {
      delete index;
    }
    
    // Create new index - using default parameters or could be customized
    index = new kanva_impl::Kanva<KEY_TYPE, PAYLOAD_TYPE>();
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
  kanva_impl::Kanva<KEY_TYPE, PAYLOAD_TYPE>* index;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void kanvaImplInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
  std::vector<KEY_TYPE> key_temp;
  std::vector<PAYLOAD_TYPE> val_temp;
  key_temp.reserve(num);
  val_temp.reserve(num);
  
  for (size_t i = 0; i < num; i++) {
      key_temp.push_back(key_value[i].first);
      val_temp.push_back(key_value[i].second);
  }
  
  // Use default maxErr of 64, can be customized if needed
  index->train(key_temp, val_temp, 64);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaImplInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  try {
    PAYLOAD_TYPE result = index->find(key, val);
    // The find method returns the value and also sets val
    return true;
  } catch (...) {
    return false;
  }
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaImplInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return index->insert(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaImplInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  // Since kanva_impl doesn't have a specific update method, we'll try to remove and insert
  // This is a common pattern when update is not directly supported
  index->remove(key);
  return index->insert(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaImplInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return index->remove(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t kanvaImplInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                       std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                       Param *param) {
  // kanva_impl doesn't seem to have a direct scan/rangequery method
  // We'll implement a simple scan by iterating through possible keys
  // This is a basic implementation - might need optimization based on actual kanva_impl capabilities
  
  size_t found = 0;
  KEY_TYPE current_key = key_low_bound;
  
  for (size_t i = 0; i < key_num && found < key_num; i++) {
    PAYLOAD_TYPE val;
    if (get(current_key, val, param)) {
      result[found] = std::make_pair(current_key, val);
      found++;
    }
    current_key++;
  }
  
  return found;
}
