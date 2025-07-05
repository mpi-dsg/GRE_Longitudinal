// #include "./src/osm/include/function.h"
#include "./src/Kanva_RS/kanva_RS.h"
#include "../indexInterface.h"
#include "./src/common/util.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class kanvaInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  kanvaInterface() : index(1) {} // Use the single parameter constructor with 1 thread
  
  void init(Param *param = nullptr) {}

  void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr);

  long long memory_consumption() { return 0; }

private:
  kanva_RS::Kanva_RS<KEY_TYPE, PAYLOAD_TYPE> index;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
  std::vector<KEY_TYPE> key_temp;
  std::vector<PAYLOAD_TYPE> val_temp;
  key_temp.reserve(num);
  val_temp.reserve(num);
  for (size_t i = 0; i < num; i++) {
      key_temp.push_back(key_value[i].first);
      val_temp.push_back(key_value[i].second);
  }
  index.train(key_temp, val_temp, 32, 0); // Add thread_id_t parameter (0)
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  kanva_RS::result_t res = index.find(key, val, 0);
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  kanva_RS::result_t res = index.insert(key, value, 0);
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  kanva_RS::result_t res = index.update(key, value, 0);
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
 kanva_RS::result_t res = index.remove(key, 0);
  if(res == kanva_RS::result_t::ok) {
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t kanvaInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> res;
  res.reserve(key_num);
  size_t scan_size = index.rangequery(key_low_bound, key_num, res, 0);
  return scan_size;
}