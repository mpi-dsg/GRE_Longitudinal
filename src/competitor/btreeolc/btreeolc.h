#include"./src/BTreeOLC/BTreeOLC_child_layout.h"
#include"../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class BTreeOLCInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) {}

  void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr) {
    return idx.scan(key_low_bound, key_num, result);
  }

  long long memory_consumption() { return 0; }

  std::vector<KEY_TYPE> get_all_keys() {
    std::vector<KEY_TYPE> keys;
    size_t key_num = 200000000;
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[key_num];
    size_t num_keys = scan(std::numeric_limits<KEY_TYPE>::min(), key_num, result, nullptr);
    for (size_t i = 0; i < num_keys; ++i) {
        keys.push_back(result[i].first);
    }
    delete[] result;
    return keys;
  }

private:
  btreeolc::BTree<KEY_TYPE, PAYLOAD_TYPE> idx;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                          Param *param) {
  for (auto i = 0; i < num; i++) {
    idx.insert(key_value[i].first, key_value[i].second);
  }
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  bool res = idx.lookup(key, val);
  return res;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  idx.insert(key, value);
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return false;
}

