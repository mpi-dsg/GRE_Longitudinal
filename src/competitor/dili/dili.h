#pragma once

#include "../indexInterface.h"
#include "dili_wrapper.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class diliInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) {}

  void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr);

  long long memory_consumption() { return static_cast<long long>(index.total_size()); }

private:
  DiliWrapper index;
};

// Template implementation - moved to separate compilation unit
extern template class diliInterface<uint64_t, uint64_t>;
extern template class diliInterface<int64_t, int64_t>;
