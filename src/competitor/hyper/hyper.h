#pragma once

#include "../indexInterface.h"
#include "./src/include/hyper_index.h"
#include <vector>
#include <optional>

template<class KEY_TYPE, class PAYLOAD_TYPE>
class HyperInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    HyperInterface() : hyper_index_(128.0, 0.1, 16*1024*1024) {}
    
    void init(Param *param = nullptr) {}

    void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

    bool remove(KEY_TYPE key, Param *param = nullptr);

    size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr);

    long long memory_consumption();

private:
    Hyper hyper_index_;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
    // Convert the input data to the format expected by Hyper
    std::vector<std::pair<KeyType, ValueType>> data;
    data.reserve(num);
    
    for (size_t i = 0; i < num; ++i) {
        data.emplace_back(static_cast<KeyType>(key_value[i].first), 
                         static_cast<ValueType>(key_value[i].second));
    }
    
    // Bulk load into the Hyper index
    hyper_index_.bulkLoad(data);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
    KeyType hyper_key = static_cast<KeyType>(key);
    std::optional<ValueType> result = hyper_index_.find(hyper_key);
    
    if (result.has_value()) {
        val = static_cast<PAYLOAD_TYPE>(result.value());
        return true;
    }
    return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    try {
        KeyType hyper_key = static_cast<KeyType>(key);
        ValueType hyper_value = static_cast<ValueType>(value);
        hyper_index_.insert(hyper_key, hyper_value);
        return true;
    } catch (const std::exception& e) {
        // Hyper may throw exceptions on insert failure
        return false;
    }
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
    // Hyper doesn't have a dedicated update method, so we treat it as insert
    // This follows the pattern of other learned indexes in the framework
    return put(key, value, param);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
    // Hyper doesn't support deletion in the current implementation
    // This is common for learned indexes optimized for read-heavy workloads
    return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num, 
                                                    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
    try {
        KeyType start_key = static_cast<KeyType>(key_low_bound);
        // For range queries, we need to estimate the upper bound
        // Since we don't know the exact upper key, we'll use a large range
        KeyType end_key = start_key + static_cast<KeyType>(key_num * 1000); // Heuristic upper bound
        
        std::vector<std::pair<KeyType, ValueType>> range_results = 
            hyper_index_.rangeQuery(start_key, end_key);
        
        size_t actual_count = 0;
        for (const auto& pair : range_results) {
            if (actual_count >= key_num) break;
            
            result[actual_count] = std::make_pair(
                static_cast<KEY_TYPE>(pair.first), 
                static_cast<PAYLOAD_TYPE>(pair.second)
            );
            actual_count++;
        }
        
        return actual_count;
    } catch (const std::exception& e) {
        return 0;
    }
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
long long HyperInterface<KEY_TYPE, PAYLOAD_TYPE>::memory_consumption() {
    // Hyper doesn't provide a direct memory consumption method
    // Return 0 as placeholder (common pattern in this framework)
    return 0;
}
