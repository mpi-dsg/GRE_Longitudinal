#pragma once

#include <vector>
#include <utility>
#include <cstddef>

// Forward declare the DILI class to avoid including the headers
class DILI;

// Wrapper class to isolate DILI implementation
class DiliWrapper {
public:
    DiliWrapper();
    ~DiliWrapper();
    
    // Forwarded methods
    void bulk_load(const std::vector<std::pair<long, long>>& data);
    long search(long key) const;
    bool insert(long key, long value);
    bool erase(long key);
    int range_query(long k1, long k2, long* results);
    size_t total_size() const;
    
private:
    DILI* impl;
};
