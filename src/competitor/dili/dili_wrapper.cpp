#include "dili_wrapper.h"
#include "./src/src/dili/DILI.h"

DiliWrapper::DiliWrapper() {
    impl = new DILI();
    impl->set_mirror_dir("/tmp/dili_mirror");
}

DiliWrapper::~DiliWrapper() {
    delete impl;
}

void DiliWrapper::bulk_load(const std::vector<std::pair<long, long>>& data) {
    impl->bulk_load(data);
}

long DiliWrapper::search(long key) const {
    return impl->search(key);
}

bool DiliWrapper::insert(long key, long value) {
    return impl->insert(key, value);
}

bool DiliWrapper::erase(long key) {
    return impl->erase(key);
}

int DiliWrapper::range_query(long k1, long k2, long* results) {
    return impl->range_query(k1, k2, results);
}

size_t DiliWrapper::total_size() const {
    return impl->total_size();
}
