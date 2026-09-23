#pragma once
#include <tbb/spin_rw_mutex.h>
namespace tbb { using reader_writer_lock = spin_rw_mutex; }
