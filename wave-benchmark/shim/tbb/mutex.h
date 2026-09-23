#pragma once
#include <tbb/spin_mutex.h>
namespace tbb { using mutex = spin_mutex; }
