cmake_minimum_required (VERSION 3.14)
project (GRE)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(FetchContent)
include(ExternalProject)

find_package(OpenMP)
find_package(MKL REQUIRED)
find_package(JeMalloc REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${MKL_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${JEMALLOC_INCLUDE_DIR})
include_directories(SYSTEM src/competitor/hot/src/libs/hot/commons/include)
include_directories(SYSTEM src/competitor/hot/src/libs/hot/rowex/include)
include_directories(SYSTEM src/competitor/hot/src/libs/hot/single-threaded/include)
include_directories(SYSTEM src/competitor/hot/src/libs/idx/content-helpers/include)
include_directories(SYSTEM src/competitor/dili/src/src)
include_directories(SYSTEM src/competitor/dili/src/src/global)
include_directories(SYSTEM src/competitor/dili/src/src/utils)
include_directories(SYSTEM src/competitor/dili/src/src/butree)
include_directories(SYSTEM src/competitor/dili/src/src/dili)
include_directories(SYSTEM src/competitor/hyper/src/include)

add_compile_options(-faligned-new -march=native -g -O3)

# add_subdirectory(src/competitor/wormhole)
add_subdirectory(src/competitor/wormhole_u64)
add_subdirectory(src/competitor/masstree)
add_subdirectory(src/competitor/dili/src)

# microbench
add_executable(microbench
        ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/dili/dili_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/dili/dili_impl.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/dilol/dilol_wrapper.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/dilol/dilol_impl.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/dilol/src/concurrent_global_types.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/hyper/src/src/hyper_index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/hyper/src/src/leaf_node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/hyper/src/src/model_inner_node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/hyper/src/src/search_inner_node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/hyper/src/src/overflow_buffer.cpp
    )

target_link_libraries(microbench
        PRIVATE
        # WORMHOLE
        WORMHOLEU64
        MASSTREE
        dili_func
        butree_func
        global_func
        utils_func
)

target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${MKL_LIBRARIES} ${TBB_LIBRARIES})
