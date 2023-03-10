cmake_minimum_required(VERSION 3.14)
project(czf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Option
option(BUILD_TESTING "build tests" ON)
option(BUILD_BENCHMARK "build benchmark" OFF)
option(ENABLE_CLANG_TIDY "enable clang-tidy" ON)
option(ENABLE_CLANG_FORMAT "enable clang-format" ON)
set(GCC_WARNING_FLAGS
    "-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Wunused")
set(GCC_WARNING_FLAGS
    "${GCC_WARNING_FLAGS} -Wsign-conversion -Wdouble-promotion")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(BUILD_TESTING OFF)
  set(ENABLE_CLANG_TIDY OFF)
  set(ENABLE_CLANG_FORMAT OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")
else()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_WARNING_FLAGS}")
  set(ENABLE_CLANG_TIDY OFF)
  set(ENABLE_CLANG_FORMAT OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

message(STATUS "BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "BUILD_BENCHMARK: ${BUILD_BENCHMARK}")
message(STATUS "ENABLE_CLANG_TIDY: ${ENABLE_CLANG_TIDY}")
message(STATUS "ENABLE_CLANG_FORMAT: ${ENABLE_CLANG_FORMAT}")

# clang-tidy
if(ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy" "-config=" "-extra-arg"
                           "-fsized-deallocation")
endif()

# Test
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(third_party/catch2)
  add_subdirectory(tests)
endif()

# Project
add_subdirectory(third_party/pybind11)
include_directories(SYSTEM third_party/concurrentqueue)
include_directories(SYSTEM third_party/pcg-cpp/include)
add_subdirectory(czf)
add_subdirectory(docs)

# clang-format
if(ENABLE_CLANG_FORMAT)
  include(ClangFormat)
  file(GLOB_RECURSE CZF_SRC "czf/*.cc" "czf/*.h")
  file(GLOB_RECURSE TESTS_SRC "tests/*.cc" "tests/*.h")
  clangformat_setup(${CZF_SRC} ${TESTS_SRC})
endif()
