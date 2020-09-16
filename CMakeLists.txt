cmake_minimum_required(VERSION 3.14)
project(czf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Option
option(BUILD_TESTING "build tests" ON)
option(ENABLE_CLANG_TIDY "enable clang-tidy" ON)
option(ENABLE_CLANG_FORMAT "enable clang-format" ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(BUILD_TESTING OFF)
  set(ENABLE_CLANG_TIDY OFF)
  set(ENABLE_CLANG_FORMAT OFF)
endif()

message(STATUS "BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "ENABLE_CLANG_TIDY: ${ENABLE_CLANG_TIDY}")
message(STATUS "ENABLE_CLANG_FORMAT: ${ENABLE_CLANG_FORMAT}")

# clang-tidy
if(ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy" "-config=")
endif()

# Test
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(third_party/catch2)
  add_subdirectory(tests)
endif()

# Project
if(NOT COMMAND pybind11_add_module)
  add_subdirectory(third_party/pybind11)
endif()
include_directories(third_party/czf_env)
add_subdirectory(czf)

# clang-format
if(ENABLE_CLANG_FORMAT)
  include(ClangFormat)
  file(GLOB_RECURSE CZF_SRC "czf/*.cc" "czf/*.h")
  file(GLOB_RECURSE TESTS_SRC "tests/*.cc" "tests/*.h")
  clangformat_setup(${CZF_SRC} ${TESTS_SRC})
endif()
