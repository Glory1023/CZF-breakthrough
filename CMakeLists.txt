cmake_minimum_required(VERSION 3.14)
project(czf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(.)

add_subdirectory(third_party)

set_target_properties(czf_env PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(czf)
