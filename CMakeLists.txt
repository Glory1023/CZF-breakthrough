cmake_minimum_required(VERSION 3.14)
project(czf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(.)

add_subdirectory(czf_env)
add_subdirectory(czf)
