cmake_minimum_required(VERSION 3.12)
project(czf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(czf OBJECT czf.cc czf.h)
target_include_directories(czf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set(CZF_OBJECTS $<TARGET_OBJECTS:czf> $<TARGET_OBJECTS:games>)

include_directories(.)
add_subdirectory(third_party/pybind11)
add_subdirectory(games)
add_subdirectory(python)

add_executable(dummy_agent dummy_agent.cc ${CZF_OBJECTS})
set_target_properties(dummy_agent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)