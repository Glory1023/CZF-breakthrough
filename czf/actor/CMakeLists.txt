find_package(Torch REQUIRED)
find_package(
  Python3
  COMPONENTS Development
  REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(.)
file(GLOB WORKERS_SRC "worker/*.cc")
add_library(czf.workers OBJECT ${WORKERS_SRC})
add_library(czf.mcts OBJECT mcts/mcts.cc utils/model.cc)

# pybind11 module
pybind11_add_module(worker module.cc $<TARGET_OBJECTS:czf.workers>
                    $<TARGET_OBJECTS:czf.mcts>)
target_link_libraries(worker PRIVATE ${TORCH_LIBRARIES})
install(TARGETS worker LIBRARY DESTINATION czf/actor)

# test module
add_test(NAME pybind-tests COMMAND ${Python3_EXECUTABLE} -m pytest
                                   ${PROJECT_SOURCE_DIR}/tests)
