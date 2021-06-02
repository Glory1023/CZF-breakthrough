include_directories(.)
file(GLOB MUZERO_WORKERS_SRC "worker/*.cc")
add_library(czf.muzero_workers OBJECT ${MUZERO_WORKERS_SRC})
add_library(czf.muzero_mcts OBJECT mcts/mcts.cc)

# pybind11 module
pybind11_add_module(muzero_worker
  module.cc
  $<TARGET_OBJECTS:czf.muzero_workers>
  $<TARGET_OBJECTS:czf.muzero_mcts>
  $<TARGET_OBJECTS:czf.proto_cpp>
)
target_link_libraries(muzero_worker PRIVATE ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES})
install(TARGETS muzero_worker LIBRARY DESTINATION czf/actor)

# benchmark
if(BUILD_BENCHMARK)
  add_executable(
    benchmark benchmark.cc $<TARGET_OBJECTS:czf.muzero_workers>
              $<TARGET_OBJECTS:czf.muzero_mcts> $<TARGET_OBJECTS:czf.proto_cpp>)
  target_link_libraries(benchmark PRIVATE pybind11::embed ${Protobuf_LIBRARIES}
                                          ${TORCH_LIBRARIES})
  install(TARGETS benchmark LIBRARY DESTINATION czf/actor)
endif()
