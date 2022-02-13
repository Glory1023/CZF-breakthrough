include_directories(.)
file(GLOB MUZERO_WORKERS_SRC_GUMBEL "worker/*.cc")
add_library(czf.muzero_workers_gumbel OBJECT ${MUZERO_WORKERS_SRC_GUMBEL})
add_library(czf.muzero_mcts_gumbel OBJECT mcts/mcts.cc)

# pybind11 module
pybind11_add_module(muzero_worker_gumbel
  module.cc
  $<TARGET_OBJECTS:czf.muzero_workers_gumbel>
  $<TARGET_OBJECTS:czf.muzero_mcts_gumbel>
  $<TARGET_OBJECTS:czf.proto_cpp>
)
target_link_libraries(muzero_worker_gumbel PRIVATE ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES})
install(TARGETS muzero_worker_gumbel LIBRARY DESTINATION czf/actor)

# benchmark
if(BUILD_BENCHMARK)
  add_executable(
    benchmark benchmark.cc $<TARGET_OBJECTS:czf.muzero_workers_gumbel>
              $<TARGET_OBJECTS:czf.muzero_mcts_gumbel> $<TARGET_OBJECTS:czf.proto_cpp>)
  target_link_libraries(benchmark PRIVATE pybind11::embed ${Protobuf_LIBRARIES}
                                          ${TORCH_LIBRARIES})
  install(TARGETS benchmark LIBRARY DESTINATION czf/actor)
endif()
