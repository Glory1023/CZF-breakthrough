include_directories(.)
file(GLOB GUMBEL_MUZERO_WORKERS_SRC "worker/*.cc")
add_library(czf.gumbel_muzero_workers OBJECT ${GUMBEL_MUZERO_WORKERS_SRC})
add_library(czf.gumbel_muzero_mcts OBJECT mcts/mcts.cc)

# pybind11 module
pybind11_add_module(gumbel_muzero_worker
  module.cc
  $<TARGET_OBJECTS:czf.gumbel_muzero_workers>
  $<TARGET_OBJECTS:czf.gumbel_muzero_mcts>
  $<TARGET_OBJECTS:czf.proto_cpp>
)
target_link_libraries(gumbel_muzero_worker PRIVATE ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES})
install(TARGETS gumbel_muzero_worker LIBRARY DESTINATION czf/actor)

# benchmark
if(BUILD_BENCHMARK)
  add_executable(
    benchmark benchmark.cc $<TARGET_OBJECTS:czf.gumbel_muzero_workers>
              $<TARGET_OBJECTS:czf.gumbel_muzero_mcts> $<TARGET_OBJECTS:czf.proto_cpp>)
  target_link_libraries(benchmark PRIVATE pybind11::embed ${Protobuf_LIBRARIES}
                                          ${TORCH_LIBRARIES})
  install(TARGETS benchmark LIBRARY DESTINATION czf/actor)
endif()
