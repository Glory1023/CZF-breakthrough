include_directories(.)
add_library(czf.alphazero_mcts OBJECT
  node.cc
  tree.cc
  job.cc
  # job_expand.cc
  model_manager.cc
  worker_manager.cc
)

# pybind11 module
pybind11_add_module(alphazero_worker
  module.cc
  $<TARGET_OBJECTS:czf.alphazero_mcts>
  $<TARGET_OBJECTS:czf.czf_env>
  $<TARGET_OBJECTS:czf.proto_cpp>
)
target_link_libraries(alphazero_worker PRIVATE ${Protobuf_LIBRARIES} ${TORCH_LIBRARIES})
install(TARGETS alphazero_worker LIBRARY DESTINATION czf/actor)