#pragma once
#include <blockingconcurrentqueue.h>

#include "mcts/mcts.h"

namespace czf::actor::worker {

namespace py = ::pybind11;

// TODO
struct GameInfo {
  std::vector<int64_t> observation_shape = {3, 3, 3};
  std::vector<int64_t> state_shape = {3, 3, 3};
  std::vector<::czf::actor::mcts::Action_t> all_actions = {0, 1, 2, 3, 4,
                                                           5, 6, 7, 8};
};

struct JobOption {
  size_t batch_size = 200u,    // GPU batch size
      simulation_count = 800u  // Mcts simulation counts
      ;
};

struct Job {
  enum class Step {
    FORWARD_ROOT,
    SELECT,
    FORWARD,
    UPDATE,
    DONE,
  } step = Step::FORWARD_ROOT;
  ::czf::actor::mcts::Tree tree;
  ::czf::actor::mcts::ForwardResult result;
  py::object job;
};

using JobQueue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<Job>>;

}  // namespace czf::actor::worker
