#pragma once
#include <blockingconcurrentqueue.h>

#include "mcts/mcts.h"

namespace czf::actor::worker {

namespace py = ::pybind11;

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
