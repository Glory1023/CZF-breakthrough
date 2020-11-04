#pragma once
#include <blockingconcurrentqueue.h>

#include "mcts/mcts.h"

namespace czf::actor::worker {

namespace py = ::pybind11;

struct Job {
  enum class Step {
    FORWARD_ROOT,               ///< forward the representation (h => f)
    SELECT,                     ///< Mcts selection & expansion
    FORWARD,                    ///< forward the dynamics model (g => f)
    UPDATE,                     ///< Mcts update
    DONE,                       ///< the job is finished
  } step = Step::FORWARD_ROOT;  ///< current step of the job
  czf::actor::mcts::Tree tree;  ///< Mcts tree
  py::object job;               ///< protobuf packet buffer
};

/** the type of a job queue */
using JobQueue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<Job>>;

}  // namespace czf::actor::worker
