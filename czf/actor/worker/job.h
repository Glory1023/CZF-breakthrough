#pragma once
#include <blockingconcurrentqueue.h>
#include <pybind11/pybind11.h>

#include "mcts/mcts.h"

namespace czf::actor::worker {

namespace py = ::pybind11;

struct Job {
  enum class Step {
    kForwardRoot,               ///< forward the representation (h => f)
    kSelect,                    ///< Mcts selection & expansion
    kForward,                   ///< forward the dynamics model (g => f)
    kUpdate,                    ///< Mcts update
    kDone,                      ///< the job is finished
  } step = Step::kForwardRoot;  ///< current step of the job
  czf::actor::mcts::Tree tree;  ///< Mcts tree
  py::object job;               ///< protobuf packet buffer
};

/** the type of a job queue */
using JobQueue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<Job>>;

}  // namespace czf::actor::worker
