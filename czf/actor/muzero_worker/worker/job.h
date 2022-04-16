#pragma once
#include <blockingconcurrentqueue.h>

#include "mcts/mcts.h"

namespace czf::actor::muzero_worker::worker {

struct Job {
  enum class Step {
    kEnqueue,                                  ///< enqueue a batch of jobs
    kForwardRoot,                              ///< forward the representation (h => f)
    kSelect,                                   ///< Mcts selection & expansion
    kForwardState,                             ///< forward the dynamics model (g => f)
    kForwardAfterstate,                        ///< forward the afterstate_dynamics model (g => f)
    kUpdate,                                   ///< Mcts update
    kDone,                                     ///< the job is finished
    kDequeue,                                  ///< dequeue a batch of jobs
  } step = Step::kEnqueue;                     ///< current step of the job
  czf::actor::muzero_worker::mcts::Tree tree;  ///< Mcts tree
  std::string job_str;                         ///< protobuf packet buffer
};

/** the type of a job queue */
using JobQueue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<Job>>;

}  // namespace czf::actor::muzero_worker::worker
