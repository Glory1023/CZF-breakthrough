#pragma once
#include <blockingconcurrentqueue.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <thread>
#include <vector>

#include "mcts/mcts.h"
#include "utils/config.h"
#include "utils/model.h"
#include "utils/random.h"

namespace czf::actor::worker {

namespace py = ::pybind11;
namespace mcts = ::czf::actor::mcts;

struct Job {
  enum class Step {
    TERMINATE,
    SELECT,
    EVALUATE_ROOT,
    EVALUATE,
    UPDATE,
    DONE
  } next_step;
  mcts::Tree tree;
  py::object job;
};

using JobQueue = moodycamel::BlockingConcurrentQueue<Job>;
using PRNG = ::czf::actor::utils::random::Xorshift;
using SeedPRNG = ::czf::actor::utils::random::Splitmix;
using Seed_t = PRNG::seed_type;

class WorkerManager {
 public:
  void run(size_t, size_t);
  void terminate();
  void enqueue_job(py::object, py::buffer, py::buffer, py::buffer);
  py::tuple wait_dequeue_result();
  void load_model(const std::string&);

 private:
  void worker_cpu(size_t, Seed_t);
  void worker_gpu(size_t, Seed_t);

 private:
  std::atomic_bool running_{false};
  std::vector<std::thread> cpu_threads_, gpu_threads_;
  std::vector<mcts::Model> model_;
  JobQueue cpu_queue_, gpu_queue_, result_queue_;
};

}  // namespace czf::actor::worker
