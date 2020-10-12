#pragma once
#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "utils/config.h"
#include "utils/random.h"
#include "worker/job.h"
#include "worker/model.h"

namespace czf::actor::worker {

namespace py = ::pybind11;
using PRNG = ::czf::actor::utils::random::Xorshift;
using SeedPRNG = ::czf::actor::utils::random::Splitmix;
using Seed_t = PRNG::seed_type;

class WorkerManager {
 public:
  static JobOption job_option;
  static GameInfo game_info;
  static MctsOption mcts_option;

 public:
  /** run cpu & gpu workers */
  void run(size_t, size_t, size_t, size_t);
  /** terminate all workers */
  void terminate();
  /** enqueue a protobuf job to the job queue */
  void enqueue_job(py::object, py::buffer, py::buffer);
  /** dequeue a result from the job queue */
  py::tuple wait_dequeue_result();
  /** load model from file */
  void load_model(const std::string&);

 private:
  void worker_cpu(Seed_t);
  void worker_gpu(Seed_t, bool);

 private:
  std::atomic_bool running_{false};
  ModelManager model_manager;
  std::vector<std::thread> cpu_threads_, gpu_threads_, gpu_root_threads_;
  JobQueue cpu_queue_, gpu_queue_, gpu_root_queue_, result_queue_;
};

}  // namespace czf::actor::worker
