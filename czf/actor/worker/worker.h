#pragma once
#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "utils/config.h"
#include "worker/job.h"
#include "worker/model.h"

namespace czf::actor::worker {

namespace py = ::pybind11;
using RNG_t = czf::actor::RNG_t;   ///< the type for the random number generator
using Seed_t = RNG_t::state_type;  ///< the type of random seed

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
  void worker_cpu(Seed_t, Seed_t);
  void worker_gpu(Seed_t, Seed_t, bool);

 private:
  std::atomic_bool running_{false};
  ModelManager model_manager;
  std::vector<std::thread> cpu_threads_, gpu_threads_, gpu_root_threads_;
  JobQueue cpu_queue_, gpu_queue_, gpu_root_queue_, result_queue_;
};

}  // namespace czf::actor::worker
