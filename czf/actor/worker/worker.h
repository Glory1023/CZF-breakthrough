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
using RNG_t = czf::actor::RNG_t;  ///< the type for the random number generator
using Seed_t = czf::actor::Seed_t;          ///< the type of random seed
using Clock_t = std::chrono::steady_clock;  ///< the type for clock

class WorkerManager {
 public:
  static WorkerOption worker_option;
  static GameInfo game_info;

 public:
  /** run cpu & gpu workers */
  void run(size_t, size_t, size_t, size_t);
  /** terminate all workers */
  void terminate();
  /** enqueue a protobuf job to the job queue */
  void enqueue_job(py::object, std::vector<float>, const std::vector<int32_t>&,
                   const TreeOption&);
  /** dequeue a result from the job queue */
  py::tuple wait_dequeue_result();
  /** load model from file */
  void load_model(const std::string&);

 private:
  void worker_cpu(Seed_t, Seed_t);
  void worker_gpu(bool);

 private:
  std::atomic_bool running_{false};
  ModelManager model_manager;
  std::vector<std::thread> cpu_threads_, gpu_threads_, gpu_root_threads_;
  JobQueue cpu_queue_, gpu_queue_, gpu_root_queue_, result_queue_;
};

}  // namespace czf::actor::worker
