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
  /** load model from bytes */
  void load_from_bytes(const std::string&);
  /** load model from file */
  void load_from_file(const std::string&);

 private:
  /** cpu worker */
  void worker_cpu(Seed_t, Seed_t);
  /** gpu worker */
  void worker_gpu(bool);

 private:
  /** can worker continue to run */
  std::atomic_bool running_{false};
  /** manage models */
  ModelManager model_manager;
  /** threads on cpu and gpu */
  std::vector<std::thread> cpu_threads_, gpu_threads_;
  /** job queues for cpu, gpu, root gpu, and result */
  JobQueue cpu_queue_, gpu_queue_, gpu_root_queue_, result_queue_;
};

}  // namespace czf::actor::worker
