#pragma once

#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "config.h"
#include "czf/env/czf_env/game.h"
#include "job.h"
#include "model_manager.h"
#include "third_party/concurrentqueue/blockingconcurrentqueue.h"

namespace czf::actor::alphazero_worker {

namespace py = ::pybind11;
using Clock_t = std::chrono::steady_clock;
using JobQueue = moodycamel::BlockingConcurrentQueue<std::unique_ptr<Job>>;

class WorkerManager {
 public:
  static WorkerOption worker_option;

 public:
  ~WorkerManager() = default;

 public:
  void run(size_t, size_t, size_t);
  void terminate();
  std::tuple<py::bytes, std::string, int> enqueue_job_batch(const std::string&);
  py::bytes wait_dequeue_result(size_t);
  void load_from_bytes(const std::string&);
  void load_from_file(const std::string&);
  void load_game(const std::string& name);

 private:
  void cpu_worker(uint32_t seed);
  void gpu_worker(uint32_t seed);

 private:
  std::atomic_bool running_{false};
  ModelManager model_manager;
  std::vector<std::thread> cpu_threads_, gpu_threads_;
  JobQueue cpu_queue_, gpu_queue_, result_queue_;
  czf::env::czf_env::GamePtr game;
};

}  // namespace czf::actor::alphazero_worker
