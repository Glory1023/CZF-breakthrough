#pragma once
#include <blockingconcurrentqueue.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "mcts/mcts.h"
#include "utils/config.h"

namespace czf::actor::worker {

namespace py = ::pybind11;
namespace mcts = ::czf::actor::mcts;
using JobQueue = moodycamel::BlockingConcurrentQueue<mcts::Tree>;
using ResultQueue = moodycamel::BlockingConcurrentQueue<mcts::MctsInfo>;

class Worker {
 public:
  virtual ~Worker(){};
  virtual void run() = 0;
  virtual void terminate() { running_ = false; };

 protected:
  std::atomic_bool running_{true};
};

class WorkerManager {
 public:
  WorkerManager() = default;
  ~WorkerManager() { terminate(); }
  void register_worker(std::shared_ptr<Worker>);
  void enqueue_job(py::object, py::buffer, py::buffer, py::buffer);
  py::tuple wait_dequeue_result();
  void run();
  void terminate();

 private:
  std::vector<std::shared_ptr<Worker>> workers_;
  std::vector<std::thread> threads_;
  JobQueue cpu_queue_, gpu_queue_;
  ResultQueue result_queue_;
};

class WorkerCPU final : public Worker {
 public:
  void run() override;
};

class WorkerGPU final : public Worker {
 public:
  void run() override;
};

}  // namespace czf::actor::worker
