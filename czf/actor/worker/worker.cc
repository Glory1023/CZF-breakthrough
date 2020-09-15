#include "worker.hpp"

namespace czf::workers {

void WorkerManager::register_worker(std::shared_ptr<Worker> worker) {
  workers_.push_back(std::move(worker));
}

void WorkerManager::run() {
  for (auto &worker : workers_) {
    threads_.emplace_back([this, worker] { worker->run(); });
  }
}
void WorkerManager::terminate() {
  for (auto &worker : workers_) {
    worker->terminate();
  }
  for (auto &thread : threads_) {
    thread.join();
  }
  workers_.clear();
  threads_.clear();
}

} // namespace czf::workers