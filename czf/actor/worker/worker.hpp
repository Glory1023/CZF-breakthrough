#pragma once
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace czf::workers {

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
  void register_worker(std::shared_ptr<Worker> worker);
  void run();
  void terminate();

private:
  std::vector<std::shared_ptr<Worker>> workers_;
  std::vector<std::thread> threads_;
};

class WorkerCPU final : public Worker {
public:
  void run() override;
};

class WorkerGPU final : public Worker {
public:
  void run() override;
};

} // namespace czf::workers
