#include "worker.h"

namespace czf::actor::worker {

void WorkerManager::register_worker(std::shared_ptr<Worker> worker) {
  workers_.push_back(std::move(worker));
}

void WorkerManager::enqueue_job(py::object job, py::buffer obs_buffer,
                                py::buffer obs_shape_buffer,
                                py::buffer legal_actions_buffer) {
  // observation
  py::buffer_info obs_info = obs_buffer.request();
  auto *observation = static_cast<float *>(obs_info.ptr);
  // observation shape
  py::buffer_info obs_shape_info = obs_shape_buffer.request();
  const int64_t *obs_shape_data = static_cast<int64_t *>(obs_shape_info.ptr);
  auto obs_shape =
      at::IntArrayRef{obs_shape_data, static_cast<size_t>(obs_shape_info.size)};
  // observation tensor
  auto input = torch::from_blob(observation, obs_shape);  // NOLINT
  // legal actions
  py::buffer_info legal_actions_info = legal_actions_buffer.request();
  const int32_t *legal_actions_data =
      static_cast<int32_t *>(legal_actions_info.ptr);
  std::vector<int32_t> legal_actions{
      legal_actions_data, legal_actions_data + legal_actions_info.size};

  mcts::Tree tree;
  tree.construct_root(std::move(job), input, legal_actions);
  cpu_queue_.enqueue(std::move(tree));
}

py::tuple WorkerManager::wait_dequeue_result() {
  mcts::MctsInfo info;
  result_queue_.wait_dequeue(info);
  return py::make_tuple(info.policy, info.visits);
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

}  // namespace czf::actor::worker
