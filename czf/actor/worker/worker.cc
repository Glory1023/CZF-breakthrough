#include "worker.h"

#include <iostream>

namespace czf::actor::worker {

void WorkerManager::run(size_t num_cpu_worker, size_t num_gpu_worker) {
  if (!running_) {
    running_ = true;
    SeedPRNG rng{1};
    for (size_t i = 0; i < num_cpu_worker; ++i) {
      Seed_t seed = rng();
      cpu_threads_.emplace_back([this, i, seed] { worker_cpu(i, seed); });
    }
    mcts::Model::prepare_nvrtc();
    for (size_t i = 0; i < num_gpu_worker; ++i) {
      Seed_t seed = rng();
      gpu_threads_.emplace_back([this, i, seed] { worker_gpu(i, seed); });
    }
  }
}

void WorkerManager::terminate() {
  if (running_) {
    running_ = false;
    // enqueue terminate job
    for (const auto &thread : cpu_threads_) {
      cpu_queue_.enqueue(Job{});
    }
    for (const auto &thread : gpu_threads_) {
      gpu_queue_.enqueue(Job{});
    }
    result_queue_.enqueue(Job{});
    // join threads
    for (auto &thread : cpu_threads_) {
      thread.join();
    }
    for (auto &thread : gpu_threads_) {
      thread.join();
    }
  }
}

void WorkerManager::enqueue_job(py::object pyjob, py::buffer obs_buffer,
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

  Job job{Job::Step::SELECT, {}, std::move(pyjob)};
  job.tree.construct_root(input, legal_actions);
  cpu_queue_.enqueue(std::move(job));
}

py::tuple WorkerManager::wait_dequeue_result() {
  Job job;
  result_queue_.wait_dequeue(job);
  // const mcts::MctsInfo & info = job.tree.
  // return py::make_tuple(info.policy, info.visits);
  return py::make_tuple(0, 1);
}

void WorkerManager::load_model(const std::string &path) {
  std::cerr << "Load model from " << path << std::endl;
}

void WorkerManager::worker_cpu(size_t, Seed_t seed) {
  PRNG rng{seed};
  while (running_) {
    Job job;
    cpu_queue_.wait_dequeue(job);
    if (job.next_step == Job::Step::TERMINATE) {
      return;
    }
    if (job.next_step == Job::Step::SELECT) {
      job.tree.before_forward(rng);
    } else if (job.next_step == Job::Step::UPDATE) {
      job.tree.after_forward(rng);
    }
    if (job.next_step == Job::Step::DONE) {
      result_queue_.enqueue(std::move(job));
    } else if (job.next_step == Job::Step::EVALUATE) {
      gpu_queue_.enqueue(std::move(job));
    }
  }
}

void WorkerManager::worker_gpu(size_t index, Seed_t seed) {
  ;
  ;
}

}  // namespace czf::actor::worker
