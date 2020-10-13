#include "worker.h"

#include <pybind11/stl.h>

#include <iostream>

namespace czf::actor::worker {

JobOption WorkerManager::job_option;
GameInfo WorkerManager::game_info;
MctsOption WorkerManager::mcts_option;

void WorkerManager::run(size_t num_cpu_worker, size_t num_gpu_worker,
                        size_t num_gpu_root_worker, size_t num_gpu) {
  if (!running_) {
    running_ = true;
    SeedPRNG rng{1};
    for (size_t i = 0; i < num_cpu_worker; ++i) {
      Seed_t seed = rng();
      cpu_threads_.emplace_back([this, seed] { worker_cpu(seed); });
    }
    ModelManager::prepare_nvrtc();
    for (size_t i = 0; i < num_gpu_worker; ++i) {
      Seed_t seed = rng();
      gpu_threads_.emplace_back([this, seed] { worker_gpu(seed, false); });
    }
    for (size_t i = 0; i < num_gpu_root_worker; ++i) {
      Seed_t seed = rng();
      gpu_root_threads_.emplace_back([this, seed] { worker_gpu(seed, true); });
    }
    model_manager.resize(num_gpu);
  }
}

void WorkerManager::terminate() {
  if (running_) {
    running_ = false;
    // enqueue terminate job
    for ([[maybe_unused]] const auto &thread : cpu_threads_) {
      cpu_queue_.enqueue(nullptr);
    }
    for ([[maybe_unused]] const auto &thread : gpu_threads_) {
      gpu_queue_.enqueue(nullptr);
    }
    for ([[maybe_unused]] const auto &thread : gpu_threads_) {
      gpu_root_queue_.enqueue(nullptr);
    }
    result_queue_.enqueue(nullptr);
    // join threads
    for (auto &thread : cpu_threads_) {
      thread.join();
    }
    for (auto &thread : gpu_threads_) {
      thread.join();
    }
    for (auto &thread : gpu_root_threads_) {
      thread.join();
    }
  }
}

void WorkerManager::enqueue_job(py::object pyjob, py::buffer obs_buffer,
                                py::buffer actions_buffer) {
  // observation
  py::buffer_info obs_info = obs_buffer.request();
  auto *obs_data = static_cast<float *>(obs_info.ptr);
  std::vector<float> observation{obs_data, obs_data + obs_info.size};  // NOLINT
  // legal actions
  py::buffer_info actions_info = actions_buffer.request();
  const int32_t *actions_data = static_cast<int32_t *>(actions_info.ptr);
  std::vector<int32_t> legal_actions{
      actions_data, actions_data + actions_info.size};  // NOLINT
  // enqueue job
  auto job = std::make_unique<Job>();
  job->job = std::move(pyjob);
  job->tree.set_forward_result({std::move(observation), {}, 0, 0});
  job->tree.expand_root(legal_actions);
  cpu_queue_.enqueue(std::move(job));
}

py::tuple WorkerManager::wait_dequeue_result() {
  std::unique_ptr<Job> job;
  result_queue_.wait_dequeue(job);
  if (job == nullptr) {
    return py::make_tuple(py::none{}, py::none{});
  }
  auto result = job->tree.get_tree_result();
  return py::make_tuple(std::move(job->job), result.value, result.total_visits,
                        std::move(result.visits));
}

void WorkerManager::load_model(const std::string &path) {
  model_manager.load(path);
}

void WorkerManager::worker_cpu(Seed_t seed) {
  PRNG rng{seed};
  while (running_) {
    // dequeue a job
    std::unique_ptr<Job> job;
    cpu_queue_.wait_dequeue(job);
    if (job == nullptr) {
      return;
    }
    // process a job
    bool next_job = false;
    while (!next_job) {
      switch (job->step) {  // NOLINT
        case Job::Step::SELECT:
          // std::cerr << "select" << std::endl;
          job->tree.before_forward(rng, WorkerManager::game_info.all_actions);
          job->step = Job::Step::FORWARD;
          break;
        case Job::Step::UPDATE:
          // std::cerr << "update" << std::endl;
          job->tree.after_forward();
          job->step = job->tree.get_root_visits() >=
                              WorkerManager::job_option.simulation_count
                          ? Job::Step::DONE
                          : Job::Step::SELECT;
          break;
        case Job::Step::FORWARD_ROOT:
          // std::cerr << "forward root" << std::endl;
          gpu_root_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::FORWARD:
          // std::cerr << "forward" << std::endl;
          gpu_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::DONE:
          // std::cerr << "done" << std::endl;
          result_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        default:
          break;
      }
    }
  }
}

void WorkerManager::worker_gpu(Seed_t seed, bool is_root) {
  auto &queue = is_root ? gpu_root_queue_ : gpu_queue_;
  PRNG rng{seed};
  constexpr auto zero = std::chrono::duration<double>::zero();
  while (running_) {
    std::vector<std::unique_ptr<Job>> jobs;
    // collect jobs
    auto timeout = std::chrono::duration<double>::max();
    std::chrono::steady_clock::time_point deadline;
    for (size_t i = 0;
         i < WorkerManager::job_option.batch_size && timeout > zero;
         ++i, timeout = deadline - std::chrono::steady_clock::now()) {
      std::unique_ptr<Job> job;
      if (queue.wait_dequeue_timed(job, timeout)) {
        if (job == nullptr) {
          return;
        }
        if (jobs.empty()) {
          deadline =
              std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
        }
        jobs.push_back(std::move(job));
      } else {
        break;
      }
    }
    // std::cerr << "gpu forward" << std::endl;
    // construct input tensor
    const size_t batch_size = jobs.size();
    auto state_shape = WorkerManager::game_info.state_shape;
    state_shape.insert(state_shape.begin(), static_cast<int64_t>(batch_size));
    std::vector<float> state_vector;
    std::vector<float> action_vector;
    action_vector.reserve(batch_size);
    for (auto &job : jobs) {
      const auto &info = job->tree.get_forward_input();
      state_vector.insert(state_vector.end(), info.state.begin(),
                          info.state.end());
      action_vector.push_back(info.action);
    }
    auto [device, model_ptr] = model_manager.get();
    auto state_tensor =
        torch::from_blob(state_vector.data(), state_shape).to(device);
    auto action_tensor = torch::from_blob(action_vector.data(),
                                          {static_cast<int64_t>(batch_size), 1})
                             .to(device);
    torch::Tensor next_state_tensor;
    if (is_root) {
      // representation function
      next_state_tensor =
          model_ptr->get_method("forward_representation")({state_tensor})
              .toTensor();
    } else {
      // dynamics function
      auto results =
          model_ptr
              ->get_method("forward_dynamics")({state_tensor, action_tensor})
              .toTuple()
              ->elements();
      next_state_tensor = results[0].toTensor();
      auto batch_reward = results[1].toTensor().cpu();
      for (size_t i = 0; i < batch_size; ++i) {
        const auto idx = static_cast<int64_t>(i);
        const auto &reward_ptr = batch_reward[idx].data_ptr<float>();
        jobs[i]->result.reward = reward_ptr[0];  // NOLINT
      }
    }
    auto batch_state = next_state_tensor.cpu();
    for (size_t i = 0; i < batch_size; ++i) {
      const auto idx = static_cast<int64_t>(i);
      const auto &state_ptr = batch_state[idx].data_ptr<float>();
      const auto &state_size = batch_state[idx].numel();
      jobs[i]->result.state = {state_ptr, state_ptr + state_size};  // NOLINT
    }
    // prediction function
    auto results =
        model_ptr->forward({next_state_tensor}).toTuple()->elements();
    auto batch_policy = results[0].toTensor().cpu();
    auto batch_value = results[1].toTensor().cpu();
    // copy results back to jobs
    for (size_t i = 0; i < batch_size; ++i) {
      const auto idx = static_cast<int64_t>(i);
      const auto &policy_ptr = batch_policy[idx].data_ptr<float>();
      const auto &policy_size = batch_policy[idx].numel();
      const auto &value_ptr = batch_value[idx].data_ptr<float>();
      jobs[i]->result.policy = {policy_ptr,
                                policy_ptr + policy_size};  // NOLINT
      jobs[i]->result.value = value_ptr[0];                 // NOLINT
      jobs[i]->tree.set_forward_result(std::move(jobs[i]->result));
      jobs[i]->tree.normalize_root_policy();
      jobs[i]->step = Job::Step::UPDATE;
      if (is_root) {
        jobs[i]->tree.add_dirichlet_noise_to_root(rng);
      }
    }
    cpu_queue_.enqueue_bulk(std::make_move_iterator(jobs.begin()), jobs.size());
    jobs.clear();
  }
}

}  // namespace czf::actor::worker
