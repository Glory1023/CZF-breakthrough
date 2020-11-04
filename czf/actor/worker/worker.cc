#include "worker.h"

#include <pybind11/stl.h>

#include <numeric>

namespace czf::actor::worker {

WorkerOption WorkerManager::worker_option;
GameInfo WorkerManager::game_info;
MctsOption WorkerManager::mcts_option;

void WorkerManager::run(size_t num_cpu_worker, size_t num_gpu_worker,
                        size_t num_gpu_root_worker, size_t num_gpu) {
  if (!running_) {
    running_ = true;
    const auto seed = WorkerManager::worker_option.seed;
    // TODO(chengscott): print config
    for (size_t i = 0; i < num_cpu_worker; ++i) {
      const Seed_t stream = 100U + i;
      cpu_threads_.emplace_back(&WorkerManager::worker_cpu, this, seed, stream);
    }
    ModelManager::prepare_nvrtc();
    for (size_t i = 0; i < num_gpu_worker; ++i) {
      gpu_threads_.emplace_back(&WorkerManager::worker_gpu, this, false);
    }
    for (size_t i = 0; i < num_gpu_root_worker; ++i) {
      gpu_root_threads_.emplace_back(&WorkerManager::worker_gpu, this, true);
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

void WorkerManager::enqueue_job(py::object pyjob,
                                std::vector<float> observation,
                                const std::vector<int32_t> &legal_actions) {
  auto job = std::make_unique<Job>();
  job->job = std::move(pyjob);
  job->tree.set_forward_result({std::move(observation), {}, 0, 0});
  job->tree.expand_root(legal_actions);
  cpu_queue_.enqueue(std::move(job));
}

py::tuple WorkerManager::wait_dequeue_result() {
  std::unique_ptr<Job> job;
  result_queue_.wait_dequeue(job);
  py::gil_scoped_acquire acquire;
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

void WorkerManager::worker_cpu(Seed_t seed, Seed_t stream) {
  RNG_t rng{seed, stream};
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
          job->tree.before_forward(WorkerManager::game_info.all_actions);
          job->step = Job::Step::FORWARD;
          break;
        case Job::Step::UPDATE:
          job->tree.after_forward(rng);
          job->step = job->tree.get_root_visits() >=
                              WorkerManager::mcts_option.simulation_count
                          ? Job::Step::DONE
                          : Job::Step::SELECT;
          break;
        case Job::Step::FORWARD_ROOT:
          gpu_root_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::FORWARD:
          gpu_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::DONE:
          result_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        default:
          break;
      }
    }
  }
}

void WorkerManager::worker_gpu(bool is_root) {
  // jobs
  auto &queue = is_root ? gpu_root_queue_ : gpu_queue_;
  constexpr auto zero = std::chrono::duration<double>::zero();
  const auto max_timeout =
      std::chrono::microseconds(WorkerManager::worker_option.timeout_us);
  const auto max_batch_size = WorkerManager::worker_option.batch_size;
  std::vector<std::unique_ptr<Job>> jobs;
  jobs.reserve(max_batch_size);
  std::vector<czf::actor::mcts::ForwardResult> forward_results;
  forward_results.reserve(max_batch_size);
  // inputs (NCHW)
  std::vector<int64_t> state_shape =
      is_root ? WorkerManager::game_info.observation_shape
              : WorkerManager::game_info.state_shape;
  state_shape.insert(state_shape.begin(), static_cast<int64_t>(max_batch_size));
  std::vector<float> state_vector;
  state_vector.reserve(std::accumulate(state_shape.begin(), state_shape.end(),
                                       1U, std::multiplies<>()));
  std::vector<float> action_vector;
  action_vector.reserve(max_batch_size);
  while (running_) {
    // collect jobs
    Clock_t::time_point deadline;
    for (Clock_t::duration timeout = max_timeout;
         jobs.size() < max_batch_size && timeout > zero && running_;
         timeout = deadline - Clock_t::now()) {
      auto count = queue.wait_dequeue_bulk_timed(
          std::back_inserter(jobs), max_batch_size - jobs.size(), timeout);
      if (count == jobs.size()) {  // set the deadline in the first dequeue
        deadline = Clock_t::now() + max_timeout;
      }
    }
    if (jobs.empty()) {
      continue;
    }
    const size_t batch_size = jobs.size();
    forward_results.clear();
    forward_results.resize(batch_size);
    // construct input tensor
    state_shape[0] = static_cast<int64_t>(batch_size);
    state_vector.clear();
    action_vector.clear();
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
      const auto batch_reward = results[1].toTensor().cpu().contiguous();
      auto *const reward_ptr = batch_reward.data_ptr<float>();
      for (size_t i = 0; i < batch_size; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        forward_results[i].reward = reward_ptr[i];
      }
    }
    const auto batch_state = next_state_tensor.cpu().contiguous();
    const auto &state_size = batch_state[0].numel();
    auto *state_ptr = batch_state.data_ptr<float>();
    for (size_t i = 0; i < batch_size; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      auto *const state_ptr_end = state_ptr + state_size;
      forward_results[i].state.assign(state_ptr, state_ptr_end);
      state_ptr = state_ptr_end;
    }
    // prediction function
    const auto results =
        model_ptr->forward({next_state_tensor}).toTuple()->elements();
    const auto batch_policy = results[0].toTensor().cpu().contiguous();
    const auto batch_value = results[1].toTensor().cpu().contiguous();
    const auto policy_size = batch_policy[0].numel();
    auto *policy_ptr = batch_policy.data_ptr<float>();
    auto *const value_ptr = batch_value.data_ptr<float>();
    // copy results back to jobs
    for (size_t i = 0; i < batch_size; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      auto *const policy_ptr_end = policy_ptr + policy_size;
      forward_results[i].policy.assign(policy_ptr, policy_ptr_end);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      forward_results[i].value = value_ptr[i];
      jobs[i]->tree.set_forward_result(std::move(forward_results[i]));
      jobs[i]->step = Job::Step::UPDATE;
      policy_ptr = policy_ptr_end;
    }
    cpu_queue_.enqueue_bulk(std::make_move_iterator(jobs.begin()), jobs.size());
    jobs.clear();
  }
}

}  // namespace czf::actor::worker
