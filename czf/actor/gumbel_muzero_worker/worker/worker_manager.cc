#include "worker_manager.h"

#include <pybind11/stl.h>

#include <numeric>

#include "czf.pb.h"  // NOLINT
#include "utils/config.h"

namespace czf::actor::gumbel_muzero_worker::worker {

WorkerOption WorkerManager::worker_option;
GameInfo WorkerManager::game_info;

void WorkerManager::run(size_t num_cpu_worker, size_t num_gpu_worker, size_t num_gpu_root_worker,
                        size_t num_gpu) {
  if (!running_) {
    std::cout << "#CPU Worker: " << num_cpu_worker << std::endl;
    std::cout << "#GPU Root Worker: " << num_gpu_root_worker << std::endl;
    std::cout << "#GPU State Worker: " << num_gpu_worker << std::endl;
    running_ = true;
    const auto seed = worker_option.seed;
    for (size_t i = 0; i < num_cpu_worker; ++i) {
      const Seed_t stream = 100U + i;
      cpu_threads_.emplace_back(&WorkerManager::cpu_worker, this, seed, stream);
    }
    for (size_t i = 0; i < num_gpu_worker; ++i) {
      gpu_threads_.emplace_back(&WorkerManager::gpu_worker, this, GpuWorkerType::kWorkerState);
    }
    for (size_t i = 0; i < num_gpu_root_worker; ++i) {
      gpu_threads_.emplace_back(&WorkerManager::gpu_worker, this, GpuWorkerType::kWorkerRoot);
    }
    if (game_info.is_stochastic) {
      std::cout << "#GPU Afterstate Worker: " << num_gpu_worker << std::endl;
      for (size_t i = 0; i < num_gpu_worker; ++i) {
        gpu_threads_.emplace_back(&WorkerManager::gpu_worker, this,
                                  GpuWorkerType::kWorkerAfterstate);
      }
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
  }
}

std::tuple<py::bytes, std::string, int> WorkerManager::enqueue_job_batch(const std::string &raw) {
  czf::pb::Packet packet;
  packet.ParseFromString(raw);
  auto *jobs_pb = packet.mutable_job_batch()->mutable_jobs();
  std::string model_name;
  int model_version = -1;
  for (auto &job_pb : *jobs_pb) {
    // TODO(chengscott): job.workers[job.step]
    job_pb.set_step(job_pb.step() + 1);
    model_name = job_pb.model().name();
    model_version = std::max(model_version, job_pb.model().version());
    if (!job_pb.has_payload()) {
      // special job: flush model (without payload)
      std::string packet_str;
      packet.SerializeToString(&packet_str);
      py::gil_scoped_acquire acquire;
      return {py::bytes(packet_str), model_name, model_version};
    }
    std::string job_str;
    job_pb.SerializeToString(&job_str);
    auto job = std::make_unique<Job>();
    job->job_str = std::move(job_str);
    cpu_queue_.enqueue(std::move(job));
  }
  py::gil_scoped_acquire acquire;
  return {{}, model_name, model_version};
}

py::bytes WorkerManager::dequeue_job_batch(size_t max_batch_size) {
  constexpr auto zero = std::chrono::duration<double>::zero();
  const auto max_timeout = std::chrono::microseconds(worker_option.timeout_us);
  // collect result jobs
  std::vector<std::unique_ptr<Job>> jobs;
  jobs.reserve(max_batch_size);
  result_queue_.wait_dequeue_bulk(std::back_inserter(jobs), max_batch_size);
  Clock_t::time_point deadline = Clock_t::now() + max_timeout;
  for (Clock_t::duration timeout = max_timeout; jobs.size() < max_batch_size && timeout > zero;
       timeout = deadline - Clock_t::now()) {
    result_queue_.wait_dequeue_bulk_timed(std::back_inserter(jobs), max_batch_size - jobs.size(),
                                          timeout);
  }
  // collect tree results from jobs
  czf::pb::Packet packet;
  auto *job_batch = packet.mutable_job_batch();
  for (const auto &job : jobs) {
    if (job == nullptr) {
      return {};
    }
    job_batch->add_jobs()->ParseFromString(job->job_str);
  }
  std::string packet_str;
  packet.SerializeToString(&packet_str);
  py::gil_scoped_acquire acquire;
  return py::bytes(packet_str);
}

void WorkerManager::load_from_bytes(const std::string &bytes) {
  model_manager.load_from_bytes(bytes);
}

void WorkerManager::load_from_file(const std::string &path) { model_manager.load_from_file(path); }

void WorkerManager::preprocess_pb_job(std::unique_ptr<Job> &job) {
  czf::pb::Job job_pb;
  job_pb.ParseFromString(job->job_str);
  const auto &state = job_pb.payload().state();
  const auto &obs_tensor = state.observation_tensor();
  std::vector<float> observation{obs_tensor.begin(), obs_tensor.end()};
  const auto &actions = state.legal_actions();
  std::vector<int32_t> legal_actions{actions.begin(), actions.end()};
  const auto &option = state.tree_option();
  TreeOption tree_option{static_cast<size_t>(option.simulation_count()),
                         option.tree_min_value(),
                         option.tree_max_value(),
                         option.c_puct(),
                         option.dirichlet_alpha(),
                         option.dirichlet_epsilon(),
                         option.discount(),
                         static_cast<size_t>(option.gumbel_sampled_actions()),
                         option.gumbel_c_visit(),
                         option.gumbel_c_scale(),
                         option.gumbel_use_noise(),
                         option.gumbel_use_best_action_value(),
                         option.gumbel_use_simple_loss()};
  job_pb.mutable_payload()->mutable_state()->clear_observation_tensor();
  // worker job
  job_pb.SerializeToString(&job->job_str);
  job->tree.set_option(tree_option, game_info.is_two_player, game_info.is_stochastic);
  job->tree.set_forward_result({std::move(observation), {}, 0, 0});
  job->tree.expand_root(legal_actions);
}

void WorkerManager::postprocess_pb_job(std::unique_ptr<Job> &job) {
  const auto &result = job->tree.get_tree_result();
  czf::pb::Job job_pb;
  job_pb.ParseFromString(job->job_str);
  auto *evaluation = job_pb.mutable_payload()->mutable_state()->mutable_evaluation();
  evaluation->set_value(result.value);
  for (size_t i = 0; i < game_info.num_actions; ++i) {
    evaluation->add_policy(0.F);
  }
  // auto total = static_cast<float>(result.total_visits);
  for (auto &&[action, policy] : result.improved_policy) {
    evaluation->set_policy(action, policy);
  }
  evaluation->set_selected_action(result.selected_action);
  job_pb.SerializeToString(&job->job_str);
}

void WorkerManager::cpu_worker(Seed_t seed, Seed_t stream) {
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
        case Job::Step::kEnqueue:
          preprocess_pb_job(job);
          job->step = Job::Step::kForwardRoot;
          break;
        case Job::Step::kGumbelInit:
          // std::cout << "step: kGumbelInit" << std::endl;
          job->tree.gumbel_init(game_info.num_actions, rng);
          job->step = Job::Step::kGumbelSearch;
          break;
        case Job::Step::kGumbelSearch: {
          // std::cout << "step: kGumbelSearch" << std::endl;
          size_t status =
              job->tree.gumbel_search(game_info.all_actions, game_info.all_chance_outcomes);
          if (status == 0)
            job->step = Job::Step::kDone;
          else if (status == 1)
            job->step = Job::Step::kForwardAfterstate;
          else if (status == 2)
            job->step = Job::Step::kSelect;
          else if (status == 3)
            job->step = Job::Step::kGumbelSearch;
          break;
        }
        case Job::Step::kSelect:
          // std::cout << "Select" << std::endl;
          job->step =
              job->tree.before_forward(game_info.all_actions, game_info.all_chance_outcomes)
                  ? Job::Step::kForwardAfterstate
                  : Job::Step::kForwardState;
          break;
        case Job::Step::kUpdateRoot:
          // std::cout << "step: kUpdateRoot" << std::endl;
          job->tree.after_forward(rng);
          job->step = Job::Step::kGumbelInit;
          break;
        case Job::Step::kUpdate:
          // std::cout << "Update" << std::endl;
          job->tree.after_forward(rng);
          job->step = Job::Step::kGumbelSearch;
          break;
        case Job::Step::kForwardRoot:
          // std::cout << "ForwardRoot" << std::endl;
          gpu_root_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::kForwardState:
          // std::cout << "ForwardState" << std::endl;
          gpu_state_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::kForwardAfterstate:
          // std::cout << "ForwardAfterstate" << std::endl;
          gpu_afterstate_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::kDone:
          // std::cout << "Done" << std::endl;
          postprocess_pb_job(job);
          job->step = Job::Step::kDequeue;
          break;
        case Job::Step::kDequeue:
          result_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        default:
          break;
      }
    }
  }
}

void WorkerManager::gpu_worker(GpuWorkerType worker_type) {
  // jobs
  auto &queue = (worker_type == GpuWorkerType::kWorkerRoot)
                    ? gpu_root_queue_
                    : (worker_type == GpuWorkerType::kWorkerState) ? gpu_state_queue_
                                                                   : gpu_afterstate_queue_;
  constexpr auto zero = std::chrono::duration<double>::zero();
  const auto max_timeout = std::chrono::microseconds(worker_option.timeout_us);
  const auto max_batch_size = worker_option.batch_size;
  std::vector<std::unique_ptr<Job>> jobs;
  jobs.reserve(max_batch_size);
  std::vector<czf::actor::gumbel_muzero_worker::mcts::ForwardResult> forward_results;
  forward_results.reserve(max_batch_size);
  // inputs (NCHW)
  std::vector<int64_t> state_shape = (worker_type == GpuWorkerType::kWorkerRoot)
                                         ? game_info.observation_shape
                                         : game_info.state_shape;
  state_shape.insert(state_shape.begin(), static_cast<int64_t>(max_batch_size));
  std::vector<float> state_vector;
  state_vector.reserve(
      std::accumulate(state_shape.begin(), state_shape.end(), 1U, std::multiplies<>()));
  std::vector<float> action_vector;
  action_vector.reserve(max_batch_size);
  while (running_) {
    // collect jobs
    Clock_t::time_point deadline;
    for (Clock_t::duration timeout = max_timeout;
         jobs.size() < max_batch_size && timeout > zero && running_;
         timeout = deadline - Clock_t::now()) {
      auto count = queue.wait_dequeue_bulk_timed(std::back_inserter(jobs),
                                                 max_batch_size - jobs.size(), timeout);
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
      state_vector.insert(state_vector.end(), info.state.begin(), info.state.end());
      action_vector.push_back(info.action);
    }
    auto [device, model_ptr] = model_manager.get();
    auto state_tensor = torch::from_blob(state_vector.data(), state_shape).to(device);
    auto action_tensor =
        torch::from_blob(action_vector.data(), {static_cast<int64_t>(batch_size), 1}).to(device);
    torch::Tensor next_state_tensor;
    if (worker_type == GpuWorkerType::kWorkerRoot) {
      // representation function
      next_state_tensor =
          model_ptr->get_method("forward_representation")({state_tensor}).toTensor();
    } else if (worker_type == GpuWorkerType::kWorkerState) {
      // dynamics function
      auto results = model_ptr->get_method("forward_dynamics")({state_tensor, action_tensor})
                         .toTuple()
                         ->elements();
      next_state_tensor = results[0].toTensor();
      const auto batch_reward = results[1].toTensor().cpu().contiguous();
      auto *const reward_ptr = batch_reward.data_ptr<float>();
      for (size_t i = 0; i < batch_size; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        forward_results[i].reward = reward_ptr[i];
      }
    } else {
      // afterstate_dynamics function
      next_state_tensor =
          model_ptr->get_method("forward_afterstate_dynamics")({state_tensor, action_tensor})
              .toTensor();
      for (size_t i = 0; i < batch_size; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        forward_results[i].reward = 0.F;
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
        (worker_type == GpuWorkerType::kWorkerAfterstate)
            ? model_ptr->get_method("forward_afterstate_prediction")({next_state_tensor})
                  .toTuple()
                  ->elements()
            : model_ptr->forward({next_state_tensor}).toTuple()->elements();
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
      jobs[i]->step = (worker_type == GpuWorkerType::kWorkerRoot) ? Job::Step::kUpdateRoot
                                                                  : Job::Step::kUpdate;
      policy_ptr = policy_ptr_end;
    }
    cpu_queue_.enqueue_bulk(std::make_move_iterator(jobs.begin()), jobs.size());
    jobs.clear();
  }
}

}  // namespace czf::actor::gumbel_muzero_worker::worker
