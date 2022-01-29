#include "worker_manager.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <random>

#include "czf.pb.h"

namespace czf::actor::alphazero_worker {

WorkerOption WorkerManager::worker_option;

void WorkerManager::run(size_t num_cpu_worker, size_t num_gpu_worker, size_t num_gpu) {
  if (running_) return;
  running_ = true;

  std::random_device rd;
  cpu_threads_.reserve(num_cpu_worker);
  for (int i = 0; i < num_cpu_worker; ++i) {
    auto seed = rd();
    cpu_threads_.emplace_back(&WorkerManager::cpu_worker, this, seed);
  }
  gpu_threads_.reserve(num_gpu_worker);
  for (int i = 0; i < num_gpu_worker; ++i) {
    auto seed = rd();
    gpu_threads_.emplace_back(&WorkerManager::gpu_worker, this, seed);
  }
  model_manager.resize(num_gpu);
}

void WorkerManager::terminate() {
  if (!running_) return;
  running_ = false;

  for (const auto& thread : cpu_threads_) cpu_queue_.enqueue(nullptr);
  result_queue_.enqueue(nullptr);

  for (auto& thread : cpu_threads_) thread.join();
  for (auto& thread : gpu_threads_) thread.join();
}

std::tuple<py::bytes, std::string, int> WorkerManager::enqueue_job_batch(const std::string& raw) {
  czf::pb::Packet packet;
  packet.ParseFromString(raw);
  auto* jobs_pb = packet.mutable_job_batch()->mutable_jobs();
  std::string model_name;
  int model_version = -1;
  for (auto& job_pb : *jobs_pb) {
    // TODO(chengscott): job.workers[job.step]
    job_pb.set_step(job_pb.step() + 1);
    model_name = job_pb.model().name();
    model_version = std::max(model_version, job_pb.model().version());
    if (!job_pb.has_payload()) {  // special job: flush model
      std::string packet_str;
      packet.SerializeToString(&packet_str);
      py::gil_scoped_acquire acquire;
      return {py::bytes(packet_str), model_name, model_version};
    }
    auto job = std::make_unique<Job>();
    const auto& state = job_pb.payload().state();
    job->root_state = game->deserialize_state(state.serialized_state());
    std::string job_str;
    job_pb.SerializeToString(&job_str);
    job->job_str = std::move(job_str);
    cpu_queue_.enqueue(std::move(job));
  }
  py::gil_scoped_acquire acquire;
  return {{}, model_name, model_version};
}

py::bytes WorkerManager::wait_dequeue_result(size_t max_batch_size) {
  constexpr auto zero = std::chrono::duration<double>::zero();
  const auto max_timeout = std::chrono::microseconds(WorkerManager::worker_option.timeout_us);
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
  auto* job_batch = packet.mutable_job_batch();
  for (const auto& job : jobs) {
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

void WorkerManager::load_from_bytes(const std::string& bytes) {
  model_manager.load_from_bytes(bytes);
}

void WorkerManager::load_from_file(const std::string& path) { model_manager.load_from_file(path); }

void WorkerManager::load_game(const std::string& name) {
  game = czf::env::czf_env::load_game(name);
}

void WorkerManager::cpu_worker(uint32_t seed) {
  std::mt19937 rng{seed};

  while (running_) {
    std::unique_ptr<Job> job;
    cpu_queue_.wait_dequeue(job);
    if (job == nullptr) return;

    bool next_job = false;
    while (!next_job) {
      switch (job->next_step) {
        case Job::Step::kEnqueue:
          job->preprocess();
          break;
        case Job::Step::kSelect:
          job->select(rng);
          break;
        case Job::Step::kEvaluate:
          gpu_queue_.enqueue(std::move(job));
          next_job = true;
          break;
        case Job::Step::kUpdate:
          job->update(rng);
          break;
        case Job::Step::kDone:
          job->postprocess(game->num_distinct_actions());
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

void WorkerManager::gpu_worker(uint32_t seed) {
  std::mt19937 rng{seed};
  const auto max_timeout = std::chrono::microseconds(WorkerManager::worker_option.timeout_us);

  const auto& observation_tensor_shape = game->observation_tensor_shape();
  std::vector<int64_t> input_shape(observation_tensor_shape.begin(),
                                   observation_tensor_shape.end());
  // add batch dimension (CHW -> BCHW)
  input_shape.insert(input_shape.begin(), 1);

  // create a vector containing all transformations
  int num_transformations = game->num_transformations();
  std::vector<int> transformations;
  transformations.reserve(num_transformations);
  for (int i = 0; i < num_transformations; ++i) transformations.push_back(i);

  while (running_) {
    // collect jobs
    Clock_t::time_point deadline;
    Clock_t::duration timeout = max_timeout;
    std::vector<std::unique_ptr<Job>> jobs;
    jobs.reserve(WorkerManager::worker_option.batch_size);

    while (jobs.size() < jobs.capacity() && timeout > Clock_t::duration::zero() && running_) {
      const int count = gpu_queue_.wait_dequeue_bulk_timed(std::back_inserter(jobs),
                                                           jobs.capacity() - jobs.size(), timeout);
      // first dequeue or didn't get any job
      if (count == jobs.size()) deadline = Clock_t::now() + max_timeout;
      timeout = deadline - Clock_t::now();
    }
    if (jobs.empty()) continue;

    // calculate input shape & size
    int batch_per_job = WorkerManager::worker_option.num_sampled_transformations;
    if (batch_per_job == 0) batch_per_job = 1;
    const int batch_size = jobs.size() * batch_per_job;
    input_shape[0] = batch_size;
    const int input_size =
        std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>());

    std::vector<float> input_vector;
    input_vector.reserve(input_size);
    std::vector<std::vector<int>> batch_transformations;
    batch_transformations.reserve(jobs.size());

    // construct input tensor, record sampled transformations for each job
    for (int i = 0; i < jobs.size(); ++i) {
      std::vector<int> sampled_transformations;
      if (WorkerManager::worker_option.num_sampled_transformations == 0) {
        sampled_transformations.push_back(0);
      } else {
        std::sample(transformations.begin(), transformations.end(),
                    std::back_inserter(sampled_transformations),
                    WorkerManager::worker_option.num_sampled_transformations, rng);
      }
      batch_transformations.push_back(sampled_transformations);

      for (int type : sampled_transformations) {
        std::vector<float> transformed_observation =
            game->transform_observation(jobs[i]->leaf_observation, type);
        input_vector.insert(input_vector.end(),
                            std::make_move_iterator(transformed_observation.begin()),
                            std::make_move_iterator(transformed_observation.end()));
      }
    }
    auto [device, model_ptr] = model_manager.get();
    auto input_tensor = torch::from_blob(input_vector.data(), input_shape).to(device);

    // inference
    const auto results = model_ptr->forward({input_tensor}).toTuple()->elements();
    const auto batch_policy = results[0].toTensor().cpu().contiguous();
    const auto batch_value = results[1].toTensor().cpu().contiguous();

    const auto policy_size = batch_policy[0].numel();
    const auto value_size = batch_value[0].numel();

    auto policy_ptr_begin = batch_policy.data_ptr<float>();
    auto value_ptr_begin = batch_value.data_ptr<float>();

    std::vector<float> average_policy(policy_size);
    std::vector<float> average_value(value_size);
    // copy results back to jobs
    for (int i = 0; i < jobs.size(); ++i) {
      std::fill(average_policy.begin(), average_policy.end(), 0);
      std::fill(average_value.begin(), average_value.end(), 0);

      for (int type : batch_transformations[i]) {
        const auto policy_ptr_end = policy_ptr_begin + policy_size;
        const auto value_ptr_end = value_ptr_begin + value_size;

        std::vector<float> policy(policy_ptr_begin, policy_ptr_end);
        std::vector<float> restored_policy = game->restore_policy(policy, type);

        // sum all policies and values
        std::transform(restored_policy.begin(), restored_policy.end(), average_policy.begin(),
                       average_policy.begin(), std::plus<float>());
        std::transform(value_ptr_begin, value_ptr_end, average_value.begin(),
                       average_value.begin(), std::plus<float>());

        policy_ptr_begin = std::move(policy_ptr_end);
        value_ptr_begin = std::move(value_ptr_end);
      }

      for (auto& p : average_policy) p /= batch_per_job;
      for (auto& v : average_value) v /= batch_per_job;
      jobs[i]->leaf_policy.assign(average_policy.begin(), average_policy.end());
      jobs[i]->leaf_returns.assign(average_value.begin(), average_value.end());
      jobs[i]->next_step = Job::Step::kUpdate;
    }

    cpu_queue_.enqueue_bulk(std::make_move_iterator(jobs.begin()), jobs.size());
  }
}

}  // namespace czf::actor::alphazero_worker
