#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "worker/worker.h"

namespace czf {
namespace {

namespace py = ::pybind11;
using py::literals::operator""_a;

PYBIND11_MODULE(worker, m) {  // NOLINT
  using czf::actor::GameInfo;
  py::class_<GameInfo>(m, "GameInfo")
      .def(py::init<>())
      .def_readwrite("observation_shape", &GameInfo::observation_shape)
      .def_readwrite("state_shape", &GameInfo::state_shape)
      .def_readwrite("all_actions", &GameInfo::all_actions)
      .def_readwrite("num_actions", &GameInfo::num_actions)
      .def_readwrite("two_player", &GameInfo::is_two_player);
  using czf::actor::JobOption;
  py::class_<JobOption>(m, "JobOption")
      .def(py::init<>())
      .def_readwrite("seed", &JobOption::seed)
      .def_readwrite("timeout_us", &JobOption::timeout_us)
      .def_readwrite("batch_size", &JobOption::batch_size)
      .def_readwrite("simulation_count", &JobOption::simulation_count);
  using czf::actor::MctsOption;
  py::class_<MctsOption>(m, "MctsOption")
      .def(py::init<>())
      .def_readwrite("C_PUCT", &MctsOption::C_PUCT)
      .def_readwrite("dirichlet_alpha", &MctsOption::dirichlet_alpha)
      .def_readwrite("dirichlet_epsilon", &MctsOption::dirichlet_epsilon)
      .def_readwrite("discount", &MctsOption::discount);
  using czf::actor::worker::WorkerManager;
  py::class_<WorkerManager>(m, "WorkerManager")
      .def(py::init<>())
      .def("run", &WorkerManager::run, "num_cpu_worker"_a = 1,
           "num_gpu_worker"_a = 1, "num_gpu_root_worker"_a = 1, "num_gpu"_a = 1)
      .def("terminate", &WorkerManager::terminate)
      .def("enqueue_job", &WorkerManager::enqueue_job, "job"_a, "observation"_a,
           "legal_actions"_a)
      .def("wait_dequeue_result", &WorkerManager::wait_dequeue_result,
           py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::move)
      .def("load_model", &WorkerManager::load_model, "path"_a)
      .def_readwrite_static("job_option", &WorkerManager::job_option)
      .def_readwrite_static("game_info", &WorkerManager::game_info)
      .def_readwrite_static("mcts_option", &WorkerManager::mcts_option);
}

}  // namespace
}  // namespace czf
